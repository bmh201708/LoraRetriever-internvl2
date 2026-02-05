"""
LoRA Retriever: Embedding-based LoRA retrieval

Based on LoraRetriever paper Section 4.1:
- Each LoRA is represented by the mean embedding of m training samples
- Cosine similarity is used for matching query to LoRAs
- Top-k LoRAs are selected for composition
"""

import os
import json
import numpy as np
import torch
import inspect
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path


def _call_with_supported_kwargs(fn, **kwargs):
    """
    智能过滤参数：只传递函数签名支持的参数
    避免不同版本 jina-embeddings-v4 API 差异导致的错误
    """
    try:
        sig = inspect.signature(fn)
        allowed = set(sig.parameters.keys())
    except Exception:
        allowed = set(kwargs.keys())
    passed = {k: v for k, v in kwargs.items() if (k in allowed and v is not None)}
    return fn(**passed)


def _to_numpy_embeddings(out: Any, debug: bool = False) -> np.ndarray:
    """
    稳健地将 jina-embeddings-v4 的返回值转为 numpy 数组
    支持多种返回类型：Tensor, ndarray, list, dict 等
    """
    # 检查是否是函数（这不应该发生）
    if callable(out) and not isinstance(out, (torch.Tensor, np.ndarray)):
        raise TypeError(f"encode_* returned a callable object (type={type(out)}), expected embeddings. "
                       f"This may indicate an API mismatch.")

    if isinstance(out, torch.Tensor):
        return out.detach().float().cpu().numpy().astype(np.float32, copy=False)
    if isinstance(out, np.ndarray):
        return out.astype(np.float32, copy=False)
    if isinstance(out, dict):
        # 尝试常见的 key
        for k in ["embeddings", "embedding", "sentence_embedding", "sentence_embeddings", "vectors", "vector"]:
            if k in out:
                return _to_numpy_embeddings(out[k], debug=debug)
        # 兜底：取第一个 value
        if out:
            return _to_numpy_embeddings(next(iter(out.values())), debug=debug)
    if isinstance(out, (list, tuple)):
        if len(out) == 0:
            return np.zeros((0, 0), dtype=np.float32)
        # list[Tensor] / list[np.ndarray]
        if all(isinstance(x, torch.Tensor) for x in out):
            t = torch.stack(list(out), dim=0)
            return t.detach().float().cpu().numpy().astype(np.float32, copy=False)
        if all(isinstance(x, np.ndarray) for x in out):
            return np.stack(list(out), axis=0).astype(np.float32, copy=False)
        # 混合类型：递归转
        arrs = [_to_numpy_embeddings(x, debug=debug) for x in out]
        return np.concatenate(arrs, axis=0) if (arrs and arrs[0].ndim == 2) else np.asarray(arrs, dtype=np.float32)

    # 最后兜底
    try:
        return np.asarray(out, dtype=np.float32)
    except TypeError:
        if hasattr(out, "detach") and hasattr(out, "cpu"):
            try:
                t = out.detach().cpu()
                if isinstance(t, torch.Tensor):
                    return t.float().numpy().astype(np.float32, copy=False)
            except Exception:
                pass
        raise TypeError(f"Cannot convert {type(out)} to numpy array")


@dataclass
class LoraRetrieverConfig:
    """Configuration for LoraRetriever"""
    lora_configs: List[Dict[str, str]] = field(default_factory=list)
    model_path: str = "/home/hmpiao/hmpiao/jina-embeddings-v4"
    device: str = "cuda"
    dtype: str = "float16"
    top_k: int = 3
    normalize_embeddings: bool = True
    

class LoraRetriever:
    """
    Embedding-based LoRA Retriever
    
    Implements the retrieval mechanism from LoraRetriever paper:
    - Uses pre-computed LoRA embeddings (mean of training samples)
    - Computes query embedding using jina-embeddings-v4
    - Retrieves top-k LoRAs by cosine similarity
    
    Args:
        config: LoraRetrieverConfig with LoRA configurations
    """
    
    def __init__(self, config: LoraRetrieverConfig):
        self.config = config
        self.device = config.device
        self.top_k = config.top_k
        
        # LoRA embeddings: {lora_name: embedding_tensor}
        self.lora_embeddings: Dict[str, torch.Tensor] = {}
        self.lora_names: List[str] = []
        
        # Embedding model (lazy loading)
        self._model = None
        
    def load_lora_embeddings(self, lora_configs: Optional[List[Dict]] = None) -> None:
        """
        Load pre-computed LoRA embeddings from .npy files
        
        Args:
            lora_configs: List of dicts with 'lora_name' and 'embedding_path'
        """
        configs = lora_configs or self.config.lora_configs
        
        for cfg in configs:
            lora_name = cfg.get('lora_name')
            embedding_path = cfg.get('embedding_path')
            
            if not lora_name:
                continue
                
            if embedding_path and os.path.exists(embedding_path):
                try:
                    embedding = np.load(embedding_path)
                    self.lora_embeddings[lora_name] = torch.from_numpy(embedding).float()
                    self.lora_names.append(lora_name)
                except Exception as e:
                    print(f"[WARNING] Failed to load embedding for {lora_name}: {e}")
            else:
                print(f"[WARNING] No embedding file for {lora_name}: {embedding_path}")
        
        print(f"[INFO] Loaded {len(self.lora_embeddings)} LoRA embeddings")
    
    def _get_model(self):
        """Lazy load the jina embedding model"""
        if self._model is None:
            from transformers import AutoModel

            # Monkey patch for PyTorch 2.5.1 + Triton 3.1.0 compatibility
            # flash_attn 需要 wrap_triton 返回支持 [grid] 语法的对象
            try:
                if not hasattr(torch.library, 'wrap_triton'):
                    import warnings
                    warnings.filterwarnings('ignore', message='.*wrap_triton.*')

                    class _TritonKernelWrapper:
                        """包装 Triton kernel 以支持 kernel[grid](...) 语法"""
                        def __init__(self, kernel):
                            self.kernel = kernel

                        def __getitem__(self, grid):
                            """支持 kernel[grid] 语法，返回真正的 Triton launcher"""
                            # 直接返回 Triton kernel 的 grid launcher
                            # 这样 Triton 可以在正确的上下文中运行
                            return self.kernel[grid]

                        def __call__(self, *args, **kwargs):
                            """支持直接调用"""
                            return self.kernel(*args, **kwargs)

                    def _dummy_wrap_triton(kernel):
                        """返回支持索引的 wrapper"""
                        return _TritonKernelWrapper(kernel)

                    torch.library.wrap_triton = _dummy_wrap_triton
            except Exception as e:
                print(f"[DEBUG] Monkey patch warning (non-critical): {e}")

            print(f"[INFO] Loading jina-embeddings-v4 from {self.config.model_path}")
            
            dtype_map = {
                'float16': torch.float16,
                'bfloat16': torch.bfloat16,
                'float32': torch.float32
            }
            torch_dtype = dtype_map.get(self.config.dtype, torch.float16)
            
            self._model = AutoModel.from_pretrained(
                self.config.model_path,
                trust_remote_code=True,
                torch_dtype=torch_dtype
            ).to(self.device)
            self._model.eval()
            
        return self._model
    
    def compute_query_embedding(
        self, 
        query: Union[str, Dict[str, Any]],
        images: Optional[List[str]] = None
    ) -> torch.Tensor:
        """
        Compute embedding for a query
        
        Args:
            query: Text query or dict with 'text' and optional 'images'
            images: Optional list of image paths
            
        Returns:
            Query embedding tensor (normalized if config.normalize_embeddings)
        """
        model = self._get_model()
        
        # Parse query
        if isinstance(query, str):
            text = query
            img_paths = images or []
        else:
            text = query.get('text', query.get('query', ''))
            img_paths = query.get('images', [])
        
        # Clean text (remove <image> placeholders)
        text_lines = text.split('\n')
        text_lines = [l for l in text_lines if l.strip() != '<image>']
        clean_text = ' '.join(text_lines).strip()
        
        embeddings = []
        
        # Encode text
        if clean_text:
            with torch.inference_mode():
                # 直接调用，传递已知的参数
                text_out = model.encode_text(
                    [clean_text],
                    task='retrieval',
                    prompt_name='query'
                )
            # 稳健转换为 numpy
            text_emb_np = _to_numpy_embeddings(text_out)
            # 清理中间变量，释放显存
            del text_out
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # 如果是二维的，取第一行
            if text_emb_np.ndim == 2:
                text_emb_np = text_emb_np[0]
            text_emb = torch.from_numpy(text_emb_np).float()
            embeddings.append(text_emb)

        
        # Encode images (if supported and available)
        if img_paths and hasattr(model, 'encode_image'):
            valid_images = [p for p in img_paths if os.path.exists(p)]
            if valid_images:
                try:
                    # 限制最多2张图片，降低显存压力
                    img_list = valid_images[:2]

                    # 从环境变量读取配置，允许用户调整
                    img_batch_size = int(os.environ.get('JINA_IMAGE_BATCH_SIZE', '1'))
                    img_max_pixels = int(os.environ.get('JINA_MAX_PIXELS', '100000'))

                    with torch.inference_mode():
                        # 直接调用，传递已知的参数
                        # 根据签名: (images, task=None, batch_size=8, return_multivector=False,
                        #            return_numpy=False, truncate_dim=None, max_pixels=None)
                        img_out = model.encode_image(
                            images=img_list,
                            task='retrieval',
                            batch_size=img_batch_size,  # 一次处理1张图，避免 OOM
                            max_pixels=img_max_pixels  # 降低分辨率（约 316x316），节省显存
                        )

                    # 稳健转换为 numpy
                    img_emb_np = _to_numpy_embeddings(img_out, debug=False)
                    # 清理中间变量，释放显存
                    del img_out
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    # 如果有多张图，取平均
                    if img_emb_np.ndim == 2 and img_emb_np.shape[0] > 1:
                        img_emb_np = img_emb_np.mean(axis=0)
                    elif img_emb_np.ndim == 2:
                        img_emb_np = img_emb_np[0]
                    img_emb = torch.from_numpy(img_emb_np).float()
                    embeddings.append(img_emb)
                except Exception as e:
                    import traceback
                    print(f"[WARNING] Image encoding failed: {e}")
                    print(f"[DEBUG] Full error:")
                    traceback.print_exc()
                finally:
                    # 确保清理显存
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

        
        # Combine embeddings (weighted average)
        if len(embeddings) == 0:
            raise ValueError("No valid input for embedding computation")
        elif len(embeddings) == 1:
            final_emb = embeddings[0]
        else:
            # Weight text more than images (0.7 vs 0.3)
            final_emb = 0.7 * embeddings[0] + 0.3 * embeddings[1]
        
        # Normalize
        if self.config.normalize_embeddings:
            final_emb = final_emb / (final_emb.norm() + 1e-8)
        
        return final_emb.float()
    
    def retrieve(
        self, 
        query: Union[str, Dict[str, Any]],
        images: Optional[List[str]] = None,
        top_k: Optional[int] = None,
        exclude_list: Optional[List[str]] = None
    ) -> Tuple[List[str], List[float]]:
        """
        Retrieve top-k LoRAs for a query
        
        Implements Equation from paper:
        g(x_i, Φ) := Φ_i = TopK{s(φ_j, x_i, I), φ_j ∈ Φ}
        
        Args:
            query: Text query or dict with 'text' and 'images'
            images: Optional list of image paths  
            top_k: Number of LoRAs to retrieve (default: self.top_k)
            exclude_list: LoRA names to exclude
            
        Returns:
            Tuple of (selected_lora_names, similarity_scores)
        """
        k = top_k or self.top_k
        exclude = set(exclude_list or [])
        
        if not self.lora_embeddings:
            raise ValueError("No LoRA embeddings loaded. Call load_lora_embeddings() first.")
        
        # Compute query embedding
        query_emb = self.compute_query_embedding(query, images)
        
        # Compute cosine similarity with all LoRA embeddings
        similarities = {}
        for lora_name, lora_emb in self.lora_embeddings.items():
            if lora_name in exclude:
                continue
            
            # Normalize LoRA embedding
            if self.config.normalize_embeddings:
                lora_emb_norm = lora_emb / (lora_emb.norm() + 1e-8)
            else:
                lora_emb_norm = lora_emb
            
            # Cosine similarity
            sim = torch.dot(query_emb, lora_emb_norm.float()).item()
            similarities[lora_name] = sim
        
        # Sort by similarity (descending) and get top-k
        sorted_items = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        top_k_items = sorted_items[:k]
        
        selected_names = [item[0] for item in top_k_items]
        scores = [item[1] for item in top_k_items]
        
        return selected_names, scores
    
    def retrieve_with_weights(
        self, 
        query: Union[str, Dict[str, Any]],
        images: Optional[List[str]] = None,
        top_k: Optional[int] = None,
        exclude_list: Optional[List[str]] = None,
        normalize_weights: bool = True
    ) -> Tuple[List[str], List[float]]:
        """
        Retrieve top-k LoRAs with normalized weights
        
        Implements Equation 5 from paper (weight normalization):
        w_j = s(x, φ_j) / Σ_{k∈TopK} s(x, φ_k)
        
        Args:
            query: Text query or dict
            images: Optional image paths
            top_k: Number of LoRAs
            exclude_list: LoRAs to exclude
            normalize_weights: Whether to normalize scores to sum to 1
            
        Returns:
            Tuple of (selected_lora_names, normalized_weights)
        """
        selected_names, scores = self.retrieve(query, images, top_k, exclude_list)
        
        if normalize_weights and scores:
            # Softmax-like normalization
            scores_array = np.array(scores)
            # Temperature scaling for sharper distribution
            scores_exp = np.exp(scores_array * 5.0)  # temperature = 0.2
            weights = (scores_exp / scores_exp.sum()).tolist()
        else:
            weights = scores
        
        return selected_names, weights
    
    def get_all_similarities(
        self, 
        query: Union[str, Dict[str, Any]],
        images: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Get similarity scores for all LoRAs
        
        Args:
            query: Text query or dict
            images: Optional image paths
            
        Returns:
            Dict mapping lora_name to similarity score
        """
        if not self.lora_embeddings:
            return {}
        
        query_emb = self.compute_query_embedding(query, images)
        
        similarities = {}
        for lora_name, lora_emb in self.lora_embeddings.items():
            if self.config.normalize_embeddings:
                lora_emb_norm = lora_emb / (lora_emb.norm() + 1e-8)
            else:
                lora_emb_norm = lora_emb
            
            sim = torch.dot(query_emb, lora_emb_norm.float()).item()
            similarities[lora_name] = sim
        
        return similarities


def load_retriever_from_config(config_paths: List[str], **kwargs) -> LoraRetriever:
    """
    Create LoraRetriever from config JSON files
    
    Args:
        config_paths: List of paths to config JSON files
        **kwargs: Additional config arguments
        
    Returns:
        Initialized LoraRetriever
    """
    lora_configs = []
    
    for path in config_paths:
        if not os.path.exists(path):
            continue
        with open(path, 'r') as f:
            configs = json.load(f)
            lora_configs.extend(configs)
    
    config = LoraRetrieverConfig(
        lora_configs=lora_configs,
        **kwargs
    )
    
    retriever = LoraRetriever(config)
    retriever.load_lora_embeddings()
    
    return retriever
