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
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path


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
            text_emb = model.encode_text(
                [clean_text],
                task='retrieval',
                prompt_name='query'
            )
            # encode_text returns a list of embeddings
            # Each element is a tensor on GPU
            if isinstance(text_emb, list):
                if len(text_emb) > 0:
                    emb = text_emb[0]
                    if hasattr(emb, 'cpu'):
                        text_emb = emb.detach().cpu().float()
                    else:
                        text_emb = torch.tensor(emb).float()
                else:
                    raise ValueError("Empty embedding list returned")
            elif hasattr(text_emb, 'cpu'):
                text_emb = text_emb.detach().cpu().float().squeeze(0)
            else:
                text_emb = torch.tensor(text_emb).float().squeeze(0)
            embeddings.append(text_emb)

        
        # Encode images (if supported and available)
        if img_paths and hasattr(model, 'encode_image'):
            valid_images = [p for p in img_paths if os.path.exists(p)]
            if valid_images:
                try:
                    img_emb = model.encode_image(valid_images[:2])  # Limit to 2 images
                    if isinstance(img_emb, torch.Tensor):
                        img_emb = img_emb.detach().cpu()
                    elif hasattr(img_emb, 'cpu'):
                        img_emb = img_emb.detach().cpu()
                    else:
                        img_emb = torch.from_numpy(np.array(img_emb)).float()
                    # Average image embeddings
                    if img_emb.dim() > 1:
                        img_emb = img_emb.mean(dim=0)
                    embeddings.append(img_emb.float())
                except Exception as e:
                    print(f"[WARNING] Image encoding failed: {e}")

        
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
