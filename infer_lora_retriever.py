#!/usr/bin/env python
"""
LoraRetriever Inference Script for InternVL2

Implements the LoraRetriever paper's retrieve-then-compose workflow:
1. Load base model and all LoRA adapters
2. For each input: retrieve top-k LoRAs using embedding similarity
3. Compose using Mixture (output-level) or Fusion (parameter-level)
4. Generate response

Usage:
    python infer_lora_retriever.py --test_data data/example.jsonl --top_k 3 --merge_method mixture
    python infer_lora_retriever.py --test_data data/Val_100.jsonl --top_k 5 --merge_method fusion
"""

import os
import sys
import argparse

# 在 import torch 之前解析 gpu_id 参数
def _parse_gpu_id():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--gpu_id', type=str, default='5')
    args, _ = parser.parse_known_args()
    return args.gpu_id

_gpu_id = _parse_gpu_id()
if _gpu_id is not None:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(_gpu_id)

# Set environment variables before importing Swift
if 'MAX_PIXELS' not in os.environ:
    os.environ['MAX_PIXELS'] = '100000'
if 'MAX_NUM' not in os.environ:
    os.environ['MAX_NUM'] = '12'
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import json
import datetime as dt
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from tqdm import tqdm

import torch

# Add project paths
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'swift'))

from swift.utils import get_logger, append_to_jsonl, seed_everything
from swift.llm.utils import (
    get_model_tokenizer,
    get_template,
    inference
)
from swift.tuners import Swift

from lora_retriever import (
    LoraRetriever,
    LoraRetrieverConfig,
    MixtureComposer,
    FusionComposer,
    CompositionStrategy
)

logger = get_logger()


# Default paths
MODEL_PATHS = {
    'internvl2-2b': '/home/hmpiao/hmpiao/InternVL2-2B-ModelScope/OpenGVLab/InternVL2-2B',
    'qwen2-vl-7b-instruct': '/home/hmpiao/hmpiao/Qwen2-VL-7B-Instruct',
}

CONFIG_PATHS = {
    'internvl2-2b': {
        'app': 'config/app_loras_config_internvl2.json',
        'category': 'config/category_loras_config_internvl2.json',
    },
    'qwen2-vl-7b-instruct': {
        'app': 'config/app_loras_config_qwen2vl.json',
        'category': 'config/category_loras_config_qwen2vl.json',
    },
}

JINA_MODEL_PATH = '/home/hmpiao/hmpiao/jina-embeddings-v4'


def parse_args():
    parser = argparse.ArgumentParser(description='LoraRetriever Inference for VLM')
    
    # Model settings
    parser.add_argument('--model_type', type=str, default='internvl2-2b',
                       choices=['internvl2-2b', 'qwen2-vl-7b-instruct'])
    parser.add_argument('--model_path', type=str, default=None)
    
    # Data paths
    parser.add_argument('--test_data', type=str, default='data/example.jsonl')
    parser.add_argument('--app_config', type=str, default=None)
    parser.add_argument('--category_config', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default='output/lora_retriever_results')
    
    # Retriever settings
    parser.add_argument('--jina_model', type=str, default=JINA_MODEL_PATH)
    parser.add_argument('--top_k', type=int, default=3)
    parser.add_argument('--lora_type', type=str, default='all',
                       choices=['all', 'app', 'category'])
    
    # Composition settings
    parser.add_argument('--merge_method', type=str, default='mixture',
                       choices=['mixture', 'fusion'],
                       help='mixture: output-level weighted sum (paper recommended), fusion: parameter-level averaging')
    parser.add_argument('--combination_type', type=str, default='linear',
                       choices=['linear', 'svd', 'cat'],
                       help='For fusion mode: how to combine LoRA parameters')
    
    # Generation settings
    parser.add_argument('--max_new_tokens', type=int, default=512)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--seed', type=int, default=42)
    
    # Debug settings
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--num_samples', type=int, default=None)
    parser.add_argument('--show_similarities', action='store_true',
                       help='Print similarity scores for all LoRAs')
    
    parser.add_argument('--gpu_id', type=str, default='5', help='GPU ID to use (e.g., "0")')

    args = parser.parse_args()
    # gpu_id 已在文件开头设置，此处不再重复设置
    return args


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    """Load JSONL dataset."""
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def load_lora_configs(config_paths: List[str]) -> List[Dict]:
    """Load LoRA configurations from JSON files."""
    configs = []
    for path in config_paths:
        if path == '/dev/null' or not os.path.exists(path):
            continue
        with open(path, 'r') as f:
            data = json.load(f)
            configs.extend(data)
    return configs


def resolve_image_paths(images: List[str], project_root: str) -> List[str]:
    """Resolve relative image paths to absolute paths."""
    resolved = []
    for img_path in images:
        if img_path.startswith('./../'):
            img_path = img_path.replace('./../', '')
            img_path = os.path.join(os.path.dirname(project_root), img_path)
        elif not os.path.isabs(img_path):
            img_path = os.path.join(project_root, img_path)
        
        if os.path.exists(img_path):
            resolved.append(img_path)
    return resolved


def prepare_query(query: str, num_images: int) -> str:
    """Adjust query to match number of available images."""
    lines = query.split('\n')
    non_image_lines = [l for l in lines if l != '<image>']
    return '\n'.join(['<image>'] * num_images + non_image_lines)


def get_template_type(model_type: str) -> str:
    """Get template type based on model type."""
    if 'qwen2-vl' in model_type:
        return 'qwen2-vl'
    elif 'internvl' in model_type:
        return 'internvl2'
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def main():
    args = parse_args()
    seed_everything(args.seed)
    
    # Set default paths
    if args.model_path is None:
        args.model_path = MODEL_PATHS.get(args.model_type)
    
    config = CONFIG_PATHS.get(args.model_type, CONFIG_PATHS['internvl2-2b'])
    if args.app_config is None:
        args.app_config = config['app']
    if args.category_config is None:
        args.category_config = config['category']
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    time_str = dt.datetime.now().strftime('%Y%m%d-%H%M%S')
    model_suffix = 'qwen2vl' if 'qwen2' in args.model_type else 'internvl2'
    output_path = os.path.join(
        args.output_dir,
        f'retriever_results_{model_suffix}_{args.merge_method}_k{args.top_k}_{time_str}.jsonl'
    )
    
    logger.info("=" * 60)
    logger.info("LoraRetriever Inference")
    logger.info("=" * 60)
    logger.info(f"Model: {args.model_type}")
    logger.info(f"Model Path: {args.model_path}")
    logger.info(f"Merge Method: {args.merge_method}")
    logger.info(f"Top-K: {args.top_k}")
    logger.info(f"Jina Model: {args.jina_model}")
    logger.info("=" * 60)
    
    # Load test data
    logger.info(f"Loading test data from: {args.test_data}")
    test_data = load_jsonl(args.test_data)
    logger.info(f"Loaded {len(test_data)} samples")
    
    if args.debug or args.num_samples:
        num_samples = args.num_samples or 5
        test_data = test_data[:num_samples]
        logger.info(f"Debug mode: processing {len(test_data)} samples")
    
    # Load LoRA configs
    config_paths = []
    if args.lora_type in ['all', 'app'] and os.path.exists(args.app_config):
        config_paths.append(args.app_config)
    if args.lora_type in ['all', 'category'] and os.path.exists(args.category_config):
        config_paths.append(args.category_config)
    
    lora_configs = load_lora_configs(config_paths)
    logger.info(f"Loaded {len(lora_configs)} LoRA configs")
    
    if not lora_configs:
        logger.error("No LoRA configs found!")
        return
    
    # =========================================================================
    # Initialize Retriever
    # =========================================================================
    logger.info("Initializing LoRA Retriever...")
    
    retriever_config = LoraRetrieverConfig(
        lora_configs=lora_configs,
        model_path=args.jina_model,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        top_k=args.top_k
    )
    retriever = LoraRetriever(retriever_config)
    retriever.load_lora_embeddings(lora_configs)
    
    logger.info(f"Retriever loaded {len(retriever.lora_embeddings)} LoRA embeddings")
    
    # =========================================================================
    # Load Base Model
    # =========================================================================
    logger.info(f"Loading base model: {args.model_type}")
    
    model_kwargs = {
        'device_map': 'auto',
        'low_cpu_mem_usage': True,
    }
    
    model, tokenizer = get_model_tokenizer(
        args.model_type,
        torch.float16,
        model_kwargs,
        model_id_or_path=args.model_path
    )
    
    template_type = get_template_type(args.model_type)
    template = get_template(
        template_type,
        tokenizer,
        None,
        2048,
        'truncation_left',
        model=model
    )
    
    # =========================================================================
    # Load All LoRA Adapters
    # =========================================================================
    logger.info("Loading all LoRA adapters...")
    adapter_names = []
    
    for cfg in lora_configs:
        lora_name = cfg['lora_name']
        lora_path = cfg['lora_path']
        
        if not os.path.exists(lora_path):
            logger.warning(f"LoRA not found: {lora_path}, skipping...")
            continue
        
        try:
            model = Swift.from_pretrained(
                model,
                lora_path,
                adapter_name=lora_name,
                inference_mode=True
            )
            adapter_names.append(lora_name)
            logger.info(f"  Loaded: {lora_name}")
        except Exception as e:
            logger.error(f"Failed to load {lora_name}: {e}")
    
    logger.info(f"Loaded {len(adapter_names)} LoRA adapters")
    
    # =========================================================================
    # Initialize Composer
    # =========================================================================
    if args.merge_method == 'mixture':
        composer = MixtureComposer(use_weighted_average=True)
    else:
        composer = FusionComposer(combination_type=args.combination_type)
    
    logger.info(f"Using {args.merge_method} composition strategy")
    
    # =========================================================================
    # Run Inference
    # =========================================================================
    logger.info("Starting LoraRetriever inference...")
    results = []
    project_root = str(PROJECT_ROOT)

    # 创建进度条，显示详细信息
    pbar = tqdm(test_data, desc="LoraRetriever Inference", ncols=120)

    for idx, sample in enumerate(pbar):
        query = sample.get('query', '')
        images = sample.get('images', [])
        label = sample.get('response', '')

        try:
            # 更新进度条：显示当前阶段
            pbar.set_description(f"[{idx+1}/{len(test_data)}] Retrieving LoRAs")

            # Resolve image paths
            resolved_images = resolve_image_paths(images, project_root)

            # =========================================================================
            # Step 1: Retrieve Top-K LoRAs
            # =========================================================================
            query_input = {
                'text': query,
                'images': resolved_images[:2]  # Use max 2 images for embedding
            }

            selected_loras, weights = retriever.retrieve_with_weights(
                query_input,
                top_k=args.top_k
            )

            # 更新进度条：显示选中的 LoRAs 和图片信息
            lora_display = ', '.join([f"{name.split('_')[-1]}" for name in selected_loras[:2]])
            img_info = f"{len(resolved_images)}imgs" if len(resolved_images) <= 10 else f"{len(resolved_images)}imgs(多!)"
            pbar.set_postfix_str(f"{img_info} | LoRAs: {lora_display}")
            
            if args.show_similarities:
                all_sims = retriever.get_all_similarities(query_input)
                sorted_sims = sorted(all_sims.items(), key=lambda x: x[1], reverse=True)
                logger.info(f"Sample {idx} similarities:")
                for name, sim in sorted_sims:
                    logger.info(f"  {name}: {sim:.4f}")
            
            # =========================================================================
            # Step 2: Compose LoRAs
            # =========================================================================
            # 更新进度条：显示推理阶段
            pbar.set_description(f"[{idx+1}/{len(test_data)}] Inferencing")

            # 限制推理时的图片数量，避免 OOM
            max_inference_images = int(os.environ.get('MAX_NUM', '12'))
            if len(resolved_images) > max_inference_images:
                logger.warning(f"Sample {idx}: {len(resolved_images)} 张图片超过限制，只使用前 {max_inference_images} 张")
                inference_images = resolved_images[:max_inference_images]
            else:
                inference_images = resolved_images

            adjusted_query = prepare_query(query, len(inference_images))

            if args.merge_method == 'mixture':
                # Create lora_mapping for mixture mode
                lora_mapping = composer.compose(
                    model=model,
                    adapter_names=selected_loras,
                    weights=weights,
                    all_adapter_names=adapter_names,
                    batch_size=1,
                    device=next(model.parameters()).device
                )
                
                # Update template model
                template.model = model
                
                # Run inference with mixture mode
                if inference_images:
                    response, _ = inference(
                        model,
                        template,
                        adjusted_query,
                        history=[],
                        system=None,
                        images=inference_images,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                        merging_type='mixture',
                        lora_mapping=lora_mapping,
                        mixture_adapter_names=adapter_names
                    )
                else:
                    response, _ = inference(
                        model,
                        template,
                        adjusted_query,
                        history=[],
                        system=None,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                        merging_type='mixture',
                        lora_mapping=lora_mapping,
                        mixture_adapter_names=adapter_names
                    )
            else:
                # Fusion mode: merge parameters
                import time as _time
                print(f"\n[INFER DEBUG] Sample {idx}: 开始 fusion compose...")
                _t_compose_start = _time.time()
                merged_name = composer.compose(
                    model=model,
                    adapter_names=selected_loras,
                    weights=weights
                )
                print(f"[INFER DEBUG] Sample {idx}: compose 完成，耗时 {_time.time() - _t_compose_start:.2f}s")

                # === 诊断：检查 active adapters 和参数状态 ===
                if hasattr(model, 'active_adapters'):
                    print(f"[INFER DEBUG] SwiftModel.active_adapters: {model.active_adapters}")
                from peft.tuners.lora import LoraLayer as _LoraLayer
                for _name, _mod in model.named_modules():
                    if isinstance(_mod, _LoraLayer):
                        _peft_active = getattr(_mod, 'active_adapters', None)
                        if _peft_active is None:
                            _peft_active = getattr(_mod, 'active_adapter', None)
                        print(f"[INFER DEBUG] 第一个LoRA层 ({_name}):")
                        print(f"[INFER DEBUG]   PEFT active_adapters: {_peft_active}")
                        print(f"[INFER DEBUG]   disable_adapters: {getattr(_mod, 'disable_adapters', 'N/A')}")
                        print(f"[INFER DEBUG]   merged: {getattr(_mod, 'merged', 'N/A')}")
                        # 检查 fused_lora 的参数状态
                        if 'fused_lora' in _mod.lora_A:
                            _fA = _mod.lora_A['fused_lora'].weight
                            _fB = _mod.lora_B['fused_lora'].weight
                            _fs = _mod.scaling.get('fused_lora', 'N/A')
                            print(f"[INFER DEBUG]   fused_lora: scaling={_fs}, A_norm={_fA.norm():.4f}, B_norm={_fB.norm():.4f}, A_device={_fA.device}, A_dtype={_fA.dtype}")
                        # 对比原始 adapter 的参数
                        _first_orig = selected_loras[0] if selected_loras else None
                        if _first_orig and _first_orig in _mod.lora_A:
                            _oA = _mod.lora_A[_first_orig].weight
                            _oB = _mod.lora_B[_first_orig].weight
                            _os = _mod.scaling.get(_first_orig, 'N/A')
                            print(f"[INFER DEBUG]   {_first_orig}: scaling={_os}, A_norm={_oA.norm():.4f}, B_norm={_oB.norm():.4f}, A_device={_oA.device}, A_dtype={_oA.dtype}")
                        break
                # === 诊断结束 ===

                template.model = model

                print(f"[INFER DEBUG] Sample {idx}: 开始 inference...")
                _t_infer_start = _time.time()
                if inference_images:
                    response, _ = inference(
                        model,
                        template,
                        adjusted_query,
                        history=[],
                        system=None,
                        images=inference_images,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature
                    )
                else:
                    response, _ = inference(
                        model,
                        template,
                        adjusted_query,
                        history=[],
                        system=None,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature
                    )
                _infer_time = _time.time() - _t_infer_start
                _resp_len = len(response) if response else 0
                _resp_tokens = _resp_len // 4  # 粗略估计 token 数
                _tokens_per_sec = _resp_tokens / _infer_time if _infer_time > 0 else 0
                print(f"[INFER DEBUG] Sample {idx}: inference 完成，耗时 {_infer_time:.2f}s, response长度={_resp_len}字符(~{_resp_tokens}tokens), ~{_tokens_per_sec:.1f}tokens/s")
            
            result = {
                'idx': idx,
                'query': query,
                'response': response,
                'label': label,
                'selected_loras': selected_loras,
                'weights': weights,
                'num_images': len(resolved_images),
                'num_images_used': len(inference_images),  # 实际用于推理的图片数
                'merge_method': args.merge_method
            }

            # 更新进度条：显示完成状态
            pbar.set_description(f"[{idx+1}/{len(test_data)}] ✓ Done")

            if args.debug:
                logger.info(f"\nSample {idx}:")
                logger.info(f"  Query: {query[:100]}...")
                logger.info(f"  Selected: {list(zip(selected_loras, [f'{w:.3f}' for w in weights]))}")
                logger.info(f"  Response: {response[:200]}...")
                
        except Exception as e:
            # 更新进度条：显示错误状态
            pbar.set_description(f"[{idx+1}/{len(test_data)}] ✗ Error")
            pbar.set_postfix_str(f"Error: {str(e)[:50]}")

            logger.error(f"Error processing sample {idx}: {e}")
            import traceback
            traceback.print_exc()

            result = {
                'idx': idx,
                'query': query,
                'response': f"ERROR: {str(e)}",
                'label': label,
                'selected_loras': [],
                'weights': [],
                'num_images': len(images),
                'merge_method': args.merge_method
            }
        
        results.append(result)
        append_to_jsonl(output_path, result)

    # 关闭进度条
    pbar.close()

    # Fusion模式下清理所有merged adapters
    if args.merge_method == 'fusion':
        logger.info("Cleaning up fused adapters...")
        composer.cleanup(model)

    # Summary
    successful = sum(1 for r in results if not r['response'].startswith('ERROR'))

    # 统计每个 LoRA 被选中的次数
    from collections import Counter
    lora_usage = Counter()
    for r in results:
        if 'selected_loras' in r and r['selected_loras']:
            for lora_name in r['selected_loras']:
                lora_usage[lora_name] += 1

    logger.info(f"\n{'='*80}")
    logger.info(f"推理完成！")
    logger.info(f"{'='*80}")
    logger.info(f"  总样本数:   {len(results)}")
    logger.info(f"  成功:       {successful} ({successful*100/len(results):.1f}%)")
    logger.info(f"  失败:       {len(results) - successful}")
    logger.info(f"  合并方法:   {args.merge_method}")
    logger.info(f"  Top-K:      {args.top_k}")
    logger.info(f"  结果保存:   {output_path}")

    if lora_usage:
        logger.info(f"\nLoRA 使用统计 (Top 10):")
        for lora_name, count in lora_usage.most_common(10):
            short_name = lora_name.replace('app_lora_', '').replace('category_lora_', '')
            logger.info(f"  {short_name:20s}: {count:3d} 次 ({count*100/len(results):.1f}%)")

    logger.info(f"{'='*80}\n")


if __name__ == '__main__':
    main()
