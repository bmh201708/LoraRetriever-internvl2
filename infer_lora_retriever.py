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
from collections import Counter

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
from swift.tuners import Swift, SwiftConfig, SwiftModel, LoRAConfig

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
    'qwen2-vl-2b-instruct': '/data1/hmpiao/Qwen2-VL-2B-Instruct',
    'qwen2-vl-7b-instruct': '/home/hmpiao/hmpiao/Qwen2-VL-7B-Instruct',
}

CONFIG_PATHS = {
    'internvl2-2b': {
        'app': 'config/app_loras_config_internvl2.json',
        'category': 'config/category_loras_config_internvl2.json',
    },
    'qwen2-vl-2b-instruct': {
        'app': 'config/app_loras_config_qwen2vl.json',
        # 当前仓库仅 app LoRA 已切到 2B；category 仍是 7B，默认禁用避免误加载
        'category': '/dev/null',
    },
    'qwen2-vl-7b-instruct': {
        'app': 'config/app_loras_config_qwen2vl.json',
        'category': 'config/category_loras_config_qwen2vl.json',
    },
}

JINA_MODEL_PATH = '/home/hmpiao/hmpiao/jina-embeddings-v4'

# Candidate app LoRA pool control (empty tuple means all app LoRAs from config).
# Example:
# APP = ('amazon', 'adidas', 'ebay')
APP: Tuple[str, ...] = (
    'amazon',
    'clock',
    'ebay',
    'etsy',
    'flipkart',
    'google_drive',
    'reminder',
    'youtube',
)


def parse_args():
    parser = argparse.ArgumentParser(description='LoraRetriever Inference for VLM')
    
    # Model settings
    parser.add_argument('--model_type', type=str, default='internvl2-2b',
                       choices=['internvl2-2b', 'qwen2-vl-2b-instruct', 'qwen2-vl-7b-instruct'])
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
    parser.add_argument('--composition_weight_mode', type=str, default='uniform',
                       choices=['uniform', 'weighted'],
                       help='uniform: 1/n (mixture) and 1/k (fusion); weighted: use retriever similarity weights')
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
    parser.add_argument('--dialog_mode', type=str, default='step',
                       choices=['step', 'episode'],
                       help='For multi-turn messages format: step=teacher forcing history, episode=autoregressive history')
    
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


def _extract_app_from_lora_name(lora_name: str) -> Optional[str]:
    """Extract app name from lora_name like 'app_lora_amazon_qwen2vl'."""
    if not isinstance(lora_name, str):
        return None
    if not lora_name.startswith('app_lora_'):
        return None
    app_part = lora_name[len('app_lora_'):]
    if '_' in app_part:
        app_part = app_part.rsplit('_', 1)[0]
    return app_part if app_part else None


def filter_lora_configs_by_app_pool(
    lora_configs: List[Dict[str, Any]],
    app_pool: Any
) -> List[Dict[str, Any]]:
    """
    Keep all non-app LoRAs; filter app LoRAs by APP tuple.
    If APP is empty, keep all configs.
    """
    if not app_pool:
        return lora_configs

    if isinstance(app_pool, str):
        # Guard against accidental tuple-without-commas:
        # APP = ("adidas" "amazon") -> one long concatenated string.
        if (',' not in app_pool) and (' ' not in app_pool):
            raise ValueError(
                f"APP format seems invalid: {app_pool!r}. "
                "If you define multiple apps, use commas, e.g. "
                "APP = ('adidas', 'amazon')."
            )
        app_pool = [x.strip() for x in app_pool.replace(',', ' ').split() if x.strip()]

    allowed = {str(x).strip() for x in app_pool if str(x).strip()}
    if not allowed:
        return lora_configs

    available_apps = {
        app for app in (_extract_app_from_lora_name(cfg.get('lora_name', '')) for cfg in lora_configs)
        if app is not None
    }
    matched_allowed = sorted([a for a in allowed if a in available_apps])
    unmatched_allowed = sorted([a for a in allowed if a not in available_apps])
    if unmatched_allowed:
        logger.warning(f"APP pool contains unknown app names (ignored): {unmatched_allowed}")
    if not matched_allowed and available_apps:
        raise ValueError(
            f"APP pool has no valid app in current configs. "
            f"Given={sorted(allowed)}, available={sorted(available_apps)}"
        )

    filtered: List[Dict[str, Any]] = []
    for cfg in lora_configs:
        lora_name = cfg.get('lora_name', '')
        app_name = _extract_app_from_lora_name(lora_name)
        if app_name is None or app_name in matched_allowed:
            filtered.append(cfg)
    return filtered


def build_lora_rank_map(lora_configs: List[Dict[str, Any]]) -> Dict[str, int]:
    """Build LoRA rank map from adapter_config.json for mixed-rank diagnostics."""
    rank_map: Dict[str, int] = {}
    for cfg in lora_configs:
        name = cfg.get('lora_name')
        path = cfg.get('lora_path')
        if not name or not path:
            continue
        cfg_path = os.path.join(path, 'adapter_config.json')
        if not os.path.exists(cfg_path):
            continue
        try:
            with open(cfg_path, 'r', encoding='utf-8') as f:
                adapter_cfg = json.load(f)
            rank_val = adapter_cfg.get('r')
            if rank_val is not None:
                rank_map[name] = int(rank_val)
        except Exception:
            continue
    return rank_map


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


def strip_image_placeholders(text: str) -> str:
    """Remove <image> placeholder lines for text-only history."""
    lines = str(text).split('\n')
    cleaned = [l for l in lines if l.strip() != '<image>']
    return '\n'.join(cleaned).strip()


def build_dialog_history(
    turns: List[Dict[str, Any]],
    up_to_turn_idx: int,
    mode: str,
    predictions: List[str],
    image_counts: Optional[List[int]] = None
) -> List[List[str]]:
    """
    Build dialog history for the current turn (exclude current turn itself).

    mode='step'    : use ground-truth assistant responses for history
    mode='episode' : use model predictions for history
    image_counts   : optional per-turn image count used to keep <image> placeholders aligned
    """
    history: List[List[str]] = []
    for k in range(up_to_turn_idx):
        user_query = turns[k].get('query', '')
        if image_counts is not None and k < len(image_counts):
            user_query = prepare_query(user_query, image_counts[k])
        if mode == 'episode':
            assistant_reply = predictions[k] if k < len(predictions) else ''
        else:
            assistant_reply = turns[k].get('label', '')
        history.append([user_query, assistant_reply])
    return history


def extract_dialog_turns(sample: Dict[str, Any]) -> Tuple[Optional[str], List[Dict[str, Any]]]:
    """
    Convert messages-format sample into dialog turns.
    Expected input format:
      {
        "episode_id": "...",
        "images": [...],
        "messages": [{"role":"system|user|assistant","content":"..."}...]
      }
    Returns:
      system_prompt, turns=[{"query":..., "images":[...], "label":...}, ...]
    """
    messages = sample.get('messages', [])
    if not isinstance(messages, list) or len(messages) == 0:
        return None, []

    all_images = sample.get('images', []) or []
    system_prompt = None
    start_idx = 0

    if str(messages[0].get('role', '')).lower().strip() == 'system':
        system_prompt = str(messages[0].get('content', ''))
        start_idx = 1

    turns: List[Dict[str, Any]] = []
    image_ptr = 0
    i = start_idx

    while i < len(messages):
        role = str(messages[i].get('role', '')).lower().strip()
        if role != 'user':
            i += 1
            continue

        query = str(messages[i].get('content', ''))
        num_images = query.count('<image>')
        turn_images = all_images[image_ptr:image_ptr + num_images] if num_images > 0 else []
        image_ptr += num_images

        label = ''
        if i + 1 < len(messages):
            next_role = str(messages[i + 1].get('role', '')).lower().strip()
            if next_role == 'assistant':
                label = str(messages[i + 1].get('content', ''))
                i += 2
            else:
                i += 1
        else:
            i += 1

        turns.append({
            'query': query,
            'images': turn_images,
            'label': label
        })

    return system_prompt, turns


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
        f'retriever_results_{model_suffix}_{args.merge_method}_{args.composition_weight_mode}_k{args.top_k}_{time_str}.jsonl'
    )
    
    logger.info("=" * 60)
    logger.info("LoraRetriever Inference")
    logger.info("=" * 60)
    logger.info(f"Model: {args.model_type}")
    logger.info(f"Model Path: {args.model_path}")
    logger.info(f"Merge Method: {args.merge_method}")
    logger.info(f"Composition Weight Mode: {args.composition_weight_mode}")
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
    lora_configs = filter_lora_configs_by_app_pool(lora_configs, APP)
    lora_rank_map = build_lora_rank_map(lora_configs)
    logger.info(f"Loaded {len(lora_configs)} LoRA configs")
    if APP:
        logger.info(f"APP candidate pool enabled: {APP}")
    
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
    
    if 'qwen2-vl' in args.model_type:
        # Custom loading for Qwen2-VL to fix target_modules regex issue
        qwen_configs = {}
        valid_adapters = []
        
        for cfg in lora_configs:
            lora_name = cfg['lora_name']
            lora_path = cfg['lora_path']
            
            if not os.path.exists(lora_path):
                logger.warning(f"LoRA not found: {lora_path}, skipping...")
                continue
                
            try:
                # 1. Load config manually to handle missing 'swift_type' in legacy/peft LoRAs
                json_path = os.path.join(lora_path, 'adapter_config.json')
                if not os.path.exists(json_path):
                     logger.warning(f"Config not found: {json_path}")
                     continue
                     
                with open(json_path, 'r') as f:
                    config_dict = json.load(f)
                
                # Create default LoRAConfig and populate
                swift_config = LoRAConfig()
                for key, value in config_dict.items():
                    if hasattr(swift_config, key):
                        setattr(swift_config, key, value)
                
                # Ensure swift_type is set
                swift_config.swift_type = 'LORA'
                
                # 2. Override target_modules with a REGEX to explicitly include all visual and LLM modules
                # Using regex '.*(suffix1|suffix2)$' approach
                target_suffixes = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj', 'qkv', 'proj', 'fc1', 'fc2', r'merger\.mlp\.0', r'merger\.mlp\.2']
                target_regex = r'.*(' + '|'.join(target_suffixes) + r')$'
                swift_config.target_modules = target_regex
                logger.info(f"Overridden target_modules for {lora_name} to regex: {target_regex}")
                
                qwen_configs[lora_name] = swift_config
                valid_adapters.append((lora_name, lora_path))
            except Exception as e:
                 logger.error(f"Failed to prepare config for {lora_name}: {e}")

        # 3. Initialize SwiftModel with all configs at once
        if qwen_configs:
            try:
                model = SwiftModel(model, qwen_configs, inference_mode=True)
                
                # 4. Load weights for each adapter
                from safetensors.torch import load_file as safe_load_file
                for lora_name, lora_path in valid_adapters:
                    try:
                        # Manual loading with key remapping
                        safe_path = os.path.join(lora_path, 'adapter_model.safetensors')
                        if not os.path.exists(safe_path):
                            logger.warning(f"  Safetensors not found at {safe_path}")
                            continue
                            
                        state_dict = safe_load_file(safe_path)
                        new_state_dict = {}
                        for k, v in state_dict.items():
                            # 1. Strip 'base_model.model.' prefix if present
                            # Keys usually start with base_model.model.model... -> model...
                            if k.startswith('base_model.model.'):
                                k = k[len('base_model.model.'):]
                            
                            # 2. Fix mismatch: model.layers -> model.language_model.layers
                            # Qwen2-VL specific fix because LoRA was trained without language_model prefix consistency
                            if 'model.layers' in k:
                                k = k.replace('model.layers', 'model.language_model.layers')
                                
                            # 3. Insert adapter name into LoRA keys
                            # ...lora_A.weight -> ...lora_A.{adapter_name}.weight
                            if 'lora_' in k and 'weight' in k:
                                # Ensure we don't double replace if some weird naming exists
                                if f".{lora_name}." not in k:
                                    k = k.replace(".weight", f".{lora_name}.weight")
                            
                            new_state_dict[k] = v
                        
                        # Load into model
                        keys_incompatible = model.load_state_dict(new_state_dict, strict=False)
                        
                        if keys_incompatible is None:
                             logger.info(f"  Loaded (Custom Qwen): {lora_name} (No keys info returned)")
                             adapter_names.append(lora_name)
                             continue

                        # Filter out expected missing keys (base model params)
                        unexpected = [x for x in keys_incompatible.unexpected_keys if 'lora_' in x]
                        if unexpected:
                            logger.warning(f"  Loaded {lora_name} with unexpected keys: {unexpected[:5]}...")
                        else:
                            logger.info(f"  Loaded (Custom Qwen): {lora_name}")
                            
                        adapter_names.append(lora_name)

                    except Exception as e:
                         logger.error(f"Failed to load state dict for {lora_name}: {e}")
            except Exception as e:
                logger.error(f"Failed to initialize SwiftModel for Qwen2-VL: {e}")

    else:
        # Standard loading for InternVL2 (and others)
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

    # [DEBUG] Inspect model layers
    logger.info("=" * 40)
    logger.info("[DEBUG] Inspecting model structure:")
    
    # Try to unwrap to see where visual is
    try:
        # model is SwiftModel (PeftModel)
        # model.model is usually the modified base model or BaseTuner
        # model.model.model is the transformers model?
        
        base = model
        if hasattr(base, 'model'):
            logger.info("  model.model found")
            base = base.model
        if hasattr(base, 'model'):
            logger.info("  model.model.model found")
            base = base.model
            
        logger.info(f"  Target base module type: {type(base)}")
        logger.info(f"  Children: {list(base._modules.keys())}")
        
    except Exception as e:
        logger.info(f"  Error inspecting structure: {e}")

    logger.info("=" * 40)

    
    # =========================================================================
    # Initialize Composer
    # =========================================================================
    if args.merge_method == 'mixture':
        composer = MixtureComposer(use_weighted_average=(args.composition_weight_mode == 'weighted'))
    else:
        composer = FusionComposer(combination_type=args.combination_type)
    
    logger.info(f"Using {args.merge_method} composition strategy")
    
    # =========================================================================
    # Run Inference
    # =========================================================================
    logger.info("Starting LoraRetriever inference...")
    results = []
    project_root = str(PROJECT_ROOT)
    lora_usage = Counter()
    mixed_rank_logged = False

    # 创建进度条，显示详细信息
    pbar = tqdm(test_data, desc="LoraRetriever Inference", ncols=120)

    for idx, sample in enumerate(pbar):
        # 兼容旧格式（query/response）和新格式（messages 多轮）
        query = sample.get('query', '')
        images = sample.get('images', [])
        label = sample.get('response', '')
        episode_id = sample.get('episode_id', f'idx_{idx}')
        turns = []
        predictions = []
        ground_truths = []

        try:
            system_prompt, turns = extract_dialog_turns(sample)
            is_multi_turn = len(turns) > 0
            if not is_multi_turn:
                turns = [{'query': query, 'images': images, 'label': label}]
                system_prompt = None

            # 预先解析并限制每一轮图片（用于当前轮推理 + 历史图片回放）
            max_images_per_turn = int(os.environ.get('MAX_NUM', '12'))
            resolved_turn_images: List[List[str]] = []
            for t_idx, t in enumerate(turns):
                imgs = resolve_image_paths(t.get('images', []), project_root)
                if len(imgs) > max_images_per_turn:
                    logger.warning(
                        f"Sample {idx} Turn {t_idx}: {len(imgs)} 张图片超过限制，只使用后 {max_images_per_turn} 张")
                    # Keep newer images for this turn, drop older ones.
                    imgs = imgs[-max_images_per_turn:]
                resolved_turn_images.append(imgs)

            # =========================================================================
            # Step 1: Retrieve once per episode (fixed LoRA combo for all turns)
            # =========================================================================
            pbar.set_description(f"[{idx+1}/{len(test_data)}] Retrieving LoRAs")

            first_query = turns[0].get('query', '')
            first_resolved_images = resolved_turn_images[0] if resolved_turn_images else []

            query_input = {
                'text': first_query,
                'images': first_resolved_images[:2]  # Use max 2 images for embedding retrieval
            }

            selected_loras, weights = retriever.retrieve_with_weights(
                query_input,
                top_k=args.top_k
            )
            for lora_name in selected_loras:
                lora_usage[lora_name] += 1

            selected_ranks = [lora_rank_map.get(n) for n in selected_loras if lora_rank_map.get(n) is not None]
            is_mixed_rank = len(set(selected_ranks)) > 1 if selected_ranks else False
            if is_mixed_rank and not mixed_rank_logged:
                if args.merge_method == 'mixture':
                    logger.info(
                        f"Detected mixed-rank LoRAs (r={selected_ranks}); "
                        "mixture mode is output-level blending and supports mixed-rank natively.")
                else:
                    logger.info(
                        f"Detected mixed-rank LoRAs (r={selected_ranks}); "
                        "fusion mode will use mixed-rank compatible weighted-delta + SVD path.")
                mixed_rank_logged = True

            lora_display = ', '.join([f"{name.split('_')[-1]}" for name in selected_loras[:2]])
            img_info = f"{len(first_resolved_images)}imgs" if len(first_resolved_images) <= 10 else f"{len(first_resolved_images)}imgs(多!)"
            pbar.set_postfix_str(f"{img_info} | LoRAs: {lora_display}")

            if args.show_similarities:
                all_sims = retriever.get_all_similarities(query_input)
                sorted_sims = sorted(all_sims.items(), key=lambda x: x[1], reverse=True)
                logger.info(f"Sample {idx} similarities (episode-level retrieve):")
                for name, sim in sorted_sims:
                    logger.info(f"  {name}: {sim:.4f}")

            # =========================================================================
            # Step 2: Compose once per episode
            # =========================================================================
            lora_mapping = None
            if args.merge_method == 'mixture':
                lora_mapping = composer.compose(
                    model=model,
                    adapter_names=selected_loras,
                    weights=weights,
                    all_adapter_names=adapter_names,
                    batch_size=1,
                    device=next(model.parameters()).device
                )
            else:
                # Fusion: merged adapter is reused for all turns in this episode
                fusion_weights = weights
                if args.composition_weight_mode == 'uniform' and selected_loras:
                    fusion_weights = [1.0 / len(selected_loras)] * len(selected_loras)
                composer.compose(
                    model=model,
                    adapter_names=selected_loras,
                    weights=fusion_weights
                )

            # =========================================================================
            # Step 3: Multi-turn / single-turn inference
            # =========================================================================
            for turn_idx, turn in enumerate(turns):
                pbar.set_description(f"[{idx+1}/{len(test_data)}][{turn_idx+1}/{len(turns)}] Inferencing")

                turn_query = turn.get('query', '')
                turn_label = turn.get('label', '')

                resolved_images = resolved_turn_images[turn_idx]
                inference_images = [
                    img for past_turn_images in resolved_turn_images[:turn_idx + 1] for img in past_turn_images
                ]
                adjusted_query = prepare_query(turn_query, len(resolved_images))

                # step: 历史用 GT；episode: 历史用模型之前输出
                infer_history = build_dialog_history(
                    turns=turns,
                    up_to_turn_idx=turn_idx,
                    mode=args.dialog_mode,
                    predictions=predictions,
                    image_counts=[len(x) for x in resolved_turn_images]
                )

                template.model = model
                if args.merge_method == 'mixture':
                    if inference_images:
                        response, _ = inference(
                            model,
                            template,
                            adjusted_query,
                            history=infer_history,
                            system=system_prompt,
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
                            history=infer_history,
                            system=system_prompt,
                            max_new_tokens=args.max_new_tokens,
                            temperature=args.temperature,
                            merging_type='mixture',
                            lora_mapping=lora_mapping,
                            mixture_adapter_names=adapter_names
                        )
                else:
                    if inference_images:
                        response, _ = inference(
                            model,
                            template,
                            adjusted_query,
                            history=infer_history,
                            system=system_prompt,
                            images=inference_images,
                            max_new_tokens=args.max_new_tokens,
                            temperature=args.temperature
                        )
                    else:
                        response, _ = inference(
                            model,
                            template,
                            adjusted_query,
                            history=infer_history,
                            system=system_prompt,
                            max_new_tokens=args.max_new_tokens,
                            temperature=args.temperature
                        )

                predictions.append(response)
                ground_truths.append(turn_label)

                if args.debug:
                    logger.info(
                        f"Sample {idx} Turn {turn_idx} | Selected: {list(zip(selected_loras, [f'{w:.3f}' for w in weights]))}")
                    logger.info(f"Sample {idx} Turn {turn_idx} | Response: {response[:200]}...")

            # 输出格式与 FedMABench/inference/generate_inference.py 保持一致
            result = {
                'episode_id': episode_id,
                'mode': args.dialog_mode,
                'num_steps': len(turns),
                'ground_truths': ground_truths,
                'predictions': predictions
            }

            pbar.set_description(f"[{idx+1}/{len(test_data)}] ✓ Done")

        except Exception as e:
            pbar.set_description(f"[{idx+1}/{len(test_data)}] ✗ Error")
            pbar.set_postfix_str(f"Error: {str(e)[:50]}")

            logger.error(f"Error processing sample {idx}: {e}")
            import traceback
            traceback.print_exc()

            result = {
                'episode_id': episode_id,
                'mode': args.dialog_mode,
                'num_steps': len(turns) if turns else 1,
                'ground_truths': ground_truths if ground_truths else ([label] if label else []),
                'predictions': [f"ERROR: {str(e)}"]
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
    def _is_success(entry: Dict[str, Any]) -> bool:
        preds = entry.get('predictions', [])
        if isinstance(preds, list):
            return not any(isinstance(x, str) and x.startswith('ERROR:') for x in preds)
        resp = entry.get('response', '')
        return not (isinstance(resp, str) and resp.startswith('ERROR:'))

    successful = sum(1 for r in results if _is_success(r))

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
