#!/usr/bin/env python
"""
End-to-End LoraRetriever Inference Test
- LoRA Retrieval using jina-embeddings-v4
- LoRA Fusion/Mixture
- InternVL2 Inference

Usage:
    python tests/test_e2e_inference.py --num_samples 5
"""

import os
import sys
import json
import argparse
from typing import Dict, List, Any, Optional

# Environment setup
if 'MAX_PIXELS' not in os.environ:
    os.environ['MAX_PIXELS'] = '150000'
if 'MAX_NUM' not in os.environ:
    os.environ['MAX_NUM'] = '12'

# Add paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGO_ROOT = '/home/hmpiao/hmpiao/jinyike/LoGO-internVL2'
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, LOGO_ROOT)  # Use LoGO's Swift framework

import torch
import random
random.seed(42)

# Swift imports from LoGO
from swift.tuners import Swift
from swift.llm.utils import get_model_tokenizer, get_template, inference
from swift.utils import get_logger

# Local imports
from lora_retriever import LoraRetriever, LoraRetrieverConfig

logger = get_logger()


def load_configs(config_path: str) -> List[Dict]:
    """Load LoRA configs and make paths absolute"""
    with open(config_path, 'r') as f:
        configs = json.load(f)
    for cfg in configs:
        if 'embedding_path' in cfg and not cfg['embedding_path'].startswith('/'):
            cfg['embedding_path'] = os.path.join(PROJECT_ROOT, cfg['embedding_path'])
    return configs


def load_jsonl(path: str) -> List[Dict]:
    """Load JSONL dataset"""
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def resolve_images(images: List[str]) -> List[str]:
    """Resolve relative image paths"""
    resolved = []
    for img in images:
        if img.startswith('./../'):
            img = img.replace('./../', '')
            img = os.path.join(os.path.dirname(PROJECT_ROOT), img)
        elif not os.path.isabs(img):
            img = os.path.join(PROJECT_ROOT, img)
        if os.path.exists(img):
            resolved.append(img)
    return resolved


def adjust_query(query: str, num_images: int) -> str:
    """Adjust image placeholders in query"""
    lines = query.split('\n')
    non_image = [l for l in lines if l.strip() != '<image>']
    return '\n'.join(['<image>'] * num_images + non_image)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_samples', type=int, default=5)
    parser.add_argument('--test_data', type=str, 
                       default=os.path.join(PROJECT_ROOT, 'data/Val_100.jsonl'))
    parser.add_argument('--top_k', type=int, default=3)
    parser.add_argument('--merge_method', type=str, default='mixture',
                       choices=['mixture', 'fusion'])
    args = parser.parse_args()
    
    print("=" * 70)
    print("LoraRetriever End-to-End Inference Test")
    print("=" * 70)
    print(f"Samples: {args.num_samples}")
    print(f"Top-K: {args.top_k}")
    print(f"Merge Method: {args.merge_method}")
    print("=" * 70)
    
    # Load test data
    test_data = load_jsonl(args.test_data)[:args.num_samples]
    print(f"\nLoaded {len(test_data)} samples")
    
    # Load LoRA configs
    app_config_path = os.path.join(PROJECT_ROOT, 'config/app_loras_config_internvl2.json')
    lora_configs = load_configs(app_config_path)
    print(f"Loaded {len(lora_configs)} LoRA configs")
    
    # =========================================================================
    # Step 1: Initialize Retriever
    # =========================================================================
    print("\n[Step 1] Initializing LoRA Retriever...")
    
    retriever_config = LoraRetrieverConfig(
        lora_configs=lora_configs,
        model_path='/home/hmpiao/hmpiao/jina-embeddings-v4',
        device='cuda',
        top_k=args.top_k
    )
    retriever = LoraRetriever(retriever_config)
    retriever.load_lora_embeddings(lora_configs)
    print(f"  Loaded {len(retriever.lora_embeddings)} LoRA embeddings")
    
    # =========================================================================
    # Step 2: Load Base Model
    # =========================================================================
    print("\n[Step 2] Loading InternVL2-2B base model...")
    
    model_path = '/home/hmpiao/hmpiao/InternVL2-2B-ModelScope/OpenGVLab/InternVL2-2B'
    model_type = 'internvl2-2b'
    
    model_kwargs = {'device_map': 'auto', 'low_cpu_mem_usage': True}
    model, tokenizer = get_model_tokenizer(
        model_type,
        torch.float16,
        model_kwargs,
        model_id_or_path=model_path
    )
    
    template = get_template(
        'internvl2',
        tokenizer,
        None,
        2048,
        'truncation_left',
        model=model
    )
    print("  Model loaded successfully")
    
    # =========================================================================
    # Step 3: Load All LoRA Adapters
    # =========================================================================
    print("\n[Step 3] Loading LoRA adapters...")
    
    adapter_names = []
    for cfg in lora_configs:
        lora_name = cfg['lora_name']
        lora_path = cfg['lora_path']
        
        if not os.path.exists(lora_path):
            print(f"  [SKIP] {lora_name}: path not found")
            continue
        
        try:
            model = Swift.from_pretrained(
                model,
                lora_path,
                adapter_name=lora_name,
                inference_mode=True
            )
            adapter_names.append(lora_name)
            print(f"  [OK] {lora_name}")
        except Exception as e:
            print(f"  [FAIL] {lora_name}: {e}")
    
    print(f"\n  Total loaded: {len(adapter_names)} adapters")
    
    # =========================================================================
    # Step 4: Run E2E Inference
    # =========================================================================
    print("\n[Step 4] Running E2E Inference...")
    print("=" * 70)
    
    from swift.tuners.lora_layers import logo_mixture_context
    
    results = []
    
    for idx, sample in enumerate(test_data):
        print(f"\n--- Sample {idx+1}/{len(test_data)} ---")
        
        query = sample.get('query', '')
        images = sample.get('images', [])
        label = sample.get('response', '')
        
        # Resolve images
        resolved_images = resolve_images(images)
        print(f"Query: {query[:80]}...")
        print(f"Images: {len(resolved_images)}/{len(images)} resolved")
        
        # =====================================================================
        # Step 4a: Retrieve Top-K LoRAs
        # =====================================================================
        query_input = {'text': query, 'images': resolved_images[:2]}
        selected_loras, weights = retriever.retrieve_with_weights(query_input, top_k=args.top_k)
        
        print(f"Selected LoRAs: {list(zip(selected_loras, [f'{w:.3f}' for w in weights]))}")
        
        # =====================================================================
        # Step 4b: Compose and Infer
        # =====================================================================
        adjusted_query = adjust_query(query, len(resolved_images))
        
        # Filter to only loaded adapters
        valid_selected = [n for n in selected_loras if n in adapter_names]
        valid_weights = [weights[i] for i, n in enumerate(selected_loras) if n in adapter_names]
        
        if not valid_selected:
            print("  [WARN] No valid LoRAs selected, using base model")
            response = "[No LoRA available]"
        else:
            # Create lora_mapping for mixture mode
            batch_size = 1
            lora_mapping = torch.zeros(
                batch_size, len(adapter_names),
                device=next(model.parameters()).device
            )
            for name, w in zip(valid_selected, valid_weights):
                if name in adapter_names:
                    lora_idx = adapter_names.index(name)
                    lora_mapping[0, lora_idx] = w
            
            # Run inference with mixture
            try:
                template.model = model
                with logo_mixture_context(lora_mapping, adapter_names):
                    if resolved_images:
                        response, _ = inference(
                            model,
                            template,
                            adjusted_query,
                            history=[],
                            system=None,
                            images=resolved_images,
                            max_new_tokens=256,
                            temperature=0.0,
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
                            max_new_tokens=256,
                            temperature=0.0,
                            merging_type='mixture',
                            lora_mapping=lora_mapping,
                            mixture_adapter_names=adapter_names
                        )
            except Exception as e:
                print(f"  [ERROR] Inference failed: {e}")
                response = f"[ERROR: {str(e)[:100]}]"
        
        print(f"Response: {response[:150]}...")
        print(f"Label:    {label[:150]}...")
        
        results.append({
            'idx': idx,
            'query': query[:100],
            'selected_loras': valid_selected,
            'weights': valid_weights,
            'response': response,
            'label': label
        })
    
    # Summary
    print("\n" + "=" * 70)
    print("INFERENCE SUMMARY")
    print("=" * 70)
    for r in results:
        print(f"Sample {r['idx']}: {r['selected_loras']}")
        print(f"  Response: {r['response'][:80]}...")
    print("=" * 70)
    print(f"Completed {len(results)} samples")
    print("=" * 70)


if __name__ == '__main__':
    main()
