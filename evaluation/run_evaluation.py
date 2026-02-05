#!/usr/bin/env python
"""
Full Evaluation Pipeline for LoraRetriever

This script runs the complete evaluation pipeline:
1. Run inference on test dataset using LoraRetriever
2. Calculate step-level and episode-level accuracy using test_swift.py
3. Compare with baseline (optional)
4. Generate evaluation report

Usage:
    # Full evaluation on Val_100.jsonl
    python evaluation/run_evaluation.py --test_data data/Val_100.jsonl --top_k 3 --merge_method mixture
    
    # IID evaluation (in-distribution: test on apps with LoRAs)
    python evaluation/run_evaluation.py --test_data data/Val_100.jsonl --eval_type iid
    
    # OOD evaluation (out-of-distribution: test on unseen apps)
    python evaluation/run_evaluation.py --test_data data/Val_100.jsonl --eval_type ood
    
    # Evaluate existing results only (no inference)
    python evaluation/run_evaluation.py --skip_inference --results_path output/xxx.jsonl
"""

import os
import sys
import json
import argparse
from typing import Dict, List, Any
from datetime import datetime
from pathlib import Path
from collections import defaultdict

# Add paths
PROJECT_ROOT = Path(__file__).parent.parent
EVAL_DIR = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'swift'))
sys.path.insert(0, str(EVAL_DIR))  # For importing test_swift

# Environment setup
if 'MAX_PIXELS' not in os.environ:
    os.environ['MAX_PIXELS'] = '150000'
if 'MAX_NUM' not in os.environ:
    os.environ['MAX_NUM'] = '12'

import torch
import random
import numpy as np
from tqdm import tqdm

random.seed(42)
torch.manual_seed(42)

# Import from eval_gpt for TF-IDF calculation
from eval_gpt import calculate_tfidf

# Known apps with LoRAs (IID set)
IID_APPS = {
    'adidas', 'amazon', 'calendar', 'clock', 'decathlon', 'ebay', 
    'etsy', 'flipkart', 'gmail', 'google_drive', 'google_maps', 
    'kitchen_stories', 'reminder', 'youtube'
}


def read_jsonl(path: str) -> List[Dict]:
    """Load JSONL file"""
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def judge_step(a: str, b: str, threshold: float = 0.6) -> int:
    """Judge if two responses match using TF-IDF similarity"""
    if calculate_tfidf(a, b) > threshold:
        return 1
    return 0


def calculate_step_accuracy(data: List[Dict], threshold: float = 0.6) -> Dict[str, Any]:
    """
    Calculate step-level accuracy.
    Each sample has one response and one label - compare them directly.
    """
    correct = 0
    total = 0
    
    for item in data:
        label = item.get('label', '')
        response = item.get('response', '')
        
        # Skip trivial cases
        if label == 'Click at a button':
            continue
        
        total += 1
        if judge_step(label, response, threshold):
            correct += 1
    
    accuracy = correct / total if total > 0 else 0.0
    return {
        'accuracy': accuracy,
        'correct': correct,
        'total': total
    }


def calculate_episode_accuracy(data: List[Dict], threshold: float = 0.6) -> Dict[str, Any]:
    """
    Calculate episode-level accuracy.
    Group samples by episode_id/instruction, episode is correct if ALL samples match.
    """
    import re
    
    episodes = {}
    
    for item in data:
        # Try episode_id first
        episode_id = item.get('episode_id', None)
        
        if not episode_id:
            # Extract instruction from query as episode key
            query = item.get('query', '')
            match = re.search(r'### User Instruction ###\n(.*?)\n###', query, re.DOTALL)
            if match:
                episode_id = match.group(1).strip()
            else:
                # Extract instruction after <image> tags
                lines = query.split('\n')
                instruction_lines = [l for l in lines if l.strip() != '<image>' and l.strip()]
                episode_id = '\n'.join(instruction_lines).strip() if instruction_lines else query
        
        if episode_id not in episodes:
            episodes[episode_id] = []
        episodes[episode_id].append(item)
    
    correct = 0
    total = len(episodes)
    
    for episode_id, items in episodes.items():
        all_correct = True
        for item in items:
            label = item.get('label', '')
            response = item.get('response', '')
            if not judge_step(label, response, threshold):
                all_correct = False
                break
        if all_correct:
            correct += 1
    
    accuracy = correct / total if total > 0 else 0.0
    return {
        'accuracy': accuracy,
        'correct': correct,
        'total': total
    }


def calculate_retrieval_stats(data: List[Dict]) -> Dict[str, Any]:
    """Calculate LoRA retrieval statistics"""
    lora_counts = defaultdict(int)
    lora_weights = defaultdict(list)
    
    for item in data:
        selected_loras = item.get('selected_loras', [])
        weights = item.get('weights', [])
        
        for lora, weight in zip(selected_loras, weights):
            lora_counts[lora] += 1
            lora_weights[lora].append(weight)
    
    lora_stats = []
    for lora in sorted(lora_counts.keys()):
        lora_stats.append({
            'lora_name': lora,
            'count': lora_counts[lora],
            'avg_weight': np.mean(lora_weights[lora]) if lora_weights[lora] else 0.0
        })
    
    return {
        'lora_stats': sorted(lora_stats, key=lambda x: x['count'], reverse=True),
        'total_samples': len(data)
    }


def detect_app_from_query(query: str) -> str:
    """Try to detect app name from query"""
    query_lower = query.lower()
    
    for app in IID_APPS:
        if app.replace('_', ' ') in query_lower:
            return app
    
    app_patterns = [
        ('youtube', ['youtube']),
        ('amazon', ['amazon']),
        ('gmail', ['gmail', 'email']),
        ('google_maps', ['maps', 'navigation']),
        ('google_drive', ['drive', 'docs']),
        ('calendar', ['calendar', 'event']),
        ('clock', ['clock', 'timer', 'alarm']),
        ('flipkart', ['flipkart']),
        ('ebay', ['ebay']),
        ('adidas', ['adidas']),
        ('decathlon', ['decathlon']),
        ('etsy', ['etsy']),
        ('kitchen_stories', ['kitchen stories', 'recipe']),
        ('reminder', ['reminder'])
    ]
    
    for app_name, patterns in app_patterns:
        for pattern in patterns:
            if pattern in query_lower:
                return app_name
    
    return 'unknown'


def split_iid_ood(data: List[Dict]) -> Dict[str, List[Dict]]:
    """Split data into IID and OOD subsets"""
    iid_data = []
    ood_data = []
    
    for item in data:
        query = item.get('query', '')
        detected_app = detect_app_from_query(query)
        
        if detected_app in IID_APPS:
            iid_data.append(item)
        else:
            ood_data.append(item)
    
    return {'iid': iid_data, 'ood': ood_data, 'all': data}


def run_inference(
    test_data: List[Dict],
    lora_configs: List[Dict],
    model_type: str = 'internvl2-2b',
    model_path: str = '/home/hmpiao/hmpiao/InternVL2-2B-ModelScope/OpenGVLab/InternVL2-2B',
    jina_model_path: str = '/home/hmpiao/hmpiao/jina-embeddings-v4',
    top_k: int = 3,
    merge_method: str = 'mixture',
    max_samples: int = None
) -> List[Dict]:
    """Run LoraRetriever inference on test data"""
    from swift.tuners import Swift
    from swift.llm.utils import get_model_tokenizer, get_template, inference
    from swift.utils import get_logger
    from lora_retriever import LoraRetriever, LoraRetrieverConfig
    
    logger = get_logger()
    
    if max_samples:
        test_data = test_data[:max_samples]
    
    print(f"\n[Inference] Running on {len(test_data)} samples")
    print(f"  Model: {model_type}")
    print(f"  Top-K: {top_k}")
    print(f"  Merge Method: {merge_method}")
    
    # Initialize retriever
    print("\n[1/4] Initializing LoRA Retriever...")
    retriever_config = LoraRetrieverConfig(
        lora_configs=lora_configs,
        model_path=jina_model_path,
        device='cuda',
        top_k=top_k
    )
    retriever = LoraRetriever(retriever_config)
    retriever.load_lora_embeddings(lora_configs)
    print(f"  Loaded {len(retriever.lora_embeddings)} LoRA embeddings")
    
    # Load base model
    print("\n[2/4] Loading base model...")
    model_kwargs = {'device_map': 'auto', 'low_cpu_mem_usage': True}
    model, tokenizer = get_model_tokenizer(
        model_type,
        torch.float16,
        model_kwargs,
        model_id_or_path=model_path
    )
    
    template_type = 'internvl2' if 'internvl' in model_type else 'qwen2-vl'
    template = get_template(template_type, tokenizer, None, 2048, 'truncation_left', model=model)
    print("  Model loaded successfully")
    
    # Load LoRA adapters
    print("\n[3/4] Loading LoRA adapters...")
    adapter_names = []
    for cfg in lora_configs:
        lora_name = cfg['lora_name']
        lora_path = cfg['lora_path']
        
        if not os.path.exists(lora_path):
            continue
        
        try:
            model = Swift.from_pretrained(model, lora_path, adapter_name=lora_name, inference_mode=True)
            adapter_names.append(lora_name)
        except Exception as e:
            print(f"  [FAIL] {lora_name}: {e}")
    
    print(f"  Loaded {len(adapter_names)} adapters")
    
    # Run inference
    print("\n[4/4] Running inference...")
    from swift.tuners.lora_layers import logo_mixture_context
    
    results = []
    
    for idx, sample in enumerate(tqdm(test_data, desc="Inference")):
        query = sample.get('query', '')
        images = sample.get('images', [])
        label = sample.get('response', '')
        episode_id = sample.get('episode_id', str(idx))
        
        # Resolve images
        resolved_images = []
        for img in images:
            if img.startswith('./../'):
                img = img.replace('./../', '')
                img = os.path.join(os.path.dirname(PROJECT_ROOT), img)
            elif not os.path.isabs(img):
                img = os.path.join(PROJECT_ROOT, img)
            if os.path.exists(img):
                resolved_images.append(img)
        
        # Retrieve top-k LoRAs
        query_input = {'text': query, 'images': resolved_images[:2] if resolved_images else None}
        selected_loras, weights = retriever.retrieve_with_weights(query_input, top_k=top_k)
        
        # Adjust query for images
        lines = query.split('\n')
        non_image = [l for l in lines if l.strip() != '<image>']
        adjusted_query = '\n'.join(['<image>'] * len(resolved_images) + non_image)
        
        # Filter to valid adapters
        valid_selected = [n for n in selected_loras if n in adapter_names]
        valid_weights = [weights[i] for i, n in enumerate(selected_loras) if n in adapter_names]
        
        if not valid_selected:
            response = "[No LoRA available]"
        else:
            batch_size = 1
            lora_mapping = torch.zeros(batch_size, len(adapter_names), device=next(model.parameters()).device)
            for name, w in zip(valid_selected, valid_weights):
                if name in adapter_names:
                    lora_mapping[0, adapter_names.index(name)] = w
            
            try:
                template.model = model
                with logo_mixture_context(lora_mapping, adapter_names):
                    response, _ = inference(
                        model, template, adjusted_query,
                        history=[], system=None,
                        images=resolved_images if resolved_images else None,
                        max_new_tokens=256, temperature=0.0,
                        merging_type='mixture',
                        lora_mapping=lora_mapping,
                        mixture_adapter_names=adapter_names
                    )
            except Exception as e:
                response = f"[ERROR: {str(e)[:100]}]"
        
        results.append({
            'idx': idx,
            'query': query,
            'response': response,
            'label': label,
            'selected_loras': valid_selected,
            'weights': [float(w) for w in valid_weights] if valid_weights else [],
            'num_images': len(resolved_images),
            'merge_method': merge_method,
            'episode_id': episode_id
        })
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Run full LoraRetriever evaluation')
    parser.add_argument('--test_data', type=str, default='data/Val_100.jsonl',
                       help='Path to test JSONL file')
    parser.add_argument('--config_path', type=str, 
                       default=str(PROJECT_ROOT / 'config/app_loras_config_internvl2.json'),
                       help='Path to LoRA config JSON')
    parser.add_argument('--top_k', type=int, default=3)
    parser.add_argument('--merge_method', type=str, default='mixture', choices=['mixture', 'fusion'])
    parser.add_argument('--eval_type', type=str, default='all', choices=['all', 'iid', 'ood'])
    parser.add_argument('--threshold', type=float, default=0.6)
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--output_dir', type=str, default=str(PROJECT_ROOT / 'output/evaluation_results'))
    parser.add_argument('--skip_inference', action='store_true')
    parser.add_argument('--results_path', type=str, default=None)
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    
    print("=" * 70)
    print("LoraRetriever Full Evaluation Pipeline")
    print("=" * 70)
    
    # Load or run inference
    if args.skip_inference and args.results_path:
        print(f"\n[Skip Inference] Loading results from: {args.results_path}")
        results = read_jsonl(args.results_path)
    else:
        test_data = read_jsonl(args.test_data)
        print(f"Loaded {len(test_data)} test samples")
        
        with open(args.config_path, 'r') as f:
            lora_configs = json.load(f)
        
        for cfg in lora_configs:
            if 'embedding_path' in cfg and not cfg['embedding_path'].startswith('/'):
                cfg['embedding_path'] = str(PROJECT_ROOT / cfg['embedding_path'])
        
        splits = split_iid_ood(test_data)
        print(f"Data Split: IID={len(splits['iid'])}, OOD={len(splits['ood'])}")
        
        if args.eval_type == 'iid':
            eval_data = splits['iid']
        elif args.eval_type == 'ood':
            eval_data = splits['ood']
        else:
            eval_data = splits['all']
        
        if not eval_data:
            print("No samples to evaluate!")
            return
        
        results = run_inference(eval_data, lora_configs, top_k=args.top_k, 
                              merge_method=args.merge_method, max_samples=args.max_samples)
        
        results_filename = f"eval_{args.eval_type}_{args.merge_method}_k{args.top_k}_{timestamp}.jsonl"
        results_path = os.path.join(args.output_dir, results_filename)
        with open(results_path, 'w', encoding='utf-8') as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
        print(f"\nResults saved to: {results_path}")
    
    # Calculate metrics using correct evaluation logic
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    
    step_results = calculate_step_accuracy(results, args.threshold)
    episode_results = calculate_episode_accuracy(results, args.threshold)
    retrieval_stats = calculate_retrieval_stats(results)
    
    print(f"\n[Step-Level Accuracy]")
    print(f"  Accuracy: {step_results['accuracy'] * 100:.2f}%")
    print(f"  Correct: {step_results['correct']}/{step_results['total']}")
    
    print(f"\n[Episode-Level Accuracy]")
    print(f"  Accuracy: {episode_results['accuracy'] * 100:.2f}%")
    print(f"  Correct: {episode_results['correct']}/{episode_results['total']}")
    
    print(f"\n[LoRA Retrieval Stats]")
    print(f"  {'LoRA Name':<30} | {'Count':>6} | {'Avg Weight':>10}")
    print(f"  {'-'*30}-+-{'-'*6}-+-{'-'*10}")
    for stat in retrieval_stats['lora_stats'][:10]:
        print(f"  {stat['lora_name']:<30} | {stat['count']:>6} | {stat['avg_weight']:>10.4f}")
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Step Accuracy:    {step_results['accuracy'] * 100:>6.2f}%")
    print(f"Episode Accuracy: {episode_results['accuracy'] * 100:>6.2f}%")
    print("=" * 70)
    
    return {'step_accuracy': step_results['accuracy'], 'episode_accuracy': episode_results['accuracy']}


if __name__ == "__main__":
    main()
