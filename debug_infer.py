#!/usr/bin/env python
"""Direct test: does the Qwen2-VL LoRA produce structured actions or natural language?"""
import os, sys, torch
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
os.environ['MAX_PIXELS'] = '100000'
os.environ['MAX_NUM'] = '12'
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
sys.path.insert(0, '/data1/hmpiao/jinyike/LoraRetriever-internvl2')

from swift.llm.utils import get_model_tokenizer, get_template, inference
from swift.tuners import Swift

# 1. Load model + LoRA
print('=== Loading model ===')
model, tokenizer = get_model_tokenizer(
    'qwen2-vl-7b-instruct', torch.float16,
    {'device_map': 'auto', 'low_cpu_mem_usage': True},
    model_id_or_path='/home/hmpiao/hmpiao/Qwen2-VL-7B-Instruct'
)
template = get_template('qwen2-vl', tokenizer, None, model.max_model_len, None, model=model)
template.model = model

# Load a single LoRA (youtube) since test query is about YouTube
lora_path = '/home/hmpiao/hmpiao/jinyike/FedMABench/qwen2_vl_lora_app/qwen2_vl_app_lora_youtube/qwen2-vl-7b-instruct/v0-20260131-093519/global_lora_10'
print(f'Loading YouTube LoRA from: {lora_path}')
model = Swift.from_pretrained(model, lora_path, adapter_name='youtube_lora', inference_mode=True)
template.model = model

# 2. Test WITHOUT mixture mode (standard PEFT forward)
print('\n=== Test 1: Standard PEFT forward (single LoRA active) ===')
# Load a test image
import json
with open('/data1/hmpiao/jinyike/LoraRetriever-internvl2/data/Val_100.jsonl') as f:
    for line in f:
        sample = json.loads(line)
        if 'youtube' in sample['query'].lower() or 'YouTube' in sample['query']:
            break

query = sample['query']
images = sample['images']
label = sample['response']

# Resolve image paths
from pathlib import Path
resolved = []
for img in images:
    p = Path(img)
    if p.exists():
        resolved.append(str(p))
    else:
        p2 = Path('/data1/hmpiao/jinyike/LoraRetriever-internvl2') / img
        if p2.exists():
            resolved.append(str(p2))
        else:
            p3 = Path('/data1/hmpiao/jinyike/LoraRetriever-internvl2').parent / img.lstrip('./')
            if p3.exists():
                resolved.append(str(p3))

print(f'Query: {query[:100]}...')
print(f'Images: {len(resolved)}')
print(f'Expected label: {label[:200]}')

response1, _ = inference(
    model, template, query,
    history=[], system=None,
    images=resolved if resolved else None,
    max_new_tokens=512, temperature=0.01,
)
print(f'\nResponse (standard PEFT): {response1[:300]}')

# 3. Test WITH mixture mode
print('\n\n=== Test 2: Mixture forward (single LoRA, weight=1.0) ===')
from swift.tuners.lora_layers import logo_mixture_context
lora_mapping = torch.zeros(1, 1, device=next(model.parameters()).device)
lora_mapping[0, 0] = 1.0

response2, _ = inference(
    model, template, query,
    history=[], system=None,
    images=resolved if resolved else None,
    max_new_tokens=512, temperature=0.01,
    merging_type='mixture',
    lora_mapping=lora_mapping,
    mixture_adapter_names=['youtube_lora']
)
print(f'Response (mixture): {response2[:300]}')

# 4. Test base model (disable LoRA)
print('\n\n=== Test 3: Base model (disable LoRA) ===')
model.disable_adapter_layers()
response3, _ = inference(
    model, template, query,
    history=[], system=None,
    images=resolved if resolved else None,
    max_new_tokens=512, temperature=0.01,
)
print(f'Response (base model): {response3[:300]}')
model.enable_adapter_layers()

print('\n\n=== COMPARISON ===')
print(f'Standard PEFT == Base model: {response1[:100] == response3[:100]}')
print(f'Mixture == Base model: {response2[:100] == response3[:100]}')
print(f'Standard PEFT == Mixture: {response1[:100] == response2[:100]}')
