#!/usr/bin/env python
"""Quick diagnostic to check Qwen2-VL model structure vs LoRA checkpoint keys."""
import os, sys, torch
os.environ['MAX_PIXELS'] = '100000'
os.environ['MAX_NUM'] = '12'
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
sys.path.insert(0, '/data1/hmpiao/jinyike/LoraRetriever-internvl2')

from swift.llm.utils import get_model_tokenizer
from swift.tuners import Swift

print('=== 1. Loading base model ===')
model, tokenizer = get_model_tokenizer(
    'qwen2-vl-7b-instruct', torch.float16,
    {'device_map': 'auto', 'low_cpu_mem_usage': True},
    model_id_or_path='/home/hmpiao/hmpiao/Qwen2-VL-7B-Instruct'
)

print('\n=== 2. Base model module paths (first layer) ===')
for name, _ in model.named_modules():
    if 'layers.0.' in name and ('q_proj' in name or 'down_proj' in name):
        print(f'  {name}')
    if 'visual.blocks.0.attn.qkv' == name.split('.')[-1] and 'blocks.0' in name:
        print(f'  {name}')

lora_path = '/home/hmpiao/hmpiao/jinyike/FedMABench/qwen2_vl_lora_app/qwen2_vl_app_lora_adidas/qwen2-vl-7b-instruct/v0-20260131-111220/global_lora_10'
print(f'\n=== 3. Loading LoRA from: {lora_path} ===')
model = Swift.from_pretrained(model, lora_path, adapter_name='app_lora_adidas_qwen2vl', inference_mode=True)

from peft.tuners.lora import LoraLayer
print('\n=== 4. LoRA layers in model (first few) ===')
count = 0
for name, mod in model.named_modules():
    if isinstance(mod, LoraLayer) and count < 8:
        ada_keys = list(mod.lora_A.keys())
        b_norm = mod.lora_B['app_lora_adidas_qwen2vl'].weight.float().norm().item()
        print(f'  {name}: B_norm={b_norm:.6f}')
        count += 1

print('\n=== 5. Model state_dict LoRA keys (first 10) ===')
lora_keys = [k for k in model.state_dict().keys() if 'lora_A' in k and 'adidas' in k]
for k in sorted(lora_keys)[:10]:
    v = model.state_dict()[k]
    print(f'  {k}  norm={v.float().norm():.4f}')

print('\n=== 6. Checkpoint keys (first 10) ===')
import safetensors.torch
ckpt = safetensors.torch.load_file(os.path.join(lora_path, 'adapter_model.safetensors'), device='cpu')
for k in sorted(ckpt.keys())[:10]:
    print(f'  {k}  norm={ckpt[k].float().norm():.4f}')
