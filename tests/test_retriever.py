#!/usr/bin/env python
"""
Test script for LoraRetriever module
Verifies:
1. LoRA embedding loading
2. Query embedding computation
3. Top-k retrieval with cosine similarity
"""

import sys
import os

# Add project paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from lora_retriever import LoraRetriever, LoraRetrieverConfig


def main():
    print("=" * 60)
    print("LoraRetriever Module Test")
    print("=" * 60)
    
    # Test 1: Load config and embeddings
    print("\n[Test 1] Loading LoRA configurations and embeddings...")
    
    import json
    config_path = os.path.join(PROJECT_ROOT, 'config', 'app_loras_config_internvl2.json')
    with open(config_path, 'r') as f:
        lora_configs = json.load(f)
    
    # Make embedding paths absolute
    for cfg in lora_configs:
        if 'embedding_path' in cfg and not cfg['embedding_path'].startswith('/'):
            cfg['embedding_path'] = os.path.join(PROJECT_ROOT, cfg['embedding_path'])
    
    config = LoraRetrieverConfig(
        lora_configs=lora_configs,
        model_path='/home/hmpiao/hmpiao/jina-embeddings-v4',
        device='cuda',
        top_k=3
    )
    
    retriever = LoraRetriever(config)
    retriever.load_lora_embeddings(lora_configs)
    
    print(f"  Loaded {len(retriever.lora_embeddings)} LoRA embeddings")
    print(f"  LoRAs: {list(retriever.lora_embeddings.keys())}")
    
    assert len(retriever.lora_embeddings) == 14, f"Expected 14 embeddings, got {len(retriever.lora_embeddings)}"
    print("  ✓ All 14 app LoRA embeddings loaded")
    
    # Test 2: Retrieve top-k for different queries
    print("\n[Test 2] Testing retrieval for different queries...")
    
    test_queries = [
        ("Search for running shoes on Adidas", ["app_lora_adidas", "app_lora_decathlon"]),
        ("Add item to cart on Amazon", ["app_lora_amazon", "app_lora_ebay", "app_lora_flipkart"]),
        ("Set a timer for 10 minutes", ["app_lora_clock", "app_lora_reminder"]),
        ("Check my emails in Gmail", ["app_lora_gmail", "app_lora_google_drive"]),
        ("Navigate to the nearest coffee shop", ["app_lora_google_maps"]),
        ("Watch a video on YouTube", ["app_lora_youtube"]),
        ("Check my calendar events", ["app_lora_calendar"]),
    ]
    
    for query, expected_any in test_queries:
        selected, weights = retriever.retrieve_with_weights(query, top_k=3)
        
        found_expected = any(exp in selected for exp in expected_any)
        status = "✓" if found_expected else "?"
        
        print(f"\n  Query: \"{query[:50]}...\"")
        print(f"  {status} Top-3: {list(zip(selected, [f'{w:.3f}' for w in weights]))}")
        
        if not found_expected:
            print(f"    (Expected one of: {expected_any})")
    
    # Test 3: Get all similarities
    print("\n[Test 3] Getting all similarities for a query...")
    
    query = "Add a product to my shopping cart"
    all_sims = retriever.get_all_similarities(query)
    sorted_sims = sorted(all_sims.items(), key=lambda x: x[1], reverse=True)
    
    print(f"  Query: \"{query}\"")
    print(f"  Top-5 similarities:")
    for name, sim in sorted_sims[:5]:
        print(f"    {name}: {sim:.4f}")
    
    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)


if __name__ == '__main__':
    main()
