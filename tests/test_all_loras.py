#!/usr/bin/env python
"""
Comprehensive LoRA Retrieval Test
Tests all 14 app LoRAs and 5 category LoRAs for retrieval accuracy
"""

import sys
import os
import json

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from lora_retriever import LoraRetriever, LoraRetrieverConfig


def load_configs(config_path):
    """Load LoRA configs and make paths absolute"""
    with open(config_path, 'r') as f:
        configs = json.load(f)
    for cfg in configs:
        if 'embedding_path' in cfg and not cfg['embedding_path'].startswith('/'):
            cfg['embedding_path'] = os.path.join(PROJECT_ROOT, cfg['embedding_path'])
    return configs


def test_app_loras():
    """Test all 14 app LoRAs with specific queries"""
    print("\n" + "=" * 70)
    print("APP LoRA RETRIEVAL TEST (14 LoRAs)")
    print("=" * 70)
    
    config_path = os.path.join(PROJECT_ROOT, 'config', 'app_loras_config_internvl2.json')
    lora_configs = load_configs(config_path)
    
    config = LoraRetrieverConfig(
        lora_configs=lora_configs,
        model_path='/home/hmpiao/hmpiao/jina-embeddings-v4',
        device='cuda',
        top_k=3
    )
    
    retriever = LoraRetriever(config)
    retriever.load_lora_embeddings(lora_configs)
    
    # Test queries - each should match its corresponding LoRA
    test_cases = [
        # (query, expected_top1_lora)
        ("Search for running shoes and sportswear on Adidas app", "app_lora_adidas"),
        ("Add product to cart and checkout on Amazon", "app_lora_amazon"),
        ("Create a new calendar event for tomorrow meeting", "app_lora_calendar"),
        ("Set alarm for 7am and create timer for cooking", "app_lora_clock"),
        ("Browse sports equipment and outdoor gear on Decathlon", "app_lora_decathlon"),
        ("List an item for sale and bid on products on eBay", "app_lora_ebay"),
        ("Search for handmade crafts and vintage items on Etsy", "app_lora_etsy"),
        ("Search for mobile phones and electronics on Flipkart", "app_lora_flipkart"),
        ("Compose email and check inbox in Gmail", "app_lora_gmail"),
        ("Upload files and share documents in Google Drive", "app_lora_google_drive"),
        ("Navigate to destination and search for nearby restaurants on Google Maps", "app_lora_google_maps"),
        ("Find cooking recipes and meal preparation on Kitchen Stories", "app_lora_kitchen_stories"),
        ("Create reminder and set notification for task", "app_lora_reminder"),
        ("Watch videos and search for music on YouTube", "app_lora_youtube"),
    ]
    
    correct = 0
    total = len(test_cases)
    
    print(f"\nTesting {total} queries against {len(lora_configs)} LoRAs:\n")
    
    for query, expected in test_cases:
        selected, weights = retriever.retrieve_with_weights(query, top_k=3)
        top1 = selected[0] if selected else None
        is_correct = top1 == expected
        
        status = "✓" if is_correct else "✗"
        if is_correct:
            correct += 1
        
        print(f"{status} Query: \"{query[:50]}...\"")
        print(f"   Expected: {expected}")
        print(f"   Got:      {top1} (weight: {weights[0]:.3f})")
        if not is_correct:
            print(f"   Top-3:    {list(zip(selected, [f'{w:.3f}' for w in weights]))}")
        print()
    
    accuracy = correct / total * 100
    print(f"\nApp LoRA Accuracy: {correct}/{total} ({accuracy:.1f}%)")
    return correct, total


def test_category_loras():
    """Test all 5 category LoRAs with specific queries"""
    print("\n" + "=" * 70)
    print("CATEGORY LoRA RETRIEVAL TEST (5 LoRAs)")
    print("=" * 70)
    
    config_path = os.path.join(PROJECT_ROOT, 'config', 'category_loras_config_internvl2.json')
    lora_configs = load_configs(config_path)
    
    config = LoraRetrieverConfig(
        lora_configs=lora_configs,
        model_path='/home/hmpiao/hmpiao/jina-embeddings-v4',
        device='cuda',
        top_k=3
    )
    
    retriever = LoraRetriever(config)
    retriever.load_lora_embeddings(lora_configs)
    
    # Test queries for each category
    test_cases = [
        # Entertainment
        ("Watch videos and stream music online", "category_lora_entertainment"),
        ("Play mobile games and watch live streams", "category_lora_entertainment"),
        
        # Lives (daily life)
        ("Set alarm and check weather forecast", "category_lora_lives"),
        ("Create reminder for daily tasks", "category_lora_lives"),
        
        # Office
        ("Send email and schedule meeting in calendar", "category_lora_office"),
        ("Upload documents and share files with colleagues", "category_lora_office"),
        
        # Shopping
        ("Add items to cart and checkout online", "category_lora_shopping"),
        ("Search for products and compare prices", "category_lora_shopping"),
        
        # Traveling
        ("Navigate to destination and find nearby hotels", "category_lora_traveling"),
        ("Search for flights and book accommodation", "category_lora_traveling"),
    ]
    
    correct = 0
    total = len(test_cases)
    
    print(f"\nTesting {total} queries against {len(lora_configs)} category LoRAs:\n")
    
    for query, expected in test_cases:
        selected, weights = retriever.retrieve_with_weights(query, top_k=3)
        top1 = selected[0] if selected else None
        is_correct = top1 == expected
        
        status = "✓" if is_correct else "✗"
        if is_correct:
            correct += 1
        
        print(f"{status} Query: \"{query[:50]}...\"")
        print(f"   Expected: {expected}")
        print(f"   Got:      {top1} (weight: {weights[0]:.3f})")
        if not is_correct:
            print(f"   Top-3:    {list(zip(selected, [f'{w:.3f}' for w in weights]))}")
        print()
    
    accuracy = correct / total * 100
    print(f"\nCategory LoRA Accuracy: {correct}/{total} ({accuracy:.1f}%)")
    return correct, total


def test_mixed_loras():
    """Test with both app and category LoRAs loaded together"""
    print("\n" + "=" * 70)
    print("MIXED LoRA RETRIEVAL TEST (App + Category)")
    print("=" * 70)
    
    app_config_path = os.path.join(PROJECT_ROOT, 'config', 'app_loras_config_internvl2.json')
    cat_config_path = os.path.join(PROJECT_ROOT, 'config', 'category_loras_config_internvl2.json')
    
    app_configs = load_configs(app_config_path)
    cat_configs = load_configs(cat_config_path)
    all_configs = app_configs + cat_configs
    
    config = LoraRetrieverConfig(
        lora_configs=all_configs,
        model_path='/home/hmpiao/hmpiao/jina-embeddings-v4',
        device='cuda',
        top_k=5
    )
    
    retriever = LoraRetriever(config)
    retriever.load_lora_embeddings(all_configs)
    
    # Test queries that should distinguish between app and category level
    test_cases = [
        ("Watch YouTube videos", "app_lora_youtube", "category_lora_entertainment"),
        ("Send Gmail email", "app_lora_gmail", "category_lora_office"),
        ("Shop on Amazon", "app_lora_amazon", "category_lora_shopping"),
        ("Navigate with Google Maps", "app_lora_google_maps", "category_lora_traveling"),
        ("Set alarm on clock app", "app_lora_clock", "category_lora_lives"),
    ]
    
    print(f"\nTesting {len(test_cases)} mixed queries against {len(all_configs)} LoRAs:\n")
    
    for query, expected_app, expected_cat in test_cases:
        selected, weights = retriever.retrieve_with_weights(query, top_k=5)
        
        app_found = expected_app in selected
        cat_found = expected_cat in selected
        
        app_rank = selected.index(expected_app) + 1 if app_found else "N/A"
        cat_rank = selected.index(expected_cat) + 1 if cat_found else "N/A"
        
        print(f"Query: \"{query}\"")
        print(f"   App LoRA  ({expected_app}): rank={app_rank}")
        print(f"   Cat LoRA  ({expected_cat}): rank={cat_rank}")
        print(f"   Top-5: {selected}")
        print()


def main():
    print("=" * 70)
    print("COMPREHENSIVE LORA RETRIEVAL TEST")
    print("=" * 70)
    print("Testing retrieval accuracy for InternVL2 LoRAs")
    print("Embeddings generated from InternVL2 LoRA training data")
    print("=" * 70)
    
    # Test App LoRAs
    app_correct, app_total = test_app_loras()
    
    # Test Category LoRAs  
    cat_correct, cat_total = test_category_loras()
    
    # Test Mixed
    test_mixed_loras()
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"App LoRA Accuracy:      {app_correct}/{app_total} ({app_correct/app_total*100:.1f}%)")
    print(f"Category LoRA Accuracy: {cat_correct}/{cat_total} ({cat_correct/cat_total*100:.1f}%)")
    print(f"Overall Accuracy:       {app_correct+cat_correct}/{app_total+cat_total} ({(app_correct+cat_correct)/(app_total+cat_total)*100:.1f}%)")
    print("=" * 70)


if __name__ == '__main__':
    main()
