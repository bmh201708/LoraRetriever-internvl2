"""
LoraRetriever: Core retrieval and composition module
Based on the paper "LoraRetriever: Input-Aware LoRA Retrieval and Composition"

Key Components:
1. LoraRetriever: Embedding-based LoRA retrieval using jina-embeddings-v4
2. MixtureComposer: Output-level weighted combination
3. FusionComposer: Parameter-level averaging
"""

from .retriever import LoraRetriever, LoraRetrieverConfig
from .composition import MixtureComposer, FusionComposer, CompositionStrategy

__all__ = [
    'LoraRetriever',
    'LoraRetrieverConfig',
    'MixtureComposer',
    'FusionComposer',
    'CompositionStrategy',
]
