"""
LoRA Composition Strategies

Based on LoraRetriever paper Section 4.2:
1. Mixture of LoRAs (Section 4.2.1): Output-level weighted combination
   x' = (1/n) * Σ_j B_j * A_j * x
   
2. Fusion of LoRAs (Section 4.2.2): Parameter-level averaging
   Θ_fusion = (1/k) * Σ_j Θ_j
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any, Literal
from enum import Enum
from abc import ABC, abstractmethod


class CompositionStrategy(str, Enum):
    """LoRA composition strategy"""
    MIXTURE = "mixture"      # Output-level weighted sum
    FUSION = "fusion"        # Parameter-level averaging


class BaseComposer(ABC):
    """Abstract base class for LoRA composers"""
    
    @abstractmethod
    def compose(
        self,
        model: nn.Module,
        adapter_names: List[str],
        weights: List[float],
        **kwargs
    ) -> Any:
        """
        Compose multiple LoRAs
        
        Args:
            model: Base model with loaded LoRA adapters
            adapter_names: Names of adapters to compose
            weights: Weights for each adapter
            **kwargs: Additional arguments
            
        Returns:
            Composed result (depends on strategy)
        """
        pass


class MixtureComposer(BaseComposer):
    """
    Mixture of LoRAs: Output-level weighted combination
    
    From paper Section 4.2.1:
    For an input x, the output is: x' = (1/n) * Σ_{j=1}^{n} B_j * A_j * x
    
    This aggregates the outputs of each LoRA submodule, effectively blending
    their contributions to form a unified output.
    
    Implementation with Swift:
    - Uses lora_mapping tensor for efficient batch computation
    - Each sample can have different active LoRAs and weights
    """
    
    def __init__(self, use_weighted_average: bool = True):
        """
        Args:
            use_weighted_average: If True, use weights; if False, use uniform averaging
        """
        self.use_weighted_average = use_weighted_average
    
    def compose(
        self,
        model: nn.Module,
        adapter_names: List[str],
        weights: List[float],
        all_adapter_names: Optional[List[str]] = None,
        batch_size: int = 1,
        device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """
        Create lora_mapping tensor for mixture mode inference
        
        The lora_mapping tensor has shape (batch_size, num_adapters) where:
        - Each row corresponds to a sample
        - Each column corresponds to an adapter
        - Values are weights (0 for non-selected adapters)
        
        Args:
            model: Base model (used to get device)
            adapter_names: Selected adapter names
            weights: Corresponding weights
            all_adapter_names: Full list of adapter names in order
            batch_size: Batch size
            device: Target device
            
        Returns:
            lora_mapping tensor of shape (batch_size, num_adapters)
        """
        if device is None:
            device = next(model.parameters()).device
        
        if all_adapter_names is None:
            # Try to get from model
            if hasattr(model, 'peft_config'):
                all_adapter_names = list(model.peft_config.keys())
            elif hasattr(model, 'base_model') and hasattr(model.base_model, 'peft_config'):
                all_adapter_names = list(model.base_model.peft_config.keys())
            else:
                all_adapter_names = adapter_names
        
        num_adapters = len(all_adapter_names)
        lora_mapping = torch.zeros(batch_size, num_adapters, device=device)
        
        # Normalize weights if using weighted average
        if self.use_weighted_average and weights:
            weight_sum = sum(weights)
            if weight_sum > 0:
                weights = [w / weight_sum for w in weights]
        else:
            # Uniform weights
            if adapter_names:
                weights = [1.0 / len(adapter_names)] * len(adapter_names)
        
        # Fill in weights for selected adapters
        for name, weight in zip(adapter_names, weights):
            if name in all_adapter_names:
                idx = all_adapter_names.index(name)
                lora_mapping[:, idx] = weight
        
        return lora_mapping
    
    def get_inference_kwargs(
        self,
        adapter_names: List[str],
        weights: List[float],
        all_adapter_names: List[str],
        batch_size: int = 1,
        device: Optional[torch.device] = None
    ) -> Dict[str, Any]:
        """
        Get kwargs for inference with mixture mode
        
        Returns:
            Dict with 'merging_type', 'lora_mapping', 'mixture_adapter_names'
        """
        lora_mapping = self.compose(
            model=None,  # Not needed for mapping
            adapter_names=adapter_names,
            weights=weights,
            all_adapter_names=all_adapter_names,
            batch_size=batch_size,
            device=device
        )
        
        return {
            'merging_type': 'mixture',
            'lora_mapping': lora_mapping,
            'mixture_adapter_names': all_adapter_names
        }


class FusionComposer(BaseComposer):
    """
    Fusion of LoRAs: Parameter-level averaging
    
    From paper Section 4.2.2:
    Θ_fusion = (1/k) * Σ_{j=1}^{k} Θ_j
    
    This fuses the parameters of LoRAs, allowing the fused parameter
    to function like a single LoRA.
    
    Implementation with Swift:
    - Uses add_weighted_adapter to create merged adapter
    - Merged adapter can be used like a regular single LoRA
    """
    
    def __init__(
        self,
        combination_type: Literal['linear', 'svd', 'cat'] = 'linear',
        merged_adapter_prefix: str = 'fused'
    ):
        """
        Args:
            combination_type: How to combine LoRA parameters
                - 'linear': Weighted average (default, matches paper)
                - 'svd': SVD-based merging
                - 'cat': Concatenation (increases rank)
            merged_adapter_prefix: Prefix for merged adapter names
        """
        self.combination_type = combination_type
        self.merged_adapter_prefix = merged_adapter_prefix
        self._merge_count = 0
    
    def compose(
        self,
        model: nn.Module,
        adapter_names: List[str],
        weights: List[float],
        **kwargs
    ) -> str:
        """
        Create a fused adapter by merging parameters
        
        Args:
            model: Model with loaded LoRA adapters
            adapter_names: Adapters to merge
            weights: Weights for each adapter
            
        Returns:
            Name of the created merged adapter
        """
        if not adapter_names:
            raise ValueError("No adapters to merge")
        
        # Normalize weights
        weight_sum = sum(weights)
        if weight_sum > 0:
            weights = [w / weight_sum for w in weights]
        
        # Create unique merged adapter name
        self._merge_count += 1
        merged_name = f'{self.merged_adapter_prefix}_{self._merge_count}'
        
        # Use add_weighted_adapter
        try:
            model.add_weighted_adapter(
                adapters=adapter_names,
                weights=weights,
                adapter_name=merged_name,
                combination_type=self.combination_type
            )
            
            # Set as active adapter
            if hasattr(model, 'set_adapter'):
                model.set_adapter(merged_name)
            elif hasattr(model, 'base_model') and hasattr(model.base_model, 'set_adapter'):
                model.base_model.set_adapter(merged_name)
            
            return merged_name
            
        except Exception as e:
            print(f"[ERROR] Failed to merge adapters: {e}")
            # Fallback to first adapter
            if adapter_names:
                top_adapter = adapter_names[0]
                if hasattr(model, 'set_adapter'):
                    model.set_adapter(top_adapter)
                return top_adapter
            raise
    
    def cleanup(self, model: nn.Module) -> None:
        """
        Delete all fused adapters to free memory
        
        Args:
            model: Model with fused adapters
        """
        if hasattr(model, 'delete_adapter'):
            for i in range(1, self._merge_count + 1):
                adapter_name = f'{self.merged_adapter_prefix}_{i}'
                try:
                    model.delete_adapter(adapter_name)
                except:
                    pass
        self._merge_count = 0


class BatchInferenceHelper:
    """
    Helper for efficient batch inference with multiple LoRAs
    
    From paper Section 4.3:
    - Process batch of samples with heterogeneous LoRA selections
    - Use LoRA mapping matrix for efficient computation
    """
    
    def __init__(self, composer: BaseComposer):
        self.composer = composer
    
    def prepare_batch(
        self,
        batch_selections: List[Tuple[List[str], List[float]]],
        all_adapter_names: List[str],
        device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """
        Prepare LoRA mapping for a batch of samples with different selections
        
        From paper Equation 3:
        X' = M ◦ (B ◦ A ◦ X)  (for mixture)
        X' = (M ◦ B)(M ◦ A) ◦ X  (for fusion)
        
        Args:
            batch_selections: List of (adapter_names, weights) for each sample
            all_adapter_names: Full ordered list of adapter names
            device: Target device
            
        Returns:
            LoRA mapping tensor of shape (batch_size, num_adapters)
        """
        batch_size = len(batch_selections)
        num_adapters = len(all_adapter_names)
        
        lora_mapping = torch.zeros(batch_size, num_adapters, device=device)
        
        for i, (names, weights) in enumerate(batch_selections):
            # Normalize weights
            weight_sum = sum(weights)
            if weight_sum > 0:
                weights = [w / weight_sum for w in weights]
            
            for name, weight in zip(names, weights):
                if name in all_adapter_names:
                    idx = all_adapter_names.index(name)
                    lora_mapping[i, idx] = weight
        
        return lora_mapping


def create_composer(
    strategy: CompositionStrategy = CompositionStrategy.MIXTURE,
    **kwargs
) -> BaseComposer:
    """
    Factory function to create a composer
    
    Args:
        strategy: Composition strategy (mixture or fusion)
        **kwargs: Additional arguments for the specific composer
        
    Returns:
        Composer instance
    """
    if strategy == CompositionStrategy.MIXTURE:
        return MixtureComposer(**kwargs)
    elif strategy == CompositionStrategy.FUSION:
        return FusionComposer(**kwargs)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
