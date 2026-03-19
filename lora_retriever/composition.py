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

    优化策略：
    - 第一次调用时，使用 add_weighted_adapter 创建 fused adapter
    - 后续调用时，直接更新 fused adapter 的参数，避免重复调用 set_active_adapters
    - 详见 docs/fusion_performance_issue.md
    """

    FUSED_ADAPTER_NAME = "fused_lora"

    def __init__(
        self,
        combination_type: Literal['linear', 'svd', 'cat'] = 'linear',
        merged_adapter_prefix: str = 'fused'
    ):
        """
        Args:
            combination_type: How to combine LoRA parameters
                - 'linear': Weighted average (default, matches paper)
            merged_adapter_prefix: Unused, kept for API compatibility
        """
        self.combination_type = combination_type
        self._initialized = False
        self._model_ref = None

    def compose(
        self,
        model: nn.Module,
        adapter_names: List[str],
        weights: List[float],
        **kwargs
    ) -> str:
        """
        Create or update a fused adapter by merging parameters

        Args:
            model: Model with loaded LoRA adapters
            adapter_names: Adapters to merge
            weights: Weights for each adapter

        Returns:
            Name of the fused adapter
        """
        import time

        if not adapter_names:
            raise ValueError("No adapters to merge")

        # Normalize weights
        weight_sum = sum(weights)
        if weight_sum > 0:
            weights = [w / weight_sum for w in weights]

        merged_name = self.FUSED_ADAPTER_NAME

        try:
            if not self._initialized:
                # 第一次：使用 add_weighted_adapter 创建 fused adapter 的层结构
                print(f"[FUSION DEBUG] 第一次调用，创建 fused adapter...")
                print(f"[FUSION DEBUG] adapter_names: {adapter_names}")
                print(f"[FUSION DEBUG] weights: {weights}")
                t0 = time.time()
                try:
                    model.add_weighted_adapter(
                        adapters=adapter_names,
                        weights=weights,
                        adapter_name=merged_name,
                        combination_type=self.combination_type
                    )
                except Exception as e:
                    # linear/ties/dare_* 在 PEFT 中要求所有 adapter 的 rank 相同
                    # 对 mixed-rank 场景自动回退到 svd 创建 fused 结构
                    err = str(e)
                    linear_like = {'linear', 'ties', 'dare_ties', 'dare_linear'}
                    if ('same r value' in err) and (self.combination_type in linear_like):
                        fallback_rank = self._infer_max_rank(model, adapter_names)
                        print(
                            f"[FUSION DEBUG] 检测到 mixed-rank，{self.combination_type} -> svd 回退, "
                            f"svd_rank={fallback_rank}"
                        )
                        model.add_weighted_adapter(
                            adapters=adapter_names,
                            weights=weights,
                            adapter_name=merged_name,
                            combination_type='svd',
                            svd_rank=fallback_rank
                        )
                    else:
                        raise
                print(f"[FUSION DEBUG] add_weighted_adapter 耗时: {time.time() - t0:.2f}s")
                # 使用 weighted-delta + SVD 方式重算 fused 参数，兼容不同 rank
                self._fix_fused_scaling(model, adapter_names, weights, merged_name)
                self._update_fused_params(model, adapter_names, weights, merged_name)
                self._ensure_fused_active(model, merged_name)
                self._initialized = True
                self._model_ref = model
                print(f"[FUSION DEBUG] 第一次调用完成，_initialized = True")
            else:
                # 后续：直接更新 fused adapter 的参数
                print(f"[FUSION DEBUG] 后续调用，直接更新参数...")
                print(f"[FUSION DEBUG] adapter_names: {adapter_names}")
                print(f"[FUSION DEBUG] weights: {weights}")
                t0 = time.time()
                self._fix_fused_scaling(model, adapter_names, weights, merged_name)
                self._update_fused_params(model, adapter_names, weights, merged_name)
                print(f"[FUSION DEBUG] _update_fused_params 耗时: {time.time() - t0:.2f}s")
                t1 = time.time()
                self._ensure_fused_active(model, merged_name)
                print(f"[FUSION DEBUG] _ensure_fused_active 耗时: {time.time() - t1:.2f}s")
                print(f"[FUSION DEBUG] 后续调用完成")

            return merged_name

        except Exception as e:
            print(f"[ERROR] Failed to merge adapters: {e}")
            import traceback
            traceback.print_exc()
            # Fallback to first adapter
            if adapter_names:
                top_adapter = adapter_names[0]
                if hasattr(model, 'set_active_adapters'):
                    model.set_active_adapters(top_adapter)
                return top_adapter
            raise

    def _infer_max_rank(self, model: nn.Module, adapter_names: List[str]) -> int:
        """Infer max rank among selected adapters from loaded LoRA layers."""
        from peft.tuners.lora import LoraLayer

        max_rank = 0
        for module in model.modules():
            if not isinstance(module, LoraLayer):
                continue
            lora_A = getattr(module, 'lora_A', {})
            for name in adapter_names:
                if name in lora_A:
                    rank = int(lora_A[name].weight.shape[0])
                    if rank > max_rank:
                        max_rank = rank

        if max_rank > 0:
            return max_rank

        # Fallback by config if layer scan fails
        peft_cfg = getattr(model, 'peft_config', None)
        if isinstance(peft_cfg, dict):
            for name in adapter_names:
                cfg = peft_cfg.get(name)
                rank = int(getattr(cfg, 'r', 0) or 0) if cfg is not None else 0
                if rank > max_rank:
                    max_rank = rank
        return max_rank if max_rank > 0 else 8

    def _update_fused_params(
        self,
        model: nn.Module,
        adapter_names: List[str],
        weights: List[float],
        fused_name: str
    ) -> None:
        """
        直接更新 fused adapter 的参数，避免调用 add_weighted_adapter

        Args:
            model: Model with loaded LoRA adapters
            adapter_names: Source adapters to merge
            weights: Weights for each adapter
            fused_name: Name of the fused adapter to update
        """
        from peft.tuners.lora import LoraLayer

        updated_layers = 0
        skipped_not_lora = 0
        skipped_no_fused = 0

        for module in model.modules():
            if not isinstance(module, LoraLayer):
                skipped_not_lora += 1
                continue

            # 检查 fused adapter 是否存在于此层
            if not hasattr(module, 'lora_A') or fused_name not in module.lora_A:
                skipped_no_fused += 1
                continue

            # 检查所有源 adapter 是否存在于此层
            valid_adapters = [name for name in adapter_names if name in module.lora_A]
            if not valid_adapters:
                continue

            # 重新计算权重（只针对此层存在的adapter）
            valid_weights = [weights[adapter_names.index(name)] for name in valid_adapters]
            weight_sum = sum(valid_weights)
            if weight_sum > 0:
                valid_weights = [w / weight_sum for w in valid_weights]

            # mixed-rank 兼容：先在 delta 空间加权，再用 SVD 回写到 fused rank
            fused_A_weight = module.lora_A[fused_name].weight.data
            fused_B_weight = module.lora_B[fused_name].weight.data
            target_rank = int(fused_A_weight.shape[0])
            target_scale = float(module.scaling.get(fused_name, 1.0))
            if abs(target_scale) < 1e-8:
                target_scale = 1.0

            merged_delta = None
            for name, w in zip(valid_adapters, valid_weights):
                src_A = module.lora_A[name].weight.data.to(dtype=torch.float32)
                src_B = module.lora_B[name].weight.data.to(dtype=torch.float32)
                src_scale = float(module.scaling.get(name, 1.0))
                cur_delta = (src_B @ src_A) * (src_scale * float(w))
                if merged_delta is None:
                    merged_delta = cur_delta
                else:
                    merged_delta += cur_delta

            if merged_delta is None:
                continue

            normalized_delta = merged_delta / float(target_scale)
            u, s, vh = torch.linalg.svd(normalized_delta, full_matrices=False)
            r = min(target_rank, int(s.shape[0]))

            s_sqrt = torch.sqrt(torch.clamp(s[:r], min=0))
            new_b = u[:, :r] * s_sqrt.unsqueeze(0)      # [out, r]
            new_a = s_sqrt.unsqueeze(1) * vh[:r, :]     # [r, in]

            fused_A_weight.zero_()
            fused_B_weight.zero_()

            new_a = new_a.to(device=fused_A_weight.device, dtype=fused_A_weight.dtype)
            new_b = new_b.to(device=fused_B_weight.device, dtype=fused_B_weight.dtype)
            fused_A_weight[:r, :].copy_(new_a)
            fused_B_weight[:, :r].copy_(new_b)

            updated_layers += 1

        print(f"[FUSION DEBUG] _update_fused_params: 更新了 {updated_layers} 个LoRA层")

    def _fix_fused_scaling(
        self,
        model: nn.Module,
        adapter_names: List[str],
        weights: List[float],
        fused_name: str
    ) -> None:
        """
        修正 fused adapter 的 scaling 值（按选中 adapter 的 scaling 加权平均）。
        """
        from peft.tuners.lora import LoraLayer

        fixed = False
        for module in model.modules():
            if not isinstance(module, LoraLayer):
                continue
            if fused_name not in getattr(module, 'scaling', {}):
                continue

            valid_pairs = [(n, w) for n, w in zip(adapter_names, weights) if n in module.scaling]
            if not valid_pairs:
                continue
            w_sum = sum(w for _, w in valid_pairs)
            if w_sum <= 0:
                continue
            normalized = [(n, w / w_sum) for n, w in valid_pairs]
            target_scaling = sum(float(module.scaling[n]) * float(w) for n, w in normalized)

            if abs(target_scaling) < 1e-8:
                target_scaling = 1.0

            if module.scaling[fused_name] != target_scaling:
                module.scaling[fused_name] = target_scaling
                if not fixed:
                    print(f"[FUSION DEBUG] 修正 scaling -> {target_scaling}")
                    fixed = True

    def _ensure_fused_active(self, model: nn.Module, fused_name: str) -> None:
        """
        确保 fused adapter 在 PEFT 层面是 active adapter

        关键：必须设置 PEFT LoraLayer 的 _active_adapter 属性，
        否则标准 forward 会使用错误的 adapter。
        active_adapter 是只读 property，需要通过 set_adapter() 或 _active_adapter 设置。
        """
        from peft.tuners.lora import LoraLayer

        count = 0
        for module in model.modules():
            if isinstance(module, LoraLayer):
                if fused_name in getattr(module, 'lora_A', {}):
                    # 使用 set_adapter 方法（如果有），否则直接设置 _active_adapter
                    if hasattr(module, 'set_adapter'):
                        module.set_adapter([fused_name])
                    else:
                        module._active_adapter = [fused_name]
                    count += 1
        print(f"[FUSION DEBUG] _ensure_fused_active: 设置了 {count} 个LoRA层的 active_adapter = ['{fused_name}']")

    def cleanup(self, model: nn.Module) -> None:
        """
        Reset state (fused adapter remains in model but can be reused)
        """
        self._initialized = False
        self._model_ref = None


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
