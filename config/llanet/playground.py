"""
Model Analysis Playground
========================
This module provides utilities to analyze model size, parameters, and FLOPs.
"""

import torch
import torch.nn as nn
from typing import Tuple, Dict, Any, Optional, Union


def analyze_model(
    model: nn.Module,
    input_shape: Tuple[int, ...] = (1, 3, 320, 800),
    model_name: str = "Model",
    use_thop: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Comprehensive model analysis including parameters, size, and FLOPs.

    Args:
        model: PyTorch model to analyze
        input_shape: Input tensor shape (batch_size, channels, height, width)
        model_name: Name of the model for display
        use_thop: Whether to use thop for FLOPs calculation (recommended, True by default)
        verbose: Whether to print detailed statistics

    Returns:
        Dictionary containing all model statistics

    Example:
        >>> from your_config import model
        >>> stats = analyze_model(model, model_name="MyLLANet")
        >>> print(f"Model has {stats['total_params']:,} parameters")
    """
    # Handle OmegaConf DictConfig - need to instantiate the model
    from omegaconf import DictConfig

    if isinstance(model, DictConfig):
        print(f"[Info] model is a DictConfig, attempting to instantiate...")
        from unlanedet.config.instantiate import instantiate

        model = instantiate(model)
        print(f"[Info] Model instantiated successfully")

    stats = {}
    device = next(model.parameters()).device

    # ========================================
    # 1. Parameter Statistics
    # ========================================
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    # Calculate model size
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    model_size_mb = (param_size + buffer_size) / 1024 / 1024

    stats.update(
        {
            "total_params": total_params,
            "trainable_params": trainable_params,
            "frozen_params": frozen_params,
            "model_size_mb": model_size_mb,
            "param_size_mb": param_size / 1024 / 1024,
            "buffer_size_mb": buffer_size / 1024 / 1024,
        }
    )

    # ========================================
    # 2. FLOPs Calculation (using thop)
    # ========================================
    from thop import profile

    # Create dummy input - wrap in dict for LLANet models
    dummy_input = torch.randn(*input_shape).to(device)
    wrapped_input = {"img": dummy_input}

    # Profile the model to get MACs and FLOPs
    # Try both dict input and tensor input
    try:
        flops, profile_params = profile(model, inputs=(wrapped_input,), verbose=False)
    except:
        flops, profile_params = profile(model, inputs=(dummy_input,), verbose=False)

    # thop returns MACs (multiply-accumulate operations)
    # Standard convention: FLOPs = 2 * MACs for most operations
    flops = flops * 2  # Convert MACs to FLOPs

    stats["flops"] = flops
    stats["macs"] = flops // 2
    stats["flops_gflops"] = flops / 1e9
    stats["macs_gmacs"] = (flops // 2) / 1e9

    # ========================================
    # 3. Layer-wise Statistics
    # ========================================
    layer_stats = []
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Only leaf modules
            layer_params = sum(p.numel() for p in module.parameters())
            if layer_params > 0:
                layer_stats.append(
                    {
                        "name": name,
                        "type": module.__class__.__name__,
                        "params": layer_params,
                        "percentage": (layer_params / total_params) * 100,
                    }
                )

    stats["layer_stats"] = layer_stats

    # ========================================
    # 4. Memory Estimate
    # ========================================
    # Estimate activation memory during forward pass
    activation_memory_mb = 0
    try:
        dummy_input = torch.randn(*input_shape).to(device)
        model.eval()

        # Hook to track intermediate activations
        activation_sizes = []
        hooks = []

        def hook_fn(module, input, output):
            if isinstance(output, torch.Tensor):
                activation_sizes.append(output.nelement() * output.element_size())
            elif isinstance(output, (tuple, list)):
                for out in output:
                    if isinstance(out, torch.Tensor):
                        activation_sizes.append(out.nelement() * out.element_size())

        for name, module in model.named_modules():
            if len(list(module.children())) == 0:
                hooks.append(module.register_forward_hook(hook_fn))

        with torch.no_grad():
            try:
                _ = model(wrapped_input)
            except:
                try:
                    _ = model(dummy_input)
                except:
                    pass

        # Remove hooks
        for hook in hooks:
            hook.remove()

        activation_memory_mb = sum(activation_sizes) / 1024 / 1024

    except Exception as e:
        if verbose:
            print(f"[Warning] Activation memory estimation failed: {e}")

    stats["activation_memory_mb"] = activation_memory_mb

    # ========================================
    # 5. Print Summary
    # ========================================
    if verbose:
        print_model_summary(stats, model_name)

    return stats


def print_model_summary(stats: Dict[str, Any], model_name: str = "Model"):
    """Print a formatted summary of model statistics."""
    print("\n" + "=" * 80)
    print(f"Model Analysis Summary: {model_name}")
    print("=" * 80)

    # Basic Information
    print(f"\n[1] Basic Information")
    print(f"{'─' * 80}")
    print(
        f"Total Parameters:      {stats['total_params']:,} ({stats['total_params'] / 1e6:.2f}M)"
    )
    print(
        f"Trainable Parameters:  {stats['trainable_params']:,} ({stats['trainable_params'] / 1e6:.2f}M)"
    )
    print(
        f"Frozen Parameters:     {stats['frozen_params']:,} ({stats['frozen_params'] / 1e6:.2f}M)"
    )
    print(f"Model Size:            {stats['model_size_mb']:.2f} MB")
    print(f"  └─ Parameters:       {stats['param_size_mb']:.2f} MB")
    print(f"  └─ Buffers:          {stats['buffer_size_mb']:.2f} MB")

    # FLOPs Information
    print(f"\n[2] Computational Complexity")
    print(f"{'─' * 80}")
    if "flops" in stats and stats["flops"] > 0:
        print(
            f"FLOPs:                 {stats['flops']:,} ({stats['flops_gflops']:.2f} GFLOPs)"
        )
        if "macs" in stats:
            print(
                f"MACs:                  {stats['macs']:,} ({stats['macs_gmacs']:.2f} GMACs)"
            )

    # Memory Information
    print(f"\n[3] Memory Usage")
    print(f"{'─' * 80}")
    print(f"Parameter Memory:      {stats['param_size_mb']:.2f} MB")
    print(f"Activation Memory:     {stats['activation_memory_mb']:.2f} MB (estimated)")
    print(
        f"Total Memory:          {stats['param_size_mb'] + stats['activation_memory_mb']:.2f} MB"
    )

    # Top Layers by Parameter Count
    print(f"\n[4] Top 10 Layers by Parameter Count")
    print(f"{'─' * 80}")
    top_layers = sorted(stats["layer_stats"], key=lambda x: x["params"], reverse=True)[
        :10
    ]
    for i, layer in enumerate(top_layers, 1):
        print(f"{i:2d}. {layer['type']:20s} - {layer['name']:50s}")
        print(f"    Parameters: {layer['params']:10,} ({layer['percentage']:5.2f}%)")

    print("=" * 80 + "\n")


def compare_models(
    models: Dict[str, nn.Module],
    input_shape: Tuple[int, ...] = (1, 3, 320, 800),
    verbose: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """
    Compare multiple models side by side.

    Args:
        models: Dictionary mapping model names to model instances
        input_shape: Input tensor shape
        verbose: Whether to print comparison table

    Returns:
        Dictionary of statistics for each model

    Example:
        >>> from config.resnet18_openlane import model as resnet_model
        >>> from config.mobilenetv4_openlane import model as mobile_model
        >>> compare = compare_models({
        ...     "ResNet18-LLANet": resnet_model,
        ...     "MobileNetV4-LLANet": mobile_model
        ... })
    """
    results = {}

    for name, model in models.items():
        try:
            results[name] = analyze_model(
                model, input_shape=input_shape, model_name=name, verbose=False
            )
        except Exception as e:
            print(f"[Error] Failed to analyze {name}: {e}")
            results[name] = None

    if verbose:
        print_comparison_table(results)

    return results


def print_comparison_table(results: Dict[str, Dict[str, Any]]):
    """Print a comparison table for multiple models."""
    print("\n" + "=" * 100)
    print("Model Comparison")
    print("=" * 100)

    headers = ["Model", "Params (M)", "Size (MB)", "FLOPs (G)", "Trainable", "Frozen"]
    print(
        f"{headers[0]:<25} {headers[1]:<15} {headers[2]:<15} {headers[3]:<15} {headers[4]:<12} {headers[5]:<12}"
    )
    print("-" * 100)

    for name, stats in results.items():
        if stats:
            print(
                f"{name:<25} "
                f"{stats['total_params'] / 1e6:<15.2f} "
                f"{stats['model_size_mb']:<15.2f} "
                f"{stats.get('flops_gflops', 0):<15.2f} "
                f"{stats['trainable_params'] / 1e6:<12.2f} "
                f"{stats['frozen_params'] / 1e6:<12.2f}"
            )

    print("=" * 100 + "\n")


def analyze_single_model(model, input_shape):
    """
    Analyze a single model.

    Args:
        model: The model to analyze.
        input_shape: The input shape for the model.
    """
    stats = analyze_model(
        model, input_shape=input_shape, model_name="ResNet18-LLANet", use_thop=True
    )
    return stats


def compare_models_with_single_model(models, input_shape):
    """
    Compare multiple models with a single model.

    Args:
        models: Dictionary of model names and model objects.
    """
    # Analyze the single model first
    single_model_stats = analyze_single_model(
        models[list(models.keys())[0]], input_shape
    )

    # Compare other models with the single model
    results = compare_models(models, input_shape)
    return results


# Example Usage
# ========================================
if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")

    print("=== LLANet Model Analysis ===\n")

    # Import models at runtime to avoid circular imports
    try:
        print("Loading ResNet18-LLANet...")
        resnet18_module = __import__(
            "config.llanet.resnet18_openlane", fromlist=["model"]
        )
        resnet18_model = resnet18_module.model
        print("ResNet18-LLANet loaded!\n")
    except Exception as e:
        print(f"Failed to load ResNet18: {e}")
        import sys

        sys.exit(1)

    try:
        print("Loading MobileNetV4-LLANet...")
        mobile_module = __import__(
            "config.llanet.mobilenetv4_openlane", fromlist=["model"]
        )
        mobilenetv4_model = mobile_module.model
        print("MobileNetV4-LLANet loaded!\n")
    except Exception as e:
        print(f"Failed to load MobileNetV4: {e}")
        mobilenetv4_model = None

    # Analyze ResNet18
    print("Analyzing ResNet18-LLANet...")
    analyze_model(
        resnet18_model, input_shape=(1, 3, 320, 800), model_name="ResNet18-LLANet"
    )

    # Compare both models if available
    if mobilenetv4_model is not None:
        print("\nComparing Both Models...")
        compare_models(
            {"ResNet18-LLANet": resnet18_model, "MobileNetV4-LLANet": mobilenetv4_model}
        )
