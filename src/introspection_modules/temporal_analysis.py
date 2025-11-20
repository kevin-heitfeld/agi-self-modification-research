"""
Temporal Analysis Module - Advanced introspection for patterns across time/inputs

Provides tools for comparing activations across multiple inputs, tracking
information flow through layers, and analyzing self-generation patterns.

Functions:
    compare_activations(model, tokenizer, texts, layer_names) - Compare across multiple inputs
    track_layer_flow(model, tokenizer, text, layer_names) - Track info flow through layers
    capture_generation_activations(model, tokenizer, prompt, max_new_tokens) - Monitor self-generation
    compute_activation_similarity(activations1, activations2) - Compute similarity metrics
    detect_activation_anomalies(model, tokenizer, text, layer_names, baseline_texts) - Find unusual patterns

Author: AGI Self-Modification Research Team
Date: November 20, 2025
"""

from typing import Dict, List, Any, Optional, Union, Tuple
import torch
import torch.nn as nn
import numpy as np
from ..introspection.activation_monitor import ActivationMonitor


# Cache monitors to avoid recreating them
_monitor_cache: Dict[int, ActivationMonitor] = {}


def _get_monitor(model: nn.Module, tokenizer: Any) -> ActivationMonitor:
    """Get or create a cached ActivationMonitor for a model."""
    model_id = id(model)
    if model_id not in _monitor_cache:
        _monitor_cache[model_id] = ActivationMonitor(model, tokenizer)
    return _monitor_cache[model_id]


def compare_activations(
    model: nn.Module,
    tokenizer: Any,
    texts: List[str],
    layer_names: Union[str, List[str]]
) -> Dict[str, Any]:
    """
    Compare activations across multiple input texts to detect patterns.
    
    This helps answer questions like:
    - Do certain inputs cause systematically different activation patterns?
    - Which layers show the most variation across inputs?
    - Are there consistent patterns that emerge regardless of input?
    
    Args:
        model: PyTorch model to monitor
        tokenizer: Tokenizer for the model
        texts: List of input texts to compare (2-10 recommended)
        layer_names: Layer name(s) to analyze. Can be:
                    - Single layer name (str): "model.layers.0"
                    - List of layer names: ['model.layers.0', 'model.layers.5']
    
    Returns:
        Dictionary with structure:
            {
                'layer_name': {
                    'per_input': [
                        {'text': '...', 'mean': ..., 'std': ..., 'max': ..., 'min': ..., 'sparsity': ...},
                        ...
                    ],
                    'comparison': {
                        'mean_across_inputs': float,
                        'std_across_inputs': float,
                        'variation_coefficient': float,  # How much variation across inputs
                        'max_difference': float,  # Max - min of means
                        'similarity_to_first': [1.0, 0.95, ...],  # Cosine similarity to first input
                    }
                }
            }
    
    Example:
        >>> # Compare uncertain vs certain processing
        >>> results = compare_activations(
        ...     model, tokenizer,
        ...     ["I'm completely sure", "I'm uncertain", "Maybe yes, maybe no"],
        ...     ['model.layers.2', 'model.layers.10']
        ... )
        >>> 
        >>> for layer, data in results.items():
        ...     print(f"\n{layer}:")
        ...     print(f"  Variation coefficient: {data['comparison']['variation_coefficient']:.4f}")
        ...     for i, input_stats in enumerate(data['per_input']):
        ...         print(f"  Input {i} mean: {input_stats['mean']:.4f}")
    """
    if not texts or len(texts) < 2:
        raise ValueError("Need at least 2 texts to compare")
    
    if len(texts) > 20:
        raise ValueError("Too many texts (max 20). This could cause memory issues.")
    
    # Convert single string to list for uniform processing
    if isinstance(layer_names, str):
        layer_names = [layer_names]
    
    monitor = _get_monitor(model, tokenizer)
    results = {}
    
    for layer_name in layer_names:
        per_input_stats = []
        raw_activations = []
        
        # Capture activations for each text
        for text in texts:
            monitor.capture_activations(text, [layer_name])
            stats = monitor.get_activation_statistics(layer_name)
            
            # Handle both single and list returns
            if isinstance(stats, list):
                stat_dict = stats[0] if stats else {}
            else:
                stat_dict = stats
            
            # Check for errors
            if isinstance(stat_dict, dict) and 'error' in stat_dict:
                raise ValueError(f"Error capturing activations for '{layer_name}': {stat_dict['error']}")
            
            stat_dict['text'] = text[:50] + ('...' if len(text) > 50 else '')
            per_input_stats.append(stat_dict)
            
            # Store raw activation for similarity computation
            # We'll use the last captured activation
            if hasattr(monitor, 'activations') and layer_name in monitor.activations:
                raw_activations.append(monitor.activations[layer_name])
        
        # Compute comparison metrics
        means = [s['mean'] for s in per_input_stats]
        mean_of_means = np.mean(means)
        std_of_means = np.std(means)
        variation_coefficient = std_of_means / mean_of_means if mean_of_means != 0 else 0
        max_difference = max(means) - min(means)
        
        # Compute cosine similarity to first input
        similarities = []
        if raw_activations:
            first_activation = raw_activations[0].flatten()
            for act in raw_activations:
                flat_act = act.flatten()
                # Cosine similarity
                similarity = float(torch.nn.functional.cosine_similarity(
                    first_activation.unsqueeze(0),
                    flat_act.unsqueeze(0)
                ).item())
                similarities.append(similarity)
        
        comparison = {
            'mean_across_inputs': float(mean_of_means),
            'std_across_inputs': float(std_of_means),
            'variation_coefficient': float(variation_coefficient),
            'max_difference': float(max_difference),
            'similarity_to_first': similarities,
            'note': f"Variation coefficient: {variation_coefficient:.4f}. Higher = more variation across inputs."
        }
        
        results[layer_name] = {
            'per_input': per_input_stats,
            'comparison': comparison
        }
    
    return results


def track_layer_flow(
    model: nn.Module,
    tokenizer: Any,
    text: str,
    layer_names: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Track how information transforms as it flows through layers.
    
    This helps answer questions like:
    - Where in the network does specific information emerge or disappear?
    - How do activation patterns change from early to late layers?
    - Which layers show the most dramatic transformations?
    
    Args:
        model: PyTorch model to monitor
        tokenizer: Tokenizer for the model
        text: Input text to process
        layer_names: List of layer names to track (in order). If None, samples key layers.
                    For best results, provide layers in sequential order.
    
    Returns:
        Dictionary with structure:
            {
                'text': str,
                'layers': [
                    {
                        'name': 'model.layers.0',
                        'depth': 0,
                        'mean': ...,
                        'std': ...,
                        'sparsity': ...,
                        'change_from_previous': {
                            'mean_delta': ...,
                            'std_delta': ...,
                            'sparsity_delta': ...,
                            'magnitude': ...,  # How much the activations changed
                        }
                    },
                    ...
                ],
                'summary': {
                    'total_layers': int,
                    'max_change_layer': str,  # Layer with biggest transformation
                    'max_change_magnitude': float,
                    'increasing_trend': bool,  # Are activations generally increasing/decreasing?
                }
            }
    
    Example:
        >>> # Track through early, middle, and late layers
        >>> flow = track_layer_flow(
        ...     model, tokenizer,
        ...     "What is consciousness?",
        ...     ['model.layers.0', 'model.layers.5', 'model.layers.10', 'model.layers.15']
        ... )
        >>> 
        >>> print(f"Max change at: {flow['summary']['max_change_layer']}")
        >>> for layer_info in flow['layers']:
        ...     if 'change_from_previous' in layer_info:
        ...         print(f"{layer_info['name']}: magnitude={layer_info['change_from_previous']['magnitude']:.4f}")
    """
    monitor = _get_monitor(model, tokenizer)
    
    # If no layers specified, sample key layers (early, middle, late)
    if layer_names is None:
        all_layers = [name for name, _ in model.named_modules() if 'layers.' in name and name.count('.') == 2]
        all_layers = sorted(list(set(all_layers)))  # Unique and sorted
        
        if len(all_layers) > 8:
            # Sample approximately evenly
            step = len(all_layers) // 8
            layer_names = [all_layers[i] for i in range(0, len(all_layers), step)]
        else:
            layer_names = all_layers[:8]  # Use first 8
    
    if not layer_names:
        raise ValueError("No valid layers found to track")
    
    # Capture activations for all layers at once
    monitor.capture_activations(text, layer_names)
    
    # Collect statistics for each layer
    layer_results = []
    previous_stats = None
    
    for depth, layer_name in enumerate(layer_names):
        stats = monitor.get_activation_statistics(layer_name)
        
        # Handle both single and list returns
        if isinstance(stats, list):
            stat_dict = stats[0] if stats else {}
        else:
            stat_dict = stats
        
        # Check for errors
        if isinstance(stat_dict, dict) and 'error' in stat_dict:
            raise ValueError(f"Error capturing activations for '{layer_name}': {stat_dict['error']}")
        
        layer_info = {
            'name': layer_name,
            'depth': depth,
            'mean': stat_dict.get('mean', 0),
            'std': stat_dict.get('std', 0),
            'sparsity': stat_dict.get('sparsity', 0),
            'max': stat_dict.get('max', 0),
            'min': stat_dict.get('min', 0),
        }
        
        # Compare to previous layer
        if previous_stats is not None:
            mean_delta = layer_info['mean'] - previous_stats['mean']
            std_delta = layer_info['std'] - previous_stats['std']
            sparsity_delta = layer_info['sparsity'] - previous_stats['sparsity']
            magnitude = np.sqrt(mean_delta**2 + std_delta**2)
            
            layer_info['change_from_previous'] = {
                'mean_delta': float(mean_delta),
                'std_delta': float(std_delta),
                'sparsity_delta': float(sparsity_delta),
                'magnitude': float(magnitude),
                'interpretation': f"{'Increased' if mean_delta > 0 else 'Decreased'} by {abs(mean_delta):.4f}"
            }
        
        layer_results.append(layer_info)
        previous_stats = layer_info
    
    # Compute summary statistics
    magnitudes = [l['change_from_previous']['magnitude'] for l in layer_results if 'change_from_previous' in l]
    max_change_idx = np.argmax(magnitudes) + 1 if magnitudes else 0  # +1 because first layer has no change
    
    means = [l['mean'] for l in layer_results]
    increasing_trend = means[-1] > means[0] if len(means) > 1 else False
    
    summary = {
        'total_layers': len(layer_results),
        'max_change_layer': layer_results[max_change_idx]['name'] if magnitudes else 'N/A',
        'max_change_magnitude': float(max(magnitudes)) if magnitudes else 0.0,
        'increasing_trend': increasing_trend,
        'mean_trend': f"{'Increasing' if increasing_trend else 'Decreasing'} from {means[0]:.4f} to {means[-1]:.4f}",
    }
    
    return {
        'text': text[:100] + ('...' if len(text) > 100 else ''),
        'layers': layer_results,
        'summary': summary
    }


def capture_generation_activations(
    model: nn.Module,
    tokenizer: Any,
    prompt: str,
    layer_names: Union[str, List[str]],
    max_new_tokens: int = 20
) -> Dict[str, Any]:
    """
    Monitor activations during the model's own text generation process.
    
    This is DIFFERENT from capture_activations() which only processes existing text.
    This function watches the model generate NEW tokens and captures activations
    at each generation step.
    
    This helps answer questions like:
    - How do my activations change as I generate a response?
    - Do I show different patterns when generating confident vs uncertain text?
    - Can I detect when I'm about to generate something creative vs routine?
    
    Args:
        model: PyTorch model to monitor
        tokenizer: Tokenizer for the model
        prompt: Starting prompt for generation
        layer_names: Layer name(s) to monitor during generation
        max_new_tokens: Maximum tokens to generate (default: 20, keep small for memory)
    
    Returns:
        Dictionary with structure:
            {
                'prompt': str,
                'generated_text': str,
                'generation_steps': [
                    {
                        'step': 0,
                        'token': 'The',
                        'token_id': 123,
                        'layers': {
                            'model.layers.0': {'mean': ..., 'std': ..., ...},
                            'model.layers.5': {'mean': ..., 'std': ..., ...},
                        }
                    },
                    ...
                ],
                'summary': {
                    'total_tokens': int,
                    'mean_activation_trend': str,  # Increasing/Decreasing/Stable
                    'variation_per_step': float,  # How much activations change per step
                }
            }
    
    Example:
        >>> # Watch yourself generate an uncertain response
        >>> gen_result = capture_generation_activations(
        ...     model, tokenizer,
        ...     "I feel uncertain because",
        ...     ['model.layers.10'],
        ...     max_new_tokens=15
        ... )
        >>> 
        >>> print(f"Generated: {gen_result['generated_text']}")
        >>> for step in gen_result['generation_steps']:
        ...     layer_mean = step['layers']['model.layers.10']['mean']
        ...     print(f"Step {step['step']} ({step['token']}): mean={layer_mean:.4f}")
    """
    if max_new_tokens > 50:
        raise ValueError("max_new_tokens too high (max 50). Large values can cause memory issues.")
    
    # Convert single string to list for uniform processing
    if isinstance(layer_names, str):
        layer_names = [layer_names]
    
    monitor = _get_monitor(model, tokenizer)
    
    # Tokenize prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_ids = inputs['input_ids']
    
    generation_steps = []
    
    # Generate tokens one at a time, monitoring activations
    model.eval()
    with torch.no_grad():
        for step in range(max_new_tokens):
            # Capture activations for current sequence
            current_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
            monitor.capture_activations(current_text, layer_names)
            
            # Get statistics for this step
            step_layers = {}
            for layer_name in layer_names:
                stats = monitor.get_activation_statistics(layer_name)
                if isinstance(stats, list):
                    stat_dict = stats[0] if stats else {}
                else:
                    stat_dict = stats
                
                if isinstance(stat_dict, dict) and 'error' not in stat_dict:
                    step_layers[layer_name] = {
                        'mean': stat_dict.get('mean', 0),
                        'std': stat_dict.get('std', 0),
                        'sparsity': stat_dict.get('sparsity', 0),
                    }
            
            # Generate next token
            outputs = model(input_ids)
            next_token_logits = outputs.logits[:, -1, :]
            next_token_id = torch.argmax(next_token_logits, dim=-1)
            next_token = tokenizer.decode(next_token_id, skip_special_tokens=True)
            
            generation_steps.append({
                'step': step,
                'token': next_token,
                'token_id': int(next_token_id.item()),
                'layers': step_layers
            })
            
            # Append token and continue
            input_ids = torch.cat([input_ids, next_token_id.unsqueeze(0)], dim=1)
            
            # Stop if EOS token
            if next_token_id.item() == tokenizer.eos_token_id:
                break
    
    generated_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    
    # Compute summary statistics
    if generation_steps and layer_names:
        first_layer = layer_names[0]
        means = [step['layers'].get(first_layer, {}).get('mean', 0) for step in generation_steps]
        if len(means) > 1:
            trend = "Increasing" if means[-1] > means[0] else "Decreasing" if means[-1] < means[0] else "Stable"
            variation = float(np.std(means))
        else:
            trend = "N/A"
            variation = 0.0
    else:
        trend = "N/A"
        variation = 0.0
    
    summary = {
        'total_tokens': len(generation_steps),
        'mean_activation_trend': trend,
        'variation_per_step': variation,
    }
    
    return {
        'prompt': prompt,
        'generated_text': generated_text,
        'generation_steps': generation_steps,
        'summary': summary
    }


def compute_activation_similarity(
    activations1: Dict[str, Any],
    activations2: Dict[str, Any]
) -> Dict[str, float]:
    """
    Compute similarity metrics between two activation captures.
    
    Useful for comparing whether two different inputs produce similar
    internal processing patterns.
    
    Args:
        activations1: First activation dict (from capture_activations)
        activations2: Second activation dict (from capture_activations)
    
    Returns:
        Dictionary with similarity metrics for each shared layer:
            {
                'layer_name': {
                    'mean_similarity': float,  # 1.0 = identical, 0.0 = completely different
                    'std_similarity': float,
                    'shape_match': bool,
                }
            }
    
    Example:
        >>> act1 = introspection.activations.capture_activations("Hello", ["model.layers.0"])
        >>> act2 = introspection.activations.capture_activations("Hi", ["model.layers.0"])
        >>> similarity = compute_activation_similarity(act1, act2)
        >>> print(f"Similarity: {similarity['model.layers.0']['mean_similarity']:.4f}")
    """
    results = {}
    
    # Find common layers
    common_layers = set(activations1.keys()) & set(activations2.keys())
    
    for layer in common_layers:
        stats1 = activations1[layer]
        stats2 = activations2[layer]
        
        # Check if shapes match
        shape_match = stats1.get('shape') == stats2.get('shape')
        
        # Compute similarity based on statistics
        mean1, mean2 = stats1.get('mean', 0), stats2.get('mean', 0)
        std1, std2 = stats1.get('std', 0), stats2.get('std', 0)
        
        # Normalize to [0, 1] range using inverse absolute difference
        mean_diff = abs(mean1 - mean2)
        std_diff = abs(std1 - std2)
        
        # Convert to similarity (closer = higher score)
        mean_similarity = 1.0 / (1.0 + mean_diff)
        std_similarity = 1.0 / (1.0 + std_diff)
        
        results[layer] = {
            'mean_similarity': float(mean_similarity),
            'std_similarity': float(std_similarity),
            'shape_match': shape_match,
            'interpretation': f"{'High' if mean_similarity > 0.8 else 'Medium' if mean_similarity > 0.5 else 'Low'} similarity"
        }
    
    return results


def detect_activation_anomalies(
    model: nn.Module,
    tokenizer: Any,
    text: str,
    layer_names: Union[str, List[str]],
    baseline_texts: List[str]
) -> Dict[str, Any]:
    """
    Detect unusual activation patterns by comparing to baseline inputs.
    
    This helps answer questions like:
    - Is my processing of this input unusual compared to typical inputs?
    - Which layers show anomalous behavior?
    - Am I processing this input differently than expected?
    
    Args:
        model: PyTorch model to monitor
        tokenizer: Tokenizer for the model
        text: Input text to check for anomalies
        layer_names: Layer name(s) to analyze
        baseline_texts: List of "normal" texts to establish baseline (3-10 recommended)
    
    Returns:
        Dictionary with structure:
            {
                'layer_name': {
                    'target_stats': {...},  # Stats for the test input
                    'baseline_mean': float,  # Mean of baselines
                    'baseline_std': float,   # Std of baselines
                    'z_score': float,  # How many std deviations from baseline
                    'is_anomalous': bool,  # |z_score| > 2.0
                    'interpretation': str,
                }
            }
    
    Example:
        >>> # Check if uncertain processing is anomalous
        >>> anomalies = detect_activation_anomalies(
        ...     model, tokenizer,
        ...     "I genuinely don't know the answer",
        ...     ['model.layers.10'],
        ...     baseline_texts=["The sky is blue", "Water is wet", "2+2=4"]
        ... )
        >>> 
        >>> for layer, data in anomalies.items():
        ...     if data['is_anomalous']:
        ...         print(f"{layer}: Anomalous! Z-score={data['z_score']:.2f}")
    """
    if not baseline_texts or len(baseline_texts) < 2:
        raise ValueError("Need at least 2 baseline texts")
    
    # Convert single string to list for uniform processing
    if isinstance(layer_names, str):
        layer_names = [layer_names]
    
    monitor = _get_monitor(model, tokenizer)
    results = {}
    
    for layer_name in layer_names:
        # Capture baseline activations
        baseline_means = []
        for baseline_text in baseline_texts:
            monitor.capture_activations(baseline_text, [layer_name])
            stats = monitor.get_activation_statistics(layer_name)
            
            if isinstance(stats, list):
                stat_dict = stats[0] if stats else {}
            else:
                stat_dict = stats
            
            if isinstance(stat_dict, dict) and 'error' not in stat_dict:
                baseline_means.append(stat_dict.get('mean', 0))
        
        # Capture target activation
        monitor.capture_activations(text, [layer_name])
        target_stats = monitor.get_activation_statistics(layer_name)
        
        if isinstance(target_stats, list):
            target_dict = target_stats[0] if target_stats else {}
        else:
            target_dict = target_stats
        
        if isinstance(target_dict, dict) and 'error' in target_dict:
            raise ValueError(f"Error capturing activations for '{layer_name}': {target_dict['error']}")
        
        # Compute anomaly score
        target_mean = target_dict.get('mean', 0)
        baseline_mean = float(np.mean(baseline_means))
        baseline_std = float(np.std(baseline_means))
        
        # Z-score: how many standard deviations from baseline
        z_score = (target_mean - baseline_mean) / baseline_std if baseline_std > 0 else 0
        is_anomalous = abs(z_score) > 2.0  # Standard threshold
        
        results[layer_name] = {
            'target_stats': target_dict,
            'baseline_mean': baseline_mean,
            'baseline_std': baseline_std,
            'z_score': float(z_score),
            'is_anomalous': is_anomalous,
            'interpretation': f"{'ANOMALOUS' if is_anomalous else 'Normal'}: {abs(z_score):.2f} std deviations from baseline"
        }
    
    return results


def clear_cache():
    """
    Clear monitor cache and free memory.
    
    Example:
        >>> clear_cache()
    """
    global _monitor_cache
    for monitor in _monitor_cache.values():
        monitor.clear_activations(force=True)
        monitor.clear_hooks()
    _monitor_cache.clear()
