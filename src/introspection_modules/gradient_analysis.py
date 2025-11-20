"""
Gradient Analysis Module - Sensitivity and attribution analysis

Provides tools for understanding what causes activations and how inputs
influence processing through gradient-based analysis.

Functions:
    compute_input_sensitivity(model, tokenizer, text, layer_names) - Which tokens influence activations
    compare_inputs_gradient(model, tokenizer, original_text, modified_text, layer_names) - Counterfactual analysis
    find_influential_tokens(model, tokenizer, text, layer_names) - Most influential input tokens
    compute_layer_gradients(model, tokenizer, text, target_layer, source_layer) - Gradient flow between layers

Author: AGI Self-Modification Research Team
Date: November 20, 2025
"""

from typing import Dict, List, Any, Optional, Union, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ..introspection.activation_monitor import ActivationMonitor


def compute_input_sensitivity(
    model: nn.Module,
    tokenizer: Any,
    text: str,
    layer_names: Union[str, List[str]]
) -> Dict[str, Any]:
    """
    Compute how sensitive layer activations are to each input token.
    
    Uses gradient-based attribution to identify which input tokens most
    influence the activations in specified layers.
    
    This helps answer questions like:
    - "Which words in 'I'm uncertain' cause uncertainty activations?"
    - "What parts of input drive processing in layer X?"
    
    Args:
        model: PyTorch model to analyze
        tokenizer: Tokenizer for the model
        text: Input text to analyze
        layer_names: Layer name(s) to analyze sensitivity for
    
    Returns:
        Dictionary with structure:
            {
                'layer_name': {
                    'text': str,
                    'tokens': [str, ...],
                    'token_sensitivity': [float, ...],  # Sensitivity score per token
                    'most_influential_tokens': [(idx, token, score), ...],
                    'least_influential_tokens': [(idx, token, score), ...],
                    'interpretation': str,
                }
            }
    
    Example:
        >>> sensitivity = compute_input_sensitivity(
        ...     model, tokenizer,
        ...     "I'm genuinely uncertain about consciousness",
        ...     ['model.layers.10', 'model.layers.20']
        ... )
        >>> 
        >>> for layer, data in sensitivity.items():
        ...     print(f"{layer}:")
        ...     for idx, token, score in data['most_influential_tokens'][:3]:
        ...         print(f"  {token}: {score:.4f}")
    """
    if isinstance(layer_names, str):
        layer_names = [layer_names]
    
    # Tokenize input
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    input_ids = inputs['input_ids']
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    
    model.eval()
    results = {}
    
    # Get embeddings (need gradient)
    embeddings = model.get_input_embeddings()
    input_embeds = embeddings(input_ids)
    input_embeds.requires_grad_(True)
    
    for layer_name in layer_names:
        # Forward pass with hooks to capture target layer
        target_activation = None
        
        def hook_fn(module, input, output):
            nonlocal target_activation
            if isinstance(output, tuple):
                target_activation = output[0]
            else:
                target_activation = output
        
        # Register hook
        target_module = dict(model.named_modules()).get(layer_name)
        if target_module is None:
            results[layer_name] = {'error': f"Layer {layer_name} not found"}
            continue
        
        hook = target_module.register_forward_hook(hook_fn)
        
        try:
            # Forward pass
            outputs = model(inputs_embeds=input_embeds, attention_mask=inputs.get('attention_mask'))
            
            if target_activation is None:
                results[layer_name] = {'error': f"No activation captured for {layer_name}"}
                continue
            
            # Compute gradient of mean activation w.r.t. input embeddings
            # We use mean as a scalar objective to backprop through
            mean_activation = target_activation.mean()
            mean_activation.backward(retain_graph=True)
            
            # Get gradients
            if input_embeds.grad is not None:
                # Compute L2 norm of gradient for each token (sensitivity)
                token_gradients = input_embeds.grad[0]  # [seq_len, hidden_dim]
                token_sensitivity = torch.norm(token_gradients, dim=1)  # [seq_len]
                token_sensitivity = token_sensitivity.detach().cpu().numpy()
                
                # Normalize to 0-1 range for interpretability
                if token_sensitivity.max() > 0:
                    token_sensitivity = token_sensitivity / token_sensitivity.max()
                
                # Find most/least influential tokens
                sorted_indices = np.argsort(token_sensitivity)[::-1]
                most_influential = [(int(i), tokens[i], float(token_sensitivity[i])) 
                                   for i in sorted_indices[:5]]
                least_influential = [(int(i), tokens[i], float(token_sensitivity[i])) 
                                    for i in sorted_indices[-5:]]
                
                interpretation = f"Tokens '{most_influential[0][1]}' and '{most_influential[1][1] if len(most_influential) > 1 else 'N/A'}' most influential"
                
                results[layer_name] = {
                    'text': text,
                    'tokens': tokens,
                    'token_sensitivity': token_sensitivity.tolist(),
                    'most_influential_tokens': most_influential,
                    'least_influential_tokens': least_influential,
                    'interpretation': interpretation,
                }
            else:
                results[layer_name] = {'error': 'No gradients computed'}
            
            # Clear gradients for next iteration
            if input_embeds.grad is not None:
                input_embeds.grad.zero_()
        
        finally:
            hook.remove()
    
    return results


def compare_inputs_gradient(
    model: nn.Module,
    tokenizer: Any,
    original_text: str,
    modified_text: str,
    layer_names: Union[str, List[str]]
) -> Dict[str, Any]:
    """
    Compare how two inputs affect layer activations (counterfactual analysis).
    
    This helps answer questions like:
    - "If I change 'uncertain' to 'certain', how does processing change?"
    - "What's the causal effect of specific word changes?"
    
    Args:
        model: PyTorch model to analyze
        tokenizer: Tokenizer for the model
        original_text: Original input text
        modified_text: Modified input text (counterfactual)
        layer_names: Layer name(s) to compare
    
    Returns:
        Dictionary with structure:
            {
                'layer_name': {
                    'original_text': str,
                    'modified_text': str,
                    'original_activation_mean': float,
                    'modified_activation_mean': float,
                    'activation_difference': float,
                    'activation_percent_change': float,
                    'interpretation': str,
                }
            }
    
    Example:
        >>> comparison = compare_inputs_gradient(
        ...     model, tokenizer,
        ...     "I'm uncertain about this",
        ...     "I'm certain about this",
        ...     ['model.layers.10', 'model.layers.20']
        ... )
        >>> 
        >>> for layer, data in comparison.items():
        ...     print(f"{layer}: {data['activation_percent_change']:.1f}% change")
    """
    if isinstance(layer_names, str):
        layer_names = [layer_names]
    
    model.eval()
    results = {}
    
    with torch.no_grad():
        # Process original text
        inputs_orig = tokenizer(original_text, return_tensors="pt").to(model.device)
        
        # Process modified text
        inputs_mod = tokenizer(modified_text, return_tensors="pt").to(model.device)
        
        for layer_name in layer_names:
            # Capture activations for both inputs
            activations_orig = None
            activations_mod = None
            
            def make_hook(storage_list, idx):
                def hook_fn(module, input, output):
                    if isinstance(output, tuple):
                        storage_list[idx] = output[0].detach()
                    else:
                        storage_list[idx] = output.detach()
                return hook_fn
            
            target_module = dict(model.named_modules()).get(layer_name)
            if target_module is None:
                results[layer_name] = {'error': f"Layer {layer_name} not found"}
                continue
            
            # Original
            storage = [None]
            hook = target_module.register_forward_hook(make_hook(storage, 0))
            model(**inputs_orig)
            activations_orig = storage[0]
            hook.remove()
            
            # Modified
            storage = [None]
            hook = target_module.register_forward_hook(make_hook(storage, 0))
            model(**inputs_mod)
            activations_mod = storage[0]
            hook.remove()
            
            if activations_orig is None or activations_mod is None:
                results[layer_name] = {'error': f"Failed to capture activations for {layer_name}"}
                continue
            
            # Compute differences
            orig_mean = float(activations_orig.mean().item())
            mod_mean = float(activations_mod.mean().item())
            difference = mod_mean - orig_mean
            percent_change = (difference / abs(orig_mean)) * 100 if orig_mean != 0 else 0
            
            # Interpretation
            if abs(percent_change) < 5:
                change_level = "Minimal"
            elif abs(percent_change) < 15:
                change_level = "Moderate"
            else:
                change_level = "Significant"
            
            direction = "increase" if difference > 0 else "decrease"
            interpretation = f"{change_level} {direction} in activation ({abs(percent_change):.1f}%)"
            
            results[layer_name] = {
                'original_text': original_text,
                'modified_text': modified_text,
                'original_activation_mean': orig_mean,
                'modified_activation_mean': mod_mean,
                'activation_difference': float(difference),
                'activation_percent_change': float(percent_change),
                'interpretation': interpretation,
            }
    
    return results


def find_influential_tokens(
    model: nn.Module,
    tokenizer: Any,
    text: str,
    layer_names: Union[str, List[str]],
    top_k: int = 5
) -> Dict[str, Any]:
    """
    Find the most influential tokens for layer activations using integrated gradients.
    
    More sophisticated than simple gradient - uses integrated gradients for
    better attribution.
    
    Args:
        model: PyTorch model to analyze
        tokenizer: Tokenizer for the model
        text: Input text to analyze
        layer_names: Layer name(s) to analyze
        top_k: Number of top tokens to return
    
    Returns:
        Dictionary mapping layer names to influential token information
    
    Example:
        >>> influential = find_influential_tokens(
        ...     model, tokenizer,
        ...     "I feel uncertain",
        ...     'model.layers.15',
        ...     top_k=3
        ... )
    """
    if isinstance(layer_names, str):
        layer_names = [layer_names]
    
    # Tokenize input
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    input_ids = inputs['input_ids']
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    
    model.eval()
    results = {}
    
    # Get embeddings
    embeddings_module = model.get_input_embeddings()
    input_embeds = embeddings_module(input_ids)  # [batch, seq, hidden]
    
    # Baseline: zero embeddings
    baseline_embeds = torch.zeros_like(input_embeds)
    
    # Integrated gradients: interpolate between baseline and input
    num_steps = 20
    integrated_grads = torch.zeros_like(input_embeds)
    
    for layer_name in layer_names:
        target_module = dict(model.named_modules()).get(layer_name)
        if target_module is None:
            results[layer_name] = {'error': f"Layer {layer_name} not found"}
            continue
        
        layer_integrated_grads = torch.zeros_like(input_embeds)
        
        for step in range(num_steps):
            # Interpolate
            alpha = (step + 1) / num_steps
            interpolated_embeds = baseline_embeds + alpha * (input_embeds - baseline_embeds)
            interpolated_embeds.requires_grad_(True)
            
            # Capture activation
            target_activation = None
            
            def hook_fn(module, input, output):
                nonlocal target_activation
                if isinstance(output, tuple):
                    target_activation = output[0]
                else:
                    target_activation = output
            
            hook = target_module.register_forward_hook(hook_fn)
            
            try:
                # Forward pass
                outputs = model(inputs_embeds=interpolated_embeds, attention_mask=inputs.get('attention_mask'))
                
                if target_activation is not None:
                    # Backprop
                    mean_activation = target_activation.mean()
                    mean_activation.backward()
                    
                    if interpolated_embeds.grad is not None:
                        layer_integrated_grads += interpolated_embeds.grad / num_steps
            finally:
                hook.remove()
        
        # Compute attribution scores
        # Attribution = (input - baseline) * integrated_gradient
        attribution = (input_embeds - baseline_embeds) * layer_integrated_grads
        token_attribution = torch.norm(attribution[0], dim=1).detach().cpu().numpy()
        
        # Normalize
        if token_attribution.max() > 0:
            token_attribution = token_attribution / token_attribution.max()
        
        # Get top tokens
        top_indices = np.argsort(token_attribution)[::-1][:top_k]
        top_tokens = [(int(i), tokens[i], float(token_attribution[i])) for i in top_indices]
        
        results[layer_name] = {
            'text': text,
            'tokens': tokens,
            'token_attribution': token_attribution.tolist(),
            'top_influential_tokens': top_tokens,
            'method': 'integrated_gradients',
        }
    
    return results


def compute_layer_gradients(
    model: nn.Module,
    tokenizer: Any,
    text: str,
    target_layer: str,
    source_layer: str
) -> Dict[str, Any]:
    """
    Compute gradient flow between two layers.
    
    This helps understand how information from one layer influences another.
    
    Args:
        model: PyTorch model to analyze
        tokenizer: Tokenizer for the model
        text: Input text
        target_layer: Layer whose activation we measure
        source_layer: Layer whose influence we measure
    
    Returns:
        Dictionary with gradient flow information
    
    Example:
        >>> flow = compute_layer_gradients(
        ...     model, tokenizer,
        ...     "I'm uncertain",
        ...     target_layer='model.layers.15',
        ...     source_layer='model.layers.5'
        ... )
        >>> print(f"Gradient magnitude: {flow['gradient_magnitude']:.4f}")
    """
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    model.eval()
    
    # Get modules
    target_module = dict(model.named_modules()).get(target_layer)
    source_module = dict(model.named_modules()).get(source_layer)
    
    if target_module is None:
        return {'error': f"Target layer {target_layer} not found"}
    if source_module is None:
        return {'error': f"Source layer {source_layer} not found"}
    
    # Capture activations
    source_activation = None
    target_activation = None
    
    def make_hook(storage, key):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                storage[key] = output[0]
            else:
                storage[key] = output
        return hook_fn
    
    storage = {}
    hook1 = source_module.register_forward_hook(make_hook(storage, 'source'))
    hook2 = target_module.register_forward_hook(make_hook(storage, 'target'))
    
    try:
        # Forward pass
        outputs = model(**inputs)
        
        source_activation = storage.get('source')
        target_activation = storage.get('target')
        
        if source_activation is None or target_activation is None:
            return {'error': 'Failed to capture activations'}
        
        # Ensure source activation requires grad
        source_activation.requires_grad_(True)
        
        # Compute gradient of target w.r.t. source
        target_mean = target_activation.mean()
        target_mean.backward()
        
        if source_activation.grad is not None:
            gradient = source_activation.grad
            gradient_magnitude = float(torch.norm(gradient).item())
            gradient_mean = float(gradient.mean().item())
            gradient_std = float(gradient.std().item())
            
            return {
                'text': text,
                'source_layer': source_layer,
                'target_layer': target_layer,
                'gradient_magnitude': gradient_magnitude,
                'gradient_mean': gradient_mean,
                'gradient_std': gradient_std,
                'interpretation': f"Gradient magnitude {gradient_magnitude:.4f} indicates {'strong' if gradient_magnitude > 1.0 else 'moderate' if gradient_magnitude > 0.1 else 'weak'} influence",
            }
        else:
            return {'error': 'No gradient computed'}
    
    finally:
        hook1.remove()
        hook2.remove()
