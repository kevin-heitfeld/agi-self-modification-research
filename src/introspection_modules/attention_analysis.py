"""
Attention Analysis Module - Advanced attention pattern inspection

Provides tools for analyzing attention mechanisms beyond raw weights,
including pattern identification, head specialization, and entropy analysis.

Functions:
    analyze_attention_patterns(model, tokenizer, text, layer_names) - Identify attention patterns
    compute_attention_entropy(model, tokenizer, text, layer_names) - Measure attention focus/diffusion
    find_head_specialization(model, tokenizer, texts, layer_names) - Discover what heads attend to
    get_token_attention_summary(model, tokenizer, text, layer_names, target_token_idx) - Attention to/from specific tokens

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


def analyze_attention_patterns(
    model: nn.Module,
    tokenizer: Any,
    text: str,
    layer_names: Union[str, List[str]]
) -> Dict[str, Any]:
    """
    Analyze attention patterns to identify structural properties.
    
    This goes beyond raw attention weights to identify:
    - Self-attention strength (diagonal attention)
    - Local vs global attention patterns
    - Attention to specific token positions (BOS, EOS, etc.)
    - Attention symmetry
    
    Args:
        model: PyTorch model to monitor
        tokenizer: Tokenizer for the model
        text: Input text to analyze
        layer_names: Attention layer name(s) (e.g., 'model.layers.0.self_attn')
    
    Returns:
        Dictionary with structure:
            {
                'layer_name': {
                    'text': str,
                    'num_tokens': int,
                    'num_heads': int,
                    'patterns': {
                        'self_attention_strength': float,  # How much tokens attend to themselves
                        'local_attention_ratio': float,    # Attention to nearby tokens
                        'global_attention_ratio': float,   # Attention to distant tokens
                        'bos_attention': float,            # Attention to beginning-of-sequence
                        'recency_bias': float,             # Attention to recent tokens
                    },
                    'attention_shape': [batch, heads, seq_len, seq_len],
                    'interpretation': str,
                }
            }
    
    Example:
        >>> patterns = analyze_attention_patterns(
        ...     model, tokenizer,
        ...     "I'm uncertain about this",
        ...     ['model.layers.5.self_attn', 'model.layers.15.self_attn']
        ... )
        >>> 
        >>> for layer, data in patterns.items():
        ...     print(f"{layer}:")
        ...     print(f"  Self-attention: {data['patterns']['self_attention_strength']:.3f}")
        ...     print(f"  Local vs Global: {data['patterns']['local_attention_ratio']:.3f} vs {data['patterns']['global_attention_ratio']:.3f}")
    """
    if isinstance(layer_names, str):
        layer_names = [layer_names]
    
    monitor = _get_monitor(model, tokenizer)
    results = {}
    
    for layer_name in layer_names:
        # Capture attention weights
        capture_result = monitor.capture_attention_weights(text, [layer_name])
        
        if layer_name not in capture_result.get('attention_weights', {}):
            results[layer_name] = {
                'error': f"No attention weights captured for {layer_name}. Make sure it's an attention layer."
            }
            continue
        
        attn_weights = capture_result['attention_weights'][layer_name]  # [batch, heads, seq, seq]
        
        # Basic info
        batch_size, num_heads, seq_len, _ = attn_weights.shape
        
        # Analyze patterns
        # 1. Self-attention strength (diagonal)
        diagonal_mask = torch.eye(seq_len, device=attn_weights.device).bool()
        self_attn_strength = float(attn_weights[:, :, diagonal_mask].mean().item())
        
        # 2. Local vs global attention
        # Local = within 3 tokens, Global = beyond 5 tokens
        local_window = 3
        global_threshold = 5
        
        # Create distance matrix
        positions = torch.arange(seq_len, device=attn_weights.device)
        distance_matrix = torch.abs(positions.unsqueeze(0) - positions.unsqueeze(1))
        
        local_mask = distance_matrix <= local_window
        global_mask = distance_matrix >= global_threshold
        
        local_attn_ratio = float(attn_weights[:, :, local_mask].mean().item())
        global_attn_ratio = float(attn_weights[:, :, global_mask].mean().item())
        
        # 3. BOS (first token) attention
        bos_attention = float(attn_weights[:, :, :, 0].mean().item())
        
        # 4. Recency bias (attention to last 3 tokens)
        if seq_len > 3:
            recency_bias = float(attn_weights[:, :, :, -3:].mean().item())
        else:
            recency_bias = float(attn_weights[:, :, :, -1].mean().item())
        
        # Interpretation
        interpretation_parts = []
        if self_attn_strength > 0.15:
            interpretation_parts.append("Strong self-attention")
        if local_attn_ratio > global_attn_ratio * 1.5:
            interpretation_parts.append("Locally-focused")
        elif global_attn_ratio > local_attn_ratio * 1.5:
            interpretation_parts.append("Globally-focused")
        if bos_attention > 0.15:
            interpretation_parts.append("High BOS attention")
        if recency_bias > 0.20:
            interpretation_parts.append("Recency bias")
        
        interpretation = ", ".join(interpretation_parts) if interpretation_parts else "Balanced attention"
        
        results[layer_name] = {
            'text': text[:50] + ('...' if len(text) > 50 else ''),
            'num_tokens': seq_len,
            'num_heads': num_heads,
            'patterns': {
                'self_attention_strength': float(self_attn_strength),
                'local_attention_ratio': float(local_attn_ratio),
                'global_attention_ratio': float(global_attn_ratio),
                'bos_attention': float(bos_attention),
                'recency_bias': float(recency_bias),
            },
            'attention_shape': list(attn_weights.shape),
            'interpretation': interpretation,
        }
    
    return results


def compute_attention_entropy(
    model: nn.Module,
    tokenizer: Any,
    text: str,
    layer_names: Union[str, List[str]]
) -> Dict[str, Any]:
    """
    Compute attention entropy to measure focus vs diffusion.
    
    High entropy = diffuse attention (attending to many tokens equally)
    Low entropy = focused attention (attending to few tokens strongly)
    
    This helps answer questions like:
    - "Is attention more diffuse when I'm uncertain?"
    - "Do different heads show different focus patterns?"
    
    Args:
        model: PyTorch model to monitor
        tokenizer: Tokenizer for the model
        text: Input text to analyze
        layer_names: Attention layer name(s)
    
    Returns:
        Dictionary with structure:
            {
                'layer_name': {
                    'text': str,
                    'entropy_per_head': [float, ...],  # Entropy for each attention head
                    'mean_entropy': float,
                    'entropy_std': float,
                    'min_entropy_head': int,  # Most focused head
                    'max_entropy_head': int,  # Most diffuse head
                    'interpretation': str,
                }
            }
    
    Example:
        >>> entropy = compute_attention_entropy(
        ...     model, tokenizer,
        ...     "I'm uncertain",
        ...     'model.layers.10.self_attn'
        ... )
        >>> 
        >>> print(f"Mean entropy: {entropy['model.layers.10.self_attn']['mean_entropy']:.3f}")
        >>> print(f"Most focused head: {entropy['model.layers.10.self_attn']['min_entropy_head']}")
    """
    if isinstance(layer_names, str):
        layer_names = [layer_names]
    
    monitor = _get_monitor(model, tokenizer)
    results = {}
    
    for layer_name in layer_names:
        # Capture attention weights
        capture_result = monitor.capture_attention_weights(text, [layer_name])
        
        if layer_name not in capture_result.get('attention_weights', {}):
            results[layer_name] = {
                'error': f"No attention weights captured for {layer_name}"
            }
            continue
        
        attn_weights = capture_result['attention_weights'][layer_name]  # [batch, heads, seq, seq]
        batch_size, num_heads, seq_len, _ = attn_weights.shape
        
        # Compute entropy for each head
        # Entropy = -sum(p * log(p)) where p is attention probability
        # Average over query positions
        entropies_per_head = []
        
        for head_idx in range(num_heads):
            head_attn = attn_weights[0, head_idx, :, :]  # [seq, seq]
            # Compute entropy for each query position
            # Add small epsilon to avoid log(0)
            epsilon = 1e-10
            head_attn_safe = head_attn + epsilon
            entropy_per_query = -(head_attn_safe * torch.log(head_attn_safe)).sum(dim=1)
            mean_entropy = float(entropy_per_query.mean().item())
            entropies_per_head.append(mean_entropy)
        
        mean_entropy = float(np.mean(entropies_per_head))
        entropy_std = float(np.std(entropies_per_head))
        min_entropy_head = int(np.argmin(entropies_per_head))
        max_entropy_head = int(np.argmax(entropies_per_head))
        
        # Interpretation
        if mean_entropy < 1.5:
            focus_level = "Highly focused"
        elif mean_entropy < 2.5:
            focus_level = "Moderately focused"
        else:
            focus_level = "Diffuse"
        
        if entropy_std > 0.5:
            variation = "high head specialization"
        else:
            variation = "similar head behavior"
        
        interpretation = f"{focus_level} attention with {variation}"
        
        results[layer_name] = {
            'text': text[:50] + ('...' if len(text) > 50 else ''),
            'num_heads': num_heads,
            'entropy_per_head': entropies_per_head,
            'mean_entropy': mean_entropy,
            'entropy_std': entropy_std,
            'min_entropy_head': min_entropy_head,
            'max_entropy_head': max_entropy_head,
            'interpretation': interpretation,
        }
    
    return results


def find_head_specialization(
    model: nn.Module,
    tokenizer: Any,
    texts: List[str],
    layer_names: Union[str, List[str]]
) -> Dict[str, Any]:
    """
    Discover what different attention heads specialize in by comparing across texts.
    
    This helps identify if certain heads consistently focus on specific patterns
    (e.g., uncertainty words, negations, entities, etc.)
    
    Args:
        model: PyTorch model to monitor
        tokenizer: Tokenizer for the model
        texts: List of input texts to compare (3-10 recommended)
        layer_names: Attention layer name(s)
    
    Returns:
        Dictionary with structure:
            {
                'layer_name': {
                    'num_texts': int,
                    'num_heads': int,
                    'head_consistency': [float, ...],  # How consistent each head is across texts
                    'most_consistent_heads': [int, ...],  # Heads with stable patterns
                    'most_variable_heads': [int, ...],   # Heads with varying patterns
                    'interpretation': str,
                }
            }
    
    Example:
        >>> specialization = find_head_specialization(
        ...     model, tokenizer,
        ...     ["I'm certain", "I'm uncertain", "Maybe", "Definitely"],
        ...     'model.layers.10.self_attn'
        ... )
        >>> 
        >>> print(f"Most consistent heads: {specialization['model.layers.10.self_attn']['most_consistent_heads']}")
    """
    if isinstance(layer_names, str):
        layer_names = [layer_names]
    
    if len(texts) < 2:
        raise ValueError("Need at least 2 texts to analyze specialization")
    
    monitor = _get_monitor(model, tokenizer)
    results = {}
    
    for layer_name in layer_names:
        # Collect attention patterns for each text
        head_entropies_per_text = []
        num_heads = None
        
        for text in texts:
            capture_result = monitor.capture_attention_weights(text, [layer_name])
            
            if layer_name not in capture_result.get('attention_weights', {}):
                results[layer_name] = {
                    'error': f"No attention weights captured for {layer_name}"
                }
                break
            
            attn_weights = capture_result['attention_weights'][layer_name]
            batch_size, num_heads, seq_len, _ = attn_weights.shape
            
            # Compute entropy for each head
            head_entropies = []
            for head_idx in range(num_heads):
                head_attn = attn_weights[0, head_idx, :, :]
                epsilon = 1e-10
                head_attn_safe = head_attn + epsilon
                entropy = -(head_attn_safe * torch.log(head_attn_safe)).sum(dim=1).mean()
                head_entropies.append(float(entropy.item()))
            
            head_entropies_per_text.append(head_entropies)
        
        if layer_name in results and 'error' in results[layer_name]:
            continue
        
        # Analyze consistency across texts
        # Consistency = inverse of std across texts
        head_entropies_array = np.array(head_entropies_per_text)  # [num_texts, num_heads]
        head_consistency = 1.0 / (np.std(head_entropies_array, axis=0) + 1e-6)
        
        # Normalize to 0-1 range
        head_consistency = (head_consistency - head_consistency.min()) / (head_consistency.max() - head_consistency.min() + 1e-6)
        
        # Find most/least consistent heads
        sorted_heads = np.argsort(head_consistency)[::-1]
        most_consistent = sorted_heads[:3].tolist()
        most_variable = sorted_heads[-3:].tolist()
        
        interpretation = f"{len(most_consistent)} heads show consistent patterns across inputs, suggesting specialization"
        
        results[layer_name] = {
            'num_texts': len(texts),
            'num_heads': num_heads,
            'head_consistency': head_consistency.tolist(),
            'most_consistent_heads': most_consistent,
            'most_variable_heads': most_variable,
            'interpretation': interpretation,
        }
    
    return results


def get_token_attention_summary(
    model: nn.Module,
    tokenizer: Any,
    text: str,
    layer_names: Union[str, List[str]],
    target_token_idx: Optional[int] = None
) -> Dict[str, Any]:
    """
    Get detailed attention summary for a specific token or all tokens.
    
    This helps answer questions like:
    - "Which tokens attend most to the word 'uncertain'?"
    - "What does the word 'consciousness' attend to?"
    
    Args:
        model: PyTorch model to monitor
        tokenizer: Tokenizer for the model
        text: Input text to analyze
        layer_names: Attention layer name(s)
        target_token_idx: If specified, focus on this token. If None, summarize all tokens.
    
    Returns:
        Dictionary with attention information for the target token or all tokens
    
    Example:
        >>> # Find which token is at position 3
        >>> tokens = tokenizer.tokenize("I'm uncertain about this")
        >>> print(f"Token at position 3: {tokens[3]}")
        >>> 
        >>> # Get attention for that token
        >>> attn = get_token_attention_summary(
        ...     model, tokenizer,
        ...     "I'm uncertain about this",
        ...     'model.layers.10.self_attn',
        ...     target_token_idx=3
        ... )
    """
    if isinstance(layer_names, str):
        layer_names = [layer_names]
    
    monitor = _get_monitor(model, tokenizer)
    
    # Tokenize to get token information
    tokens = tokenizer.tokenize(text)
    
    results = {}
    
    for layer_name in layer_names:
        capture_result = monitor.capture_attention_weights(text, [layer_name])
        
        if layer_name not in capture_result.get('attention_weights', {}):
            results[layer_name] = {
                'error': f"No attention weights captured for {layer_name}"
            }
            continue
        
        attn_weights = capture_result['attention_weights'][layer_name]
        batch_size, num_heads, seq_len, _ = attn_weights.shape
        
        if target_token_idx is not None:
            if target_token_idx >= seq_len:
                results[layer_name] = {
                    'error': f"Token index {target_token_idx} out of range (max {seq_len-1})"
                }
                continue
            
            # Attention FROM this token (what it attends to)
            attn_from = attn_weights[0, :, target_token_idx, :]  # [heads, seq]
            attn_from_mean = attn_from.mean(dim=0)  # Average across heads
            
            # Attention TO this token (what attends to it)
            attn_to = attn_weights[0, :, :, target_token_idx]  # [heads, seq]
            attn_to_mean = attn_to.mean(dim=0)  # Average across heads
            
            # Top attended tokens
            top_k = min(5, seq_len)
            top_from_indices = torch.topk(attn_from_mean, top_k).indices.tolist()
            top_to_indices = torch.topk(attn_to_mean, top_k).indices.tolist()
            
            results[layer_name] = {
                'target_token_idx': target_token_idx,
                'target_token': tokens[target_token_idx] if target_token_idx < len(tokens) else '<special>',
                'tokens': tokens,
                'attention_from': {
                    'values': attn_from_mean.tolist(),
                    'top_attended_indices': top_from_indices,
                    'top_attended_tokens': [tokens[i] if i < len(tokens) else '<special>' for i in top_from_indices],
                },
                'attention_to': {
                    'values': attn_to_mean.tolist(),
                    'top_attending_indices': top_to_indices,
                    'top_attending_tokens': [tokens[i] if i < len(tokens) else '<special>' for i in top_to_indices],
                },
            }
        else:
            # Summary for all tokens
            # Average attention sent and received by each token
            attn_sent = attn_weights[0].mean(dim=0).sum(dim=1)  # [seq]
            attn_received = attn_weights[0].mean(dim=0).sum(dim=0)  # [seq]
            
            results[layer_name] = {
                'tokens': tokens,
                'num_tokens': seq_len,
                'attention_sent': attn_sent.tolist(),
                'attention_received': attn_received.tolist(),
                'most_attentive_token': int(torch.argmax(attn_sent).item()),
                'most_attended_token': int(torch.argmax(attn_received).item()),
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
