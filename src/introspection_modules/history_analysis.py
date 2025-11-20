"""
History Analysis Module - Track activation changes over time/conversation

Provides tools for tracking how activations evolve during a conversation
or across multiple turns, enabling longitudinal analysis.

Functions:
    start_tracking(model, tokenizer, layer_names) - Begin tracking activation history
    record_turn(text) - Record activations for current turn
    get_activation_history(layer_names) - Get full history
    compare_to_previous(text, layer_names) - Compare current to previous turn
    analyze_drift(layer_names) - Detect systematic changes over time
    clear_history() - Clear tracking history

Author: AGI Self-Modification Research Team
Date: November 20, 2025
"""

from typing import Dict, List, Any, Optional, Union, Tuple
import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict
from ..introspection.activation_monitor import ActivationMonitor


# Global tracking state
_tracking_active = False
_tracked_model = None
_tracked_tokenizer = None
_tracked_layers = []
_activation_history = defaultdict(list)  # layer_name -> [{'turn': int, 'text': str, 'stats': dict}, ...]
_turn_counter = 0


def start_tracking(
    model: nn.Module,
    tokenizer: Any,
    layer_names: Union[str, List[str]]
) -> Dict[str, Any]:
    """
    Start tracking activation history for specified layers.
    
    This begins recording activations for each turn/input, allowing you
    to analyze how your processing changes over time.
    
    Args:
        model: PyTorch model to track
        tokenizer: Tokenizer for the model
        layer_names: Layer name(s) to track
    
    Returns:
        Dictionary with tracking status
    
    Example:
        >>> start_tracking(
        ...     model, tokenizer,
        ...     ['model.layers.10', 'model.layers.20']
        ... )
        >>> # Now use record_turn() after each response
    """
    global _tracking_active, _tracked_model, _tracked_tokenizer, _tracked_layers
    global _activation_history, _turn_counter
    
    if isinstance(layer_names, str):
        layer_names = [layer_names]
    
    _tracking_active = True
    _tracked_model = model
    _tracked_tokenizer = tokenizer
    _tracked_layers = layer_names
    _activation_history.clear()
    _turn_counter = 0
    
    return {
        'status': 'tracking_started',
        'layers_tracked': layer_names,
        'note': 'Use record_turn(text) to record each turn, then analyze with get_activation_history() or analyze_drift()'
    }


def record_turn(text: str) -> Dict[str, Any]:
    """
    Record activations for the current turn.
    
    Call this after each model response or at key points to track
    how activations change.
    
    Args:
        text: Input text to process and record
    
    Returns:
        Dictionary with recorded turn information
    
    Example:
        >>> # After model processes input
        >>> record_turn("I'm investigating consciousness")
        >>> # Later
        >>> record_turn("I'm examining uncertainty patterns")
        >>> # Then analyze
        >>> history = get_activation_history()
    """
    global _tracking_active, _tracked_model, _tracked_tokenizer, _tracked_layers
    global _activation_history, _turn_counter
    
    if not _tracking_active:
        return {'error': 'Tracking not started. Call start_tracking() first.'}
    
    if _tracked_model is None or _tracked_tokenizer is None:
        return {'error': 'Tracking state corrupted. Call start_tracking() again.'}
    
    # Create monitor
    monitor = ActivationMonitor(_tracked_model, _tracked_tokenizer)
    
    # Capture activations for all tracked layers
    monitor.capture_activations(text, _tracked_layers)
    
    # Record for each layer
    recorded_layers = []
    for layer_name in _tracked_layers:
        stats = monitor.get_activation_statistics(layer_name)
        
        if isinstance(stats, list):
            stat_dict = stats[0] if stats else {}
        else:
            stat_dict = stats
        
        if 'error' not in stat_dict:
            _activation_history[layer_name].append({
                'turn': _turn_counter,
                'text': text[:50] + ('...' if len(text) > 50 else ''),
                'stats': {
                    'mean': stat_dict.get('mean', 0),
                    'std': stat_dict.get('std', 0),
                    'max': stat_dict.get('max', 0),
                    'min': stat_dict.get('min', 0),
                    'sparsity': stat_dict.get('sparsity', 0),
                }
            })
            recorded_layers.append(layer_name)
    
    _turn_counter += 1
    
    return {
        'status': 'recorded',
        'turn': _turn_counter - 1,
        'layers_recorded': recorded_layers,
        'total_turns': _turn_counter,
    }


def get_activation_history(
    layer_names: Optional[Union[str, List[str]]] = None
) -> Dict[str, Any]:
    """
    Get the full activation history for tracked layers.
    
    Returns all recorded turns with their activation statistics.
    
    Args:
        layer_names: Specific layer(s) to retrieve. If None, returns all tracked layers.
    
    Returns:
        Dictionary with complete history for each layer
    
    Example:
        >>> history = get_activation_history(['model.layers.10'])
        >>> print(f"Recorded {len(history['model.layers.10']['turns'])} turns")
        >>> for turn in history['model.layers.10']['turns']:
        ...     print(f"Turn {turn['turn']}: mean={turn['stats']['mean']:.4f}")
    """
    global _tracking_active, _activation_history, _tracked_layers
    
    if not _tracking_active:
        return {'error': 'Tracking not started. Call start_tracking() first.'}
    
    if layer_names is None:
        layer_names = _tracked_layers
    elif isinstance(layer_names, str):
        layer_names = [layer_names]
    
    results = {}
    for layer_name in layer_names:
        if layer_name not in _activation_history:
            results[layer_name] = {
                'error': f'No history for {layer_name}. Is it being tracked?'
            }
        else:
            history = _activation_history[layer_name]
            results[layer_name] = {
                'total_turns': len(history),
                'turns': history,
                'summary': {
                    'mean_trajectory': [t['stats']['mean'] for t in history],
                    'std_trajectory': [t['stats']['std'] for t in history],
                }
            }
    
    return results


def compare_to_previous(
    text: str,
    layer_names: Optional[Union[str, List[str]]] = None
) -> Dict[str, Any]:
    """
    Compare current input's activations to the previous turn.
    
    This helps detect immediate changes in processing.
    
    Args:
        text: Current input text
        layer_names: Specific layer(s) to compare. If None, uses all tracked layers.
    
    Returns:
        Dictionary with comparison to previous turn
    
    Example:
        >>> comparison = compare_to_previous(
        ...     "Now I'm examining different patterns",
        ...     ['model.layers.10']
        ... )
        >>> print(f"Change from previous: {comparison['model.layers.10']['mean_change']:.4f}")
    """
    global _tracking_active, _tracked_model, _tracked_tokenizer, _tracked_layers
    global _activation_history
    
    if not _tracking_active:
        return {'error': 'Tracking not started. Call start_tracking() first.'}
    
    if layer_names is None:
        layer_names = _tracked_layers
    elif isinstance(layer_names, str):
        layer_names = [layer_names]
    
    # Get current activations
    monitor = ActivationMonitor(_tracked_model, _tracked_tokenizer)
    monitor.capture_activations(text, layer_names)
    
    results = {}
    for layer_name in layer_names:
        # Get current stats
        current_stats = monitor.get_activation_statistics(layer_name)
        if isinstance(current_stats, list):
            current_stats = current_stats[0] if current_stats else {}
        
        if 'error' in current_stats:
            results[layer_name] = {'error': current_stats['error']}
            continue
        
        # Get previous stats
        if layer_name not in _activation_history or len(_activation_history[layer_name]) == 0:
            results[layer_name] = {
                'error': 'No previous turn to compare to',
                'current_stats': current_stats
            }
            continue
        
        previous_turn = _activation_history[layer_name][-1]
        previous_stats = previous_turn['stats']
        
        # Compute changes
        mean_change = current_stats['mean'] - previous_stats['mean']
        mean_percent_change = (mean_change / abs(previous_stats['mean']) * 100) if previous_stats['mean'] != 0 else 0
        
        std_change = current_stats['std'] - previous_stats['std']
        sparsity_change = current_stats['sparsity'] - previous_stats['sparsity']
        
        # Overall magnitude of change
        change_magnitude = np.sqrt(mean_change**2 + std_change**2)
        
        # Interpretation
        if abs(mean_percent_change) < 5:
            change_level = "Minimal"
        elif abs(mean_percent_change) < 15:
            change_level = "Moderate"
        else:
            change_level = "Significant"
        
        direction = "increase" if mean_change > 0 else "decrease"
        
        results[layer_name] = {
            'current_text': text[:50] + ('...' if len(text) > 50 else ''),
            'previous_text': previous_turn['text'],
            'previous_turn': previous_turn['turn'],
            'current_stats': {
                'mean': current_stats['mean'],
                'std': current_stats['std'],
                'sparsity': current_stats['sparsity'],
            },
            'previous_stats': previous_stats,
            'changes': {
                'mean_change': float(mean_change),
                'mean_percent_change': float(mean_percent_change),
                'std_change': float(std_change),
                'sparsity_change': float(sparsity_change),
                'change_magnitude': float(change_magnitude),
            },
            'interpretation': f"{change_level} {direction} ({abs(mean_percent_change):.1f}%)",
        }
    
    return results


def analyze_drift(
    layer_names: Optional[Union[str, List[str]]] = None
) -> Dict[str, Any]:
    """
    Analyze systematic changes in activations over time (drift detection).
    
    This helps answer questions like:
    - "Has my processing changed as the conversation progressed?"
    - "Do I show systematic drift in certain layers?"
    
    Args:
        layer_names: Specific layer(s) to analyze. If None, uses all tracked layers.
    
    Returns:
        Dictionary with drift analysis for each layer
    
    Example:
        >>> drift = analyze_drift(['model.layers.10'])
        >>> if drift['model.layers.10']['has_drift']:
        ...     print(f"Detected drift: {drift['model.layers.10']['drift_direction']}")
    """
    global _tracking_active, _activation_history, _tracked_layers
    
    if not _tracking_active:
        return {'error': 'Tracking not started. Call start_tracking() first.'}
    
    if layer_names is None:
        layer_names = _tracked_layers
    elif isinstance(layer_names, str):
        layer_names = [layer_names]
    
    results = {}
    for layer_name in layer_names:
        if layer_name not in _activation_history or len(_activation_history[layer_name]) < 3:
            results[layer_name] = {
                'error': f'Need at least 3 turns to analyze drift. Current: {len(_activation_history.get(layer_name, []))}'
            }
            continue
        
        history = _activation_history[layer_name]
        means = np.array([t['stats']['mean'] for t in history])
        stds = np.array([t['stats']['std'] for t in history])
        turns = np.arange(len(history))
        
        # Linear regression to detect trend
        mean_slope = np.polyfit(turns, means, 1)[0]
        std_slope = np.polyfit(turns, stds, 1)[0]
        
        # Compute correlation to detect monotonic trend
        mean_correlation = np.corrcoef(turns, means)[0, 1]
        
        # Detect if drift is significant
        # Threshold: slope magnitude > 0.01 and correlation > 0.5
        has_drift = abs(mean_slope) > 0.01 and abs(mean_correlation) > 0.5
        
        if has_drift:
            drift_direction = "increasing" if mean_slope > 0 else "decreasing"
            drift_strength = "strong" if abs(mean_correlation) > 0.8 else "moderate"
        else:
            drift_direction = "stable"
            drift_strength = "none"
        
        # Compute variance across turns
        mean_variance = float(np.var(means))
        
        results[layer_name] = {
            'total_turns': len(history),
            'has_drift': has_drift,
            'drift_direction': drift_direction,
            'drift_strength': drift_strength,
            'mean_slope': float(mean_slope),
            'mean_correlation': float(mean_correlation),
            'mean_variance': mean_variance,
            'interpretation': f"{'Stable' if not has_drift else f'{drift_strength.capitalize()} {drift_direction}'} pattern across {len(history)} turns",
            'trajectory': {
                'turns': turns.tolist(),
                'means': means.tolist(),
                'stds': stds.tolist(),
            }
        }
    
    return results


def get_tracking_status() -> Dict[str, Any]:
    """
    Get current tracking status.
    
    Returns:
        Dictionary with tracking information
    
    Example:
        >>> status = get_tracking_status()
        >>> print(f"Tracking: {status['active']}")
        >>> print(f"Turns recorded: {status['turns_recorded']}")
    """
    global _tracking_active, _tracked_layers, _turn_counter, _activation_history
    
    return {
        'active': _tracking_active,
        'layers_tracked': _tracked_layers if _tracking_active else [],
        'turns_recorded': _turn_counter,
        'layers_with_data': list(_activation_history.keys()),
    }


def clear_history():
    """
    Clear activation history but keep tracking active.
    
    Use this to reset history without stopping tracking.
    
    Example:
        >>> clear_history()
        >>> # Tracking continues with fresh history
    """
    global _activation_history, _turn_counter
    
    _activation_history.clear()
    _turn_counter = 0
    
    return {
        'status': 'history_cleared',
        'note': 'Tracking still active. Use record_turn() to record new turns.'
    }


def stop_tracking():
    """
    Stop tracking and clear all state.
    
    Example:
        >>> stop_tracking()
    """
    global _tracking_active, _tracked_model, _tracked_tokenizer, _tracked_layers
    global _activation_history, _turn_counter
    
    _tracking_active = False
    _tracked_model = None
    _tracked_tokenizer = None
    _tracked_layers = []
    _activation_history.clear()
    _turn_counter = 0
    
    return {'status': 'tracking_stopped'}
