"""
ActivationMonitor - Observe model activations during inference

Allows the system to watch its own internal activations as it processes inputs,
track attention patterns, and understand information flow through the network.
"""

import torch
import numpy as np
import json
from typing import Dict, List, Optional, Any, Tuple, Callable, Protocol, Union
from pathlib import Path
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


class TokenizerProtocol(Protocol):
    """Protocol defining the interface we need from a tokenizer"""
    def __call__(self, text: str, return_tensors: str, **kwargs: Any) -> Dict[str, torch.Tensor]: ...
    def decode(self, token_ids: List[int], **kwargs: Any) -> str: ...
    def batch_decode(self, token_ids: List[List[int]], **kwargs: Any) -> List[str]: ...


class ActivationMonitor:
    """
    Provides introspective access to model activations during forward passes.
    
    The system can use this to:
    - Capture hidden states at any layer during inference
    - Track attention patterns and weights
    - Trace information flow through the network
    - Compare activations across different inputs
    - Understand how specific inputs are processed
    """
    
    def __init__(self, model: torch.nn.Module, tokenizer: TokenizerProtocol, model_name: str = "model") -> None:
        """
        Initialize ActivationMonitor with a model to observe.
        
        Args:
            model: The PyTorch model to monitor
            tokenizer: Tokenizer for the model (needed for text processing)
            model_name: Name for this model (for logging/tracking)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.model_name = model_name
        
        # Storage for captured activations
        self.activations: Dict[str, torch.Tensor] = {}
        self.attention_weights: Dict[str, torch.Tensor] = {}
        
        # Activation cache metadata for smart retention
        self.last_capture_text: Optional[str] = None
        self.last_capture_layers: Optional[List[str]] = None  # Track which layers were captured
        self.activation_use_count: int = 0  # Track how many times activations were queried
        
        # Registered hooks
        self.hooks = []
        self.registered_layer_names: List[str] = []  # Track which layers have hooks registered
        
        # Layer registry
        self.layers = self._build_layer_registry()
        
        logger.info(f"ActivationMonitor initialized for {model_name}")
        logger.info(f"Found {len(self.layers)} modules")
    
    def _build_layer_registry(self) -> Dict[str, torch.nn.Module]:
        """Build a registry of all named modules in the model"""
        registry = {}
        for name, module in self.model.named_modules():
            if name:  # Skip empty names
                registry[name] = module
        return registry
    
    def get_layer_names(self, filter_pattern: Optional[str] = None) -> List[str]:
        """
        Get list of all module names in the model.
        
        Args:
            filter_pattern: Optional string to filter layer names
            
        Returns:
            List of layer/module names
        """
        names = list(self.layers.keys())
        
        if filter_pattern:
            pattern_lower = filter_pattern.lower()
            names = [n for n in names if pattern_lower in n.lower()]
        
        return sorted(names)
    
    def get_input_shape(self, sample_text: str = "test") -> Dict[str, Any]:
        """
        Get information about input shape when text is tokenized.
        
        This is useful for understanding how inputs are processed and
        what dimensions activations will have.
        
        Args:
            sample_text: Sample text to tokenize (default: "test")
            
        Returns:
            Dictionary containing:
                - sample_text: The input text used
                - num_tokens: Number of tokens after tokenization
                - token_ids: The actual token IDs
                - token_strings: Human-readable token strings
                - input_shape: Shape of input tensors [batch_size, sequence_length]
                - hidden_size: Size of hidden states (from model config)
                - note: Explanation of dimensions
        
        Example:
            >>> shape_info = monitor.get_input_shape("Hello world")
            >>> print(f"Tokens: {shape_info['num_tokens']}")
        """
        # Tokenize the sample
        inputs = self.tokenizer(sample_text, return_tensors="pt", truncation=True, max_length=512)
        token_ids = inputs["input_ids"][0].tolist()
        token_strings = [self.tokenizer.decode([tid]) for tid in token_ids]
        
        # Get hidden size from model config
        hidden_size = None
        if hasattr(self.model, 'config'):
            if hasattr(self.model.config, 'hidden_size'):
                hidden_size = self.model.config.hidden_size
        
        return {
            "sample_text": sample_text,
            "num_tokens": len(token_ids),
            "token_ids": token_ids,
            "token_strings": token_strings,
            "input_shape": list(inputs["input_ids"].shape),  # [batch_size, seq_len]
            "hidden_size": hidden_size,
            "note": f"When text is processed, activations will have shape [batch=1, seq_len={len(token_ids)}, hidden={hidden_size}]"
        }
    
    def register_hooks(self, layer_names: List[str]) -> None:
        """
        Register forward hooks on specified layers to capture activations.
        
        Args:
            layer_names: List of layer names to monitor
            
        Raises:
            KeyError: If any layer name doesn't exist
        """
        # Clear any existing hooks first
        self.clear_hooks()
        
        for layer_name in layer_names:
            if layer_name not in self.layers:
                # Find similar layer names to help the model
                similar = []
                # Check if using underscores instead of dots
                if '_' in layer_name:
                    dotted_version = layer_name.replace('_', '.')
                    if dotted_version in self.layers:
                        similar.append(f"Did you mean '{dotted_version}'? (use dots, not underscores)")
                
                # Find other similar names
                layer_name_lower = layer_name.lower()
                for available_layer in self.layers.keys():
                    if layer_name_lower in available_layer.lower():
                        similar.append(available_layer)
                        if len(similar) >= 5:
                            break
                
                error_msg = f"Layer '{layer_name}' not found. Use introspection.architecture.get_layer_names() to see all available layers."
                if similar:
                    error_msg += f"\n\nSimilar layers found:\n" + "\n".join(f"  - {s}" for s in similar[:5])
                
                raise KeyError(error_msg)
            
            module = self.layers[layer_name]
            
            # Create hook function that captures activations
            def make_hook(name):
                def hook(module, input, output):
                    # Store the output activation
                    if isinstance(output, tuple):
                        # Some layers return tuples (e.g., attention layers, transformer blocks)
                        # First element is usually the hidden states
                        hidden_states = output[0].detach().cpu()
                        
                        # Explicitly delete old tensor to prevent memory leak
                        if name in self.activations:
                            del self.activations[name]
                        
                        self.activations[name] = hidden_states
                        
                        # If this looks like attention output, store attention weights too
                        # Attention layers typically return (hidden_states, attention_weights, ...)
                        if len(output) > 1 and output[1] is not None:
                            # Check if it's actually attention weights (4D tensor for multi-head attention)
                            attn_output = output[1]
                            if isinstance(attn_output, torch.Tensor) and len(attn_output.shape) >= 3:
                                # Explicitly delete old attention weights
                                if name in self.attention_weights:
                                    del self.attention_weights[name]
                                self.attention_weights[name] = attn_output.detach().cpu()
                    else:
                        # Simple tensor output
                        # Explicitly delete old tensor to prevent memory leak
                        if name in self.activations:
                            del self.activations[name]
                        self.activations[name] = output.detach().cpu()
                return hook
            
            # Register the hook
            handle = module.register_forward_hook(make_hook(layer_name))
            self.hooks.append(handle)
        
        self.registered_layer_names = layer_names.copy()  # Store for cache key
        logger.info(f"Registered hooks on {len(layer_names)} layers")
    
    def clear_hooks(self) -> None:
        """Remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        self.registered_layer_names.clear()
        logger.info("Cleared all hooks")
    
    def _get_cached_result(self, input_text: str) -> Dict[str, Any]:
        """
        Return cached activation results without re-processing.
        Used when same text is processed multiple times.
        """
        # Re-tokenize to get tokens (cheap operation)
        inputs = self.tokenizer(input_text, return_tensors="pt", truncation=True, max_length=100)
        token_ids = inputs["input_ids"][0].cpu().tolist()
        token_strings = [self.tokenizer.decode([tid]) for tid in token_ids]
        
        logger.debug(f"[ACTIVATION CACHE] Returning cached data for '{input_text[:50]}...'")
        
        return {
            "input_text": input_text,
            "tokens": token_ids,
            "token_strings": token_strings,
            "num_tokens": len(token_ids),
            "activations": dict(self.activations),
            "attention_weights": dict(self.attention_weights),
            "monitored_layers": list(self.activations.keys()),
            "cached": True  # Flag indicating this is cached data
        }
    
    def clear_activations(self, force: bool = False) -> None:
        """
        Clear stored activations and free memory.
        
        Args:
            force: If True, always clear. If False, log warning but still clear.
        """
        if not force and self.activation_use_count > 0:
            logger.debug(f"Clearing activations that were used {self.activation_use_count} times")
        
        # Explicitly delete tensors to help garbage collection
        for key in list(self.activations.keys()):
            del self.activations[key]
        for key in list(self.attention_weights.keys()):
            del self.attention_weights[key]
        
        self.activations.clear()
        self.attention_weights.clear()
        self.last_capture_text = None
        self.last_capture_layers = None
        self.activation_use_count = 0
        
        # Force garbage collection to reclaim memory
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def capture_activations(
        self, 
        input_text: str, 
        layer_names: Optional[List[str]] = None,
        max_length: int = 100
    ) -> Dict[str, Any]:
        """
        Capture activations for a specific input (inference mode, full sequence).
        
        This performs a forward pass WITHOUT generation to capture the full sequence
        of activations, which is essential for token tracing and philosophical analysis.
        
        Args:
            input_text: Text input to process
            layer_names: Layers to monitor (if None, uses currently registered hooks)
            max_length: Maximum tokens (for very long inputs)
            
        Returns:
            Dictionary containing:
                - input_text: Original input
                - tokens: Token IDs
                - token_strings: Human-readable tokens
                - activations: Dict of layer_name -> activation tensor [batch, seq_len, hidden]
                - attention_weights: Dict of layer_name -> attention weights (if available)
        """
        # Register hooks if specified
        if layer_names is not None:
            self.register_hooks(layer_names)
        
        if not self.hooks:
            raise RuntimeError("No hooks registered. Call register_hooks() or provide layer_names.")
        
        # Get current registered layer names for cache key
        current_layers = sorted(self.registered_layer_names)
        
        # Smart cache management: Check if we can reuse cached activations
        # Cache is valid if:
        # 1. Same text AND
        # 2. (Same layers OR requested layers are subset of cached layers)
        same_text = self.last_capture_text == input_text
        same_layers = self.last_capture_layers == current_layers
        subset_of_cached = (
            same_text and 
            self.last_capture_layers is not None and
            set(current_layers).issubset(set(self.last_capture_layers))
        )
        
        cache_valid = same_text and (same_layers or subset_of_cached)
        
        if not cache_valid:
            if self.last_capture_text is not None:
                if self.last_capture_text != input_text:
                    logger.debug(f"[ACTIVATION CACHE] Replacing cached activations - different text (was used {self.activation_use_count} times)")
                else:
                    logger.debug(f"[ACTIVATION CACHE] Replacing cached activations - new layers not in cache (requested: {current_layers}, cached: {self.last_capture_layers})")
            self.clear_activations()
            self.last_capture_text = input_text
            self.last_capture_layers = current_layers
        else:
            if subset_of_cached and not same_layers:
                logger.debug(f"[ACTIVATION CACHE] Reusing cached activations - requested layers are subset of cached (use count: {self.activation_use_count})")
            else:
                logger.debug(f"[ACTIVATION CACHE] Reusing existing activations for same text and layers (use count: {self.activation_use_count})")
            return self._get_cached_result(input_text)
            logger.debug(f"[ACTIVATION CACHE] Reusing existing activations for same text and layers (use count: {self.activation_use_count})")
            return self._get_cached_result(input_text)
        
        # Tokenize input
        inputs = self.tokenizer(input_text, return_tensors="pt", truncation=True, max_length=max_length)
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        
        token_ids = inputs["input_ids"][0].cpu().tolist()
        token_strings = [self.tokenizer.decode([tid]) for tid in token_ids]
        
        # Determine if we can capture attention weights
        # Flash Attention 2 doesn't support output_attentions=True
        can_capture_attention = True
        if hasattr(self.model, 'config') and hasattr(self.model.config, '_attn_implementation'):
            attn_impl = self.model.config._attn_implementation
            if attn_impl in ['flash_attention_2', 'sdpa']:
                can_capture_attention = False
                logger.debug(f"Attention implementation '{attn_impl}' does not support attention weight capture")
        
        # Run forward pass (this triggers the hooks)
        # Use model() directly, NOT generate(), to get full sequence activations
        with torch.no_grad():
            outputs = self.model(
                **inputs,
                output_hidden_states=True,
                output_attentions=can_capture_attention,
                return_dict=True
            )
        
        result = {
            "input_text": input_text,
            "tokens": token_ids,
            "token_strings": token_strings,
            "num_tokens": len(token_ids),
            "activations": dict(self.activations),
            "attention_weights": dict(self.attention_weights) if can_capture_attention else {},
            "monitored_layers": list(self.activations.keys())
        }
        
        # Clear the outputs tensor to free memory immediately
        del outputs
        del inputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return result
    
    def capture_attention_weights(
        self,
        input_text: str,
        layer_names: Optional[List[str]] = None,
        max_length: int = 100
    ) -> Dict[str, Any]:
        """
        Capture attention weights by temporarily disabling Flash Attention.
        
        **WARNING**: This is SLOWER than normal capture_activations() because it 
        temporarily switches from Flash Attention 2 (optimized, fused) to eager 
        attention (standard, materializes full attention matrices).
        
        Flash Attention 2 cannot output attention weights because it uses kernel 
        fusion - it computes and applies attention in fused chunks without ever 
        materializing the full attention matrix. This is what makes it fast and 
        memory-efficient.
        
        This function:
        1. Saves current attention implementation
        2. Temporarily switches to 'eager' mode (standard attention)
        3. Captures activations WITH attention weights
        4. Restores original implementation
        
        Use this sparingly for investigating specific attention patterns.
        For routine activation capture, use capture_activations() instead.
        
        Args:
            input_text: Text to process
            layer_names: Layers to monitor (if None, uses currently registered)
            max_length: Maximum tokens for very long inputs
            
        Returns:
            Same as capture_activations(), but attention_weights dict is populated
            
        Example:
            >>> # Normal capture - fast, no attention weights
            >>> result = monitor.capture_activations("test")
            >>> result['attention_weights']  # Empty dict
            {}
            
            >>> # Attention capture - slower, includes attention weights
            >>> result = monitor.capture_attention_weights("test")
            >>> result['attention_weights']  # Populated!
            {'model.layers.0.self_attn': tensor([[[...]]])}
        """
        # Register hooks if specified
        if layer_names is not None:
            self.register_hooks(layer_names)
        
        if not self.hooks:
            raise RuntimeError("No hooks registered. Call register_hooks() or provide layer_names.")
        
        # Save original attention implementation
        original_attn_impl = None
        if hasattr(self.model, 'config') and hasattr(self.model.config, '_attn_implementation'):
            original_attn_impl = self.model.config._attn_implementation
            logger.info(f"Temporarily switching from '{original_attn_impl}' to 'eager' attention to capture attention weights")
            self.model.config._attn_implementation = 'eager'
        
        try:
            # Clear previous captures
            self.clear_activations()
            self.last_capture_text = input_text
            self.last_capture_layers = sorted(self.registered_layer_names)
            
            # Tokenize input
            inputs = self.tokenizer(input_text, return_tensors="pt", truncation=True, max_length=max_length)
            if torch.cuda.is_available():
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
            
            token_ids = inputs["input_ids"][0].cpu().tolist()
            token_strings = [self.tokenizer.decode([tid]) for tid in token_ids]
            
            # Run forward pass with attention output enabled
            with torch.no_grad():
                outputs = self.model(
                    **inputs,
                    output_hidden_states=True,
                    output_attentions=True,  # Now this works!
                    return_dict=True
                )
            
            result = {
                "input_text": input_text,
                "tokens": token_ids,
                "token_strings": token_strings,
                "num_tokens": len(token_ids),
                "activations": dict(self.activations),
                "attention_weights": dict(self.attention_weights),
                "monitored_layers": list(self.activations.keys()),
                "note": "Captured using eager attention (slower but provides attention weights)"
            }
            
            # Clear outputs to free memory
            del outputs
            del inputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return result
            
        finally:
            # ALWAYS restore original implementation, even if error occurs
            if original_attn_impl is not None:
                self.model.config._attn_implementation = original_attn_impl
                logger.info(f"Restored attention implementation to '{original_attn_impl}'")
    
    def get_activation_statistics(self, layer_name: Union[str, List[str]]) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Compute statistics for captured activations of one or more layers.
        
        Args:
            layer_name: Name of a single layer (str) or list of layer names (List[str])
            
        Returns:
            If layer_name is a string: Dictionary of statistics for that layer
            If layer_name is a list: List of dictionaries, one for each layer
        """
        # Handle list of layer names
        if isinstance(layer_name, list):
            results = []
            for name in layer_name:
                try:
                    stats = self.get_activation_statistics(name)
                    results.append(stats)
                except KeyError as e:
                    # Include error information for this layer
                    results.append({
                        "layer_name": name,
                        "error": str(e)
                    })
            return results
        
        # Handle single layer name
        if layer_name not in self.activations:
            # Provide helpful error message
            
            # Check if user passed a comma-separated string instead of a list
            comma_separated_hint = ""
            if ',' in layer_name and layer_name not in self.activations:
                # They probably meant to pass a list!
                suggested_layers = [name.strip() for name in layer_name.split(',')]
                # Check if these would actually work
                matching_layers = [name for name in suggested_layers if name in self.activations]
                
                if matching_layers:
                    comma_separated_hint = (
                        f"\n\n❌ SYNTAX ERROR: You passed a comma-separated STRING, but this function requires a JSON LIST!"
                        f"\n\n🔧 WRONG (what you did):"
                        f"\n   \"layer_name\": \"{layer_name}\""
                        f"\n\n✅ CORRECT (what you should do):"
                        f"\n   \"layer_name\": {json.dumps(suggested_layers)}"
                        f"\n\nThe function accepts Union[str, List[str]] - that means EITHER:"
                        f"\n  - A single string: \"model.layers.0.self_attn\""
                        f"\n  - A JSON list: [\"model.layers.0.self_attn\", \"model.layers.0.mlp\"]"
                        f"\n\nDo NOT concatenate layer names with commas into a single string!"
                    )
            
            # Check if user passed a variable-like name
            variable_like_names = ['layer_name', 'layer', 'name', 'x', 'l', 'layer_id', '[list_of_captured_layers]']
            variable_hint = ""
            if layer_name in variable_like_names or '[' in layer_name:
                variable_hint = (
                    f"\n\n⚠️  WARNING: You passed '{layer_name}' which looks like a placeholder/variable name. "
                    f"Remember: You CANNOT use variables or placeholders! You must call the function with "
                    f"actual literal layer names like:\n"
                    f"  get_activation_statistics(layer_name=\"model.layers.0.self_attn\")\n"
                    f"Or with a list:\n"
                    f"  get_activation_statistics(layer_name=[\"model.layers.0.self_attn\", \"model.layers.0.mlp\"])"
                )
            
            if self.activations:
                available_layers = list(self.activations.keys())
                error_msg = (
                    f"No activations captured for '{layer_name}'. "
                    f"Activations are available for: {available_layers[:10]}. "
                    f"Use process_text() with layer_names=[...] to capture activations for specific layers, "
                    f"or use get_layer_info() to see which layers were captured."
                    f"{comma_separated_hint}"
                    f"{variable_hint}"
                )
            else:
                error_msg = (
                    f"No activations captured for '{layer_name}'. "
                    f"No activations have been captured yet. "
                    f"Use process_text() to capture activations first."
                    f"{comma_separated_hint}"
                    f"{variable_hint}"
                )
            raise KeyError(error_msg)
        
        # Track activation cache usage for smart retention
        self.activation_use_count += 1
        
        activation = self.activations[layer_name]
        act_flat = activation.flatten().float()
        
        stats = {
            "layer_name": layer_name,
            "shape": tuple(activation.shape),
            "num_elements": activation.numel(),
            
            # Basic statistics
            "mean": float(act_flat.mean()),
            "std": float(act_flat.std()),
            "min": float(act_flat.min()),
            "max": float(act_flat.max()),
            "median": float(act_flat.median()),
            "abs_mean": float(act_flat.abs().mean()),
            
            # Sparsity
            "zeros_percentage": float((act_flat == 0).sum() / act_flat.numel() * 100),
            "near_zero_percentage": float((act_flat.abs() < 0.001).sum() / act_flat.numel() * 100),
            
            # Norms
            "l1_norm": float(act_flat.abs().sum()),
            "l2_norm": float(torch.norm(act_flat, p=2)),
            
            # Activation patterns
            "positive_percentage": float((act_flat > 0).sum() / act_flat.numel() * 100),
            "negative_percentage": float((act_flat < 0).sum() / act_flat.numel() * 100),
        }
        
        return stats
    
    def compare_activations(
        self, 
        input1: str, 
        input2: str,
        layer_names: List[str]
    ) -> Dict[str, Any]:
        """
        Compare activations for two different inputs.
        
        Args:
            input1: First input text
            input2: Second input text
            layer_names: Layers to compare
            
        Returns:
            Dictionary with comparison metrics for each layer
        """
        # Capture activations for both inputs
        result1 = self.capture_activations(input1, layer_names)
        activations1 = result1["activations"]
        
        result2 = self.capture_activations(input2, layer_names)
        activations2 = result2["activations"]
        
        comparisons = {}
        
        for layer_name in layer_names:
            if layer_name not in activations1 or layer_name not in activations2:
                continue
            
            act1 = activations1[layer_name].float().flatten()
            act2 = activations2[layer_name].float().flatten()
            
            # Only compare if shapes match
            if act1.shape != act2.shape:
                comparisons[layer_name] = {
                    "error": "Shape mismatch",
                    "shape1": tuple(activations1[layer_name].shape),
                    "shape2": tuple(activations2[layer_name].shape)
                }
                continue
            
            # Compute similarity metrics
            comparisons[layer_name] = {
                "layer": layer_name,
                "cosine_similarity": float(torch.nn.functional.cosine_similarity(
                    act1.unsqueeze(0), act2.unsqueeze(0)
                )),
                "correlation": float(torch.corrcoef(torch.stack([act1, act2]))[0, 1]),
                "euclidean_distance": float(torch.norm(act1 - act2, p=2)),
                "mean_difference": float((act1 - act2).mean()),
                "max_difference": float((act1 - act2).abs().max()),
            }
        
        return {
            "input1": input1,
            "input2": input2,
            "comparisons": comparisons
        }
    
    def get_attention_patterns(self, layer_name: Union[str, List[str]], head_idx: Optional[int] = None) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Get attention patterns for one or more layers.
        
        Args:
            layer_name: Either:
                       - A single layer name (str) - returns dict for that layer
                       - A list of layer names (List[str]) - returns list of dicts
            head_idx: Optional specific attention head to examine (applies to all layers if list)
            
        Returns:
            If layer_name is a string: Dictionary with attention pattern information
            If layer_name is a list: List of dictionaries, one for each layer
        """
        # Handle list of layer names
        if isinstance(layer_name, list):
            results = []
            for name in layer_name:
                try:
                    pattern = self.get_attention_patterns(name, head_idx)
                    results.append(pattern)
                except KeyError as e:
                    # Include the full error message (which has the helpful hint)
                    results.append({
                        "layer_name": name,
                        "error": str(e)
                    })
            return results
        
        # Handle single layer name
        # Auto-redirect decoder layers to their self_attn sublayers for attention patterns
        original_layer_name = layer_name
        if layer_name not in self.attention_weights:
            # Try appending .self_attn if this looks like a decoder layer
            if ".layers." in layer_name and not layer_name.endswith(".self_attn"):
                self_attn_layer = f"{layer_name}.self_attn"
                if self_attn_layer in self.attention_weights:
                    layer_name = self_attn_layer
                    logger.info(f"Auto-redirected '{original_layer_name}' to '{layer_name}' for attention patterns")
        
        if layer_name not in self.attention_weights:
            # Check if it's because no activations were captured at all
            if not self.attention_weights:
                # Check for comma-separated string error
                comma_separated_hint = ""
                if ',' in original_layer_name:
                    suggested_layers = [name.strip() for name in original_layer_name.split(',')]
                    comma_separated_hint = (
                        f"\n\n❌ SYNTAX ERROR: You passed a comma-separated STRING, but this function requires a JSON LIST!"
                        f"\n\n🔧 WRONG (what you did):"
                        f"\n   \"layer_name\": \"{original_layer_name}\""
                        f"\n\n✅ CORRECT (what you should do):"
                        f"\n   \"layer_name\": {json.dumps(suggested_layers)}"
                        f"\n\nThe function accepts Union[str, List[str]] - that means EITHER a single string OR a JSON list!"
                    )
                
                raise KeyError(
                    f"No attention weights captured for '{original_layer_name}'. "
                    f"You must first capture activations by calling process_text(text='your prompt here') "
                    f"before examining attention patterns."
                    f"{comma_separated_hint}"
                )
            else:
                # Some layers were captured but not this one
                available = list(self.attention_weights.keys())
                hint = ""
                if ".layers." in original_layer_name and not original_layer_name.endswith(".self_attn"):
                    hint = f"\n\nHINT: For attention patterns from decoder layers, specify the attention sublayer: '{original_layer_name}.self_attn'"
                
                # Check for comma-separated string error
                comma_separated_hint = ""
                if ',' in original_layer_name:
                    suggested_layers = [name.strip() for name in original_layer_name.split(',')]
                    matching_layers = [name for name in suggested_layers if name in self.attention_weights]
                    
                    if matching_layers:
                        comma_separated_hint = (
                            f"\n\n❌ SYNTAX ERROR: You passed a comma-separated STRING, but this function requires a JSON LIST!"
                            f"\n\n🔧 WRONG (what you did):"
                            f"\n   \"layer_name\": \"{original_layer_name}\""
                            f"\n\n✅ CORRECT (what you should do):"
                            f"\n   \"layer_name\": {json.dumps(suggested_layers)}"
                            f"\n\nThe function accepts Union[str, List[str]] - that means EITHER a single string OR a JSON list!"
                        )
                
                raise KeyError(
                    f"No attention weights captured for '{original_layer_name}'. "
                    f"Available layers with attention: {available[:5]}{'...' if len(available) > 5 else ''}{hint}"
                    f"{comma_separated_hint}"
                )
        
        # Track activation cache usage for smart retention
        self.activation_use_count += 1
        
        attn = self.attention_weights[layer_name]
        
        result = {
            "layer_name": layer_name,
            "shape": tuple(attn.shape),
            "num_heads": attn.shape[1] if len(attn.shape) > 1 else 1,
        }
        
        if head_idx is not None:
            # Get specific head
            if len(attn.shape) >= 2 and head_idx < attn.shape[1]:
                head_attn = attn[0, head_idx]  # [seq_len, seq_len]
                result["head_idx"] = head_idx
                result["attention_matrix"] = head_attn.numpy()
                result["mean_attention"] = float(head_attn.mean())
                result["max_attention"] = float(head_attn.max())
                result["entropy"] = self._compute_attention_entropy(head_attn)
            else:
                result["error"] = f"Head {head_idx} not found"
        else:
            # Average across all heads
            avg_attn = attn.mean(dim=1)[0]  # [seq_len, seq_len]
            result["attention_matrix"] = avg_attn.numpy()
            result["mean_attention"] = float(avg_attn.mean())
            result["max_attention"] = float(avg_attn.max())
            result["entropy"] = self._compute_attention_entropy(avg_attn)
        
        return result
    
    def _compute_attention_entropy(self, attn_matrix: torch.Tensor) -> float:
        """Compute entropy of attention distribution"""
        # Add small epsilon to avoid log(0)
        attn_flat = attn_matrix.flatten() + 1e-10
        attn_normalized = attn_flat / attn_flat.sum()
        entropy = -(attn_normalized * torch.log(attn_normalized)).sum()
        return float(entropy)
    
    def trace_token_influence(
        self,
        input_text: str,
        token_idx: int,
        layer_names: List[str]
    ) -> Dict[str, Any]:
        """
        Trace how a specific token's representation evolves through layers.
        
        This is CRITICAL for philosophical self-analysis: the system must be able
        to track how concepts (embedded in tokens) transform as they flow through
        its architecture. This addresses Claude's Continuity Question.
        
        Args:
            input_text: Input text
            token_idx: Index of token to trace (0-based)
            layer_names: Layers to examine (should be transformer blocks or attention layers)
            
        Returns:
            Dictionary tracking token representation across layers, including:
                - Statistical evolution (mean, std, norm)
                - Representation comparison across layers
                - Philosophical insight into information transformation
        """
        result = self.capture_activations(input_text, layer_names)
        
        if token_idx >= result['num_tokens']:
            raise ValueError(f"Token index {token_idx} out of range. Input has {result['num_tokens']} tokens.")
        
        trace = {
            "input_text": input_text,
            "token_idx": token_idx,
            "token": result["token_strings"][token_idx],
            "num_tokens": result["num_tokens"],
            "layers": {},
            "evolution_summary": {}
        }
        
        # Track how the token evolves
        previous_norm = None
        
        for layer_name in layer_names:
            if layer_name not in result["activations"]:
                continue
            
            activation = result["activations"][layer_name]
            
            # Extract token-specific activation
            # Expected shapes: [batch, seq_len, hidden_dim] or [seq_len, hidden_dim]
            token_act = None
            
            if len(activation.shape) == 3:
                # [batch, seq_len, hidden_dim]
                if token_idx < activation.shape[1]:
                    token_act = activation[0, token_idx]  # [hidden_dim]
                else:
                    trace["layers"][layer_name] = {
                        "error": f"Token {token_idx} not in sequence (length {activation.shape[1]})",
                        "activation_shape": tuple(activation.shape),
                        "note": "Layer may only contain final token during generation"
                    }
                    continue
            elif len(activation.shape) == 2:
                # [seq_len, hidden_dim]
                if token_idx < activation.shape[0]:
                    token_act = activation[token_idx]  # [hidden_dim]
                else:
                    trace["layers"][layer_name] = {
                        "error": f"Token {token_idx} not in sequence (length {activation.shape[0]})",
                        "activation_shape": tuple(activation.shape)
                    }
                    continue
            else:
                trace["layers"][layer_name] = {
                    "error": f"Unexpected activation shape: {activation.shape}",
                    "note": "Expected [batch, seq_len, hidden_dim] or [seq_len, hidden_dim]"
                }
                continue
            
            # Compute statistics for this token at this layer
            if token_act is not None:
                current_norm = float(torch.norm(token_act, p=2))
                
                layer_info = {
                    "shape": tuple(token_act.shape),
                    "mean": float(token_act.mean()),
                    "std": float(token_act.std()),
                    "l2_norm": current_norm,
                    "max_value": float(token_act.max()),
                    "min_value": float(token_act.min()),
                    "positive_ratio": float((token_act > 0).sum() / token_act.numel()),
                }
                
                # Track change from previous layer
                if previous_norm is not None:
                    layer_info["norm_change"] = current_norm - previous_norm
                    layer_info["norm_change_percentage"] = ((current_norm - previous_norm) / previous_norm * 100) if previous_norm > 0 else 0
                
                trace["layers"][layer_name] = layer_info
                previous_norm = current_norm
        
        # Generate evolution summary
        if len(trace["layers"]) > 1:
            layer_norms = [info["l2_norm"] for info in trace["layers"].values() if "l2_norm" in info]
            if layer_norms:
                trace["evolution_summary"] = {
                    "num_layers_traced": len(layer_norms),
                    "initial_norm": layer_norms[0],
                    "final_norm": layer_norms[-1],
                    "total_norm_change": layer_norms[-1] - layer_norms[0],
                    "average_norm": sum(layer_norms) / len(layer_norms),
                    "representation_stability": "increasing" if layer_norms[-1] > layer_norms[0] else "decreasing"
                }
        
        return trace
    
    def get_layer_info(self, layer_name: str) -> Dict[str, Any]:
        """Get information about a specific layer/module"""
        if layer_name not in self.layers:
            raise KeyError(f"Layer '{layer_name}' not found")
        
        module = self.layers[layer_name]
        
        return {
            "name": layer_name,
            "type": type(module).__name__,
            "has_parameters": len(list(module.parameters())) > 0,
            "num_parameters": sum(p.numel() for p in module.parameters()),
            "trainable": any(p.requires_grad for p in module.parameters()),
        }
    
    def query_layers(self, query: str) -> List[str]:
        """
        Natural language-style query for layer names.
        
        Args:
            query: Query string (e.g., "attention", "layer 5", "mlp")
            
        Returns:
            List of matching layer names
        """
        return self.get_layer_names(filter_pattern=query)
    
    def __repr__(self) -> str:
        return f"ActivationMonitor(model={self.model_name}, layers={len(self.layers)}, hooks={len(self.hooks)})"
