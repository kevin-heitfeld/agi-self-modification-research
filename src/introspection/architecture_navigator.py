"""
Architecture Navigator - Understanding Model Structure

This module provides tools for understanding and describing neural network
architecture in natural language. It enables the system to explain its own
structure, map connections, and reason about architectural patterns.

Part of the AGI Self-Modification Research Project's introspection capabilities.

Author: AGI Self-Modification Research Team
Date: November 6, 2025
"""

import torch
import torch.nn as nn
import json
from typing import Dict, List, Optional, Set, Tuple, Any, Union
from collections import defaultdict
import re


class ArchitectureNavigator:
    """
    Navigate and understand neural network architecture.
    
    This class provides natural language descriptions of model structure,
    enabling the system to understand and explain its own architecture.
    
    Capabilities:
    - Describe overall model architecture
    - Explain individual components and layers
    - Map connections between layers
    - Generate architectural diagrams
    - Answer natural language queries about structure
    - Compare to known architectural patterns
    
    Design Philosophy:
    - Educational: Explain architecture in understandable terms
    - Introspective: Help the system understand itself
    - Query-based: Natural language interface
    - Read-only: No modifications to model structure
    
    Example:
        >>> navigator = ArchitectureNavigator(model)
        >>> summary = navigator.get_architecture_summary()
        >>> print(summary['description'])
        >>> layer_info = navigator.describe_layer('model.layers.0')
        >>> print(layer_info['explanation'])
    """
    
    def __init__(self, model: nn.Module, model_config: Optional[Dict] = None):
        """
        Initialize the Architecture Navigator.
        
        Args:
            model: The neural network model to navigate
            model_config: Optional config dict with model hyperparameters
                         (e.g., from transformers config.to_dict())
        """
        self.model = model
        self.config = model_config or {}
        
        # Cache for expensive operations
        self._layer_cache: Dict[str, Dict] = {}
        self._module_cache: Optional[Dict[str, nn.Module]] = None
        self._architecture_type: Optional[str] = None
        
        # Optional WeightInspector for weight sharing detection
        self._weight_inspector = None
        
    def set_weight_inspector(self, inspector) -> None:
        """
        Set WeightInspector for enhanced architectural analysis.
        
        When a WeightInspector is attached, the navigator can detect
        and report weight sharing (e.g., tied embeddings).
        
        Args:
            inspector: WeightInspector instance
            
        Example:
            >>> from introspection import WeightInspector, ArchitectureNavigator
            >>> inspector = WeightInspector(model)
            >>> navigator = ArchitectureNavigator(model)
            >>> navigator.set_weight_inspector(inspector)
            >>> summary = navigator.get_architecture_summary()
            >>> print(summary['weight_sharing'])
        """
        self._weight_inspector = inspector
        
    def get_architecture_summary(self) -> Dict[str, Any]:
        """
        Get a high-level summary of the model architecture.
        
        Returns:
            Dictionary containing:
            - model_type: Type of model (Transformer, CNN, RNN, etc.)
            - description: Natural language description
            - total_parameters: Total number of parameters
            - trainable_parameters: Number of trainable parameters
            - total_layers: Number of layers
            - layer_types: Count of each layer type
            - config: Model configuration (if available)
            - structure_summary: High-level structural information
        
        Example:
            >>> summary = navigator.get_architecture_summary()
            >>> print(f"{summary['model_type']}: {summary['total_parameters']:,} parameters")
        """
        # Detect model type
        model_type = self._detect_model_type()
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        # Analyze layer structure
        modules = dict(self.model.named_modules())
        layer_types = self._count_layer_types(modules)
        
        # Generate natural language description
        description = self._generate_model_description(
            model_type, total_params, layer_types
        )
        
        # Extract structure information
        structure = self._extract_structure_info()
        
        # Detect weight sharing if WeightInspector available
        weight_sharing = self._detect_weight_sharing()
        
        summary = {
            'model_type': model_type,
            'description': description,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'frozen_parameters': total_params - trainable_params,
            'total_layers': len(modules),
            'layer_types': layer_types,
            'config': self.config,
            'structure_summary': structure
        }
        
        # Add weight sharing info if detected
        if weight_sharing:
            summary['weight_sharing'] = weight_sharing
        
        return summary
    
    def describe_layer(self, layer_name: Union[str, List[str]]) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Describe one or more specific layers in natural language.
        
        Args:
            layer_name: Either:
                       - A single layer name (str) - returns dict for that layer
                       - A list of layer names (List[str]) - returns list of dicts
            
        Returns:
            If layer_name is a string:
                Dictionary containing:
                - name: Layer name
                - type: Layer type (Linear, LayerNorm, etc.)
                - explanation: Natural language explanation
                - parameters: Parameter count and details
                - input_shape: Expected input shape (if determinable)
                - output_shape: Expected output shape (if determinable)
                - role: What this layer does in the model
                - connections: What layers connect to this one
            
            If layer_name is a list:
                List of dicts (one per layer) with the same structure as above.
                If a layer has an error, its dict will contain 'error' key.
        
        Examples:
            >>> # Single layer
            >>> info = navigator.describe_layer('model.layers.0.self_attn.q_proj')
            >>> print(info['explanation'])
            
            >>> # Multiple layers (recommended for examining many layers!)
            >>> infos = navigator.describe_layer([
            ...     'model.layers.0.self_attn',
            ...     'model.layers.0.mlp',
            ...     'model.layers.1.self_attn'
            ... ])
        """
        # Handle list of layer names
        if isinstance(layer_name, list):
            results = []
            for name in layer_name:
                result = self.describe_layer(name)
                results.append(result)
            return results
        
        # Handle single layer name
        # Check cache
        if layer_name in self._layer_cache:
            return self._layer_cache[layer_name]
        
        # Get the module
        modules = dict(self.model.named_modules())
        if layer_name not in modules:
            # Check for comma-separated string error
            comma_separated_hint = ""
            if ',' in layer_name:
                suggested_layers = [name.strip() for name in layer_name.split(',')]
                # Check if these are valid layer names
                matching_layers = [name for name in suggested_layers if name in modules]
                
                if matching_layers:
                    comma_separated_hint = (
                        f"\n\n❌ SYNTAX ERROR: You passed a comma-separated STRING, but this function requires a JSON LIST!"
                        f"\n\n🔧 WRONG (what you did):"
                        f"\n   \"layer_name\": \"{layer_name}\""
                        f"\n\n✅ CORRECT (what you should do):"
                        f"\n   \"layer_name\": {json.dumps(suggested_layers)}"
                        f"\n\nThe function accepts Union[str, List[str]] - that means EITHER a single string OR a JSON list!"
                    )
            
            return {
                'error': f"Layer '{layer_name}' not found.{comma_separated_hint}",
                'available_layers': list(modules.keys())[:10]  # Show first 10
            }
        
        module = modules[layer_name]
        module_type = type(module).__name__
        
        # Count parameters
        param_count = sum(p.numel() for p in module.parameters())
        trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        
        # Get parameter details
        param_details = self._get_parameter_details(module)
        
        # Generate explanation
        explanation = self._explain_layer_type(layer_name, module_type, param_details)
        
        # Determine role in architecture
        role = self._determine_layer_role(layer_name, module_type)
        
        # Get shape information
        shapes = self._infer_layer_shapes(module, module_type)
        
        result = {
            'name': layer_name,
            'type': module_type,
            'explanation': explanation,
            'role': role,
            'parameters': {
                'total': param_count,
                'trainable': trainable,
                'frozen': param_count - trainable,
                'details': param_details
            },
            'input_shape': shapes.get('input'),
            'output_shape': shapes.get('output')
        }
        
        # Cache the result
        self._layer_cache[layer_name] = result
        return result
    
    def map_connections(self, layer_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Map how layers connect to each other.
        
        Args:
            layer_name: Optional specific layer to focus on.
                       If None, returns overall connection map.
        
        Returns:
            Dictionary containing:
            - layer: The layer being mapped (if specific)
            - upstream: Layers that feed into this layer
            - downstream: Layers this layer feeds into
            - connection_type: Type of connection (sequential, residual, etc.)
            - diagram: Text-based connection diagram
        
        Example:
            >>> connections = navigator.map_connections('model.layers.0')
            >>> print(connections['diagram'])
        """
        if layer_name:
            return self._map_layer_connections(layer_name)
        else:
            return self._map_overall_connections()
    
    def explain_component(self, component_type: str) -> Dict[str, Any]:
        """
        Explain what a type of component does.
        
        Args:
            component_type: Type of component (e.g., 'attention', 'mlp', 'embedding')
        
        Returns:
            Dictionary containing:
            - component: Component type
            - explanation: What it does
            - purpose: Why it's used
            - instances: How many instances in this model
            - locations: Where they appear
            - typical_structure: Common structure patterns
        
        Example:
            >>> info = navigator.explain_component('attention')
            >>> print(info['explanation'])
        """
        component_lower = component_type.lower()
        
        # Find all instances of this component type
        instances = []
        modules = dict(self.model.named_modules())
        
        for name, module in modules.items():
            if component_lower in name.lower():
                instances.append(name)
        
        # Get explanation based on component type
        explanation = self._get_component_explanation(component_lower)
        purpose = self._get_component_purpose(component_lower)
        structure = self._get_typical_structure(component_lower)
        
        return {
            'component': component_type,
            'explanation': explanation,
            'purpose': purpose,
            'instances_count': len(instances),
            'locations': instances[:5],  # Show first 5
            'typical_structure': structure
        }
    
    def query_architecture(self, query: str) -> Dict[str, Any]:
        """
        Answer natural language questions about the architecture.
        
        Args:
            query: Natural language query (e.g., "How many attention heads?")
        
        Returns:
            Dictionary containing:
            - query: The original query
            - answer: Natural language answer
            - details: Supporting details
            - relevant_components: Related architectural components
        
        Example:
            >>> result = navigator.query_architecture("How many layers?")
            >>> print(result['answer'])
        """
        query_lower = query.lower()
        
        # Pattern matching for common queries
        if any(word in query_lower for word in ['how many', 'number of']):
            return self._answer_count_query(query_lower)
        elif any(word in query_lower for word in ['what is', 'what does', 'explain']):
            return self._answer_explanation_query(query_lower)
        elif any(word in query_lower for word in ['where', 'location']):
            return self._answer_location_query(query_lower)
        elif any(word in query_lower for word in ['why', 'purpose']):
            return self._answer_purpose_query(query_lower)
        else:
            return {
                'query': query,
                'answer': "I can answer questions like: 'How many layers?', 'What is attention?', 'Where are the embeddings?', 'Why use LayerNorm?'",
                'suggestion': 'Try rephrasing your question.'
            }
    
    def get_weight_sharing_info(self, layer_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get information about weight sharing in the model.
        
        Args:
            layer_name: Optional specific layer to check. If None, returns all sharing.
        
        Returns:
            Dictionary with weight sharing information:
            - has_sharing: Whether any weight sharing is detected
            - summary: Natural language summary
            - groups: List of sharing groups (if layer_name is None)
            - coupled_with: List of layers coupled with layer_name (if specified)
            - implications: What this means for modifications
        
        Example:
            >>> # Get all weight sharing
            >>> info = navigator.get_weight_sharing_info()
            >>> print(info['summary'])
            >>>
            >>> # Check specific layer
            >>> info = navigator.get_weight_sharing_info('model.embed_tokens.weight')
            >>> print(f"Coupled with: {info['coupled_with']}")
        """
        if not self._weight_inspector:
            return {
                'has_sharing': False,
                'error': 'WeightInspector not attached. Use set_weight_inspector() first.',
                'suggestion': 'Attach a WeightInspector to enable weight sharing detection.'
            }
        
        try:
            if layer_name:
                # Query for specific layer
                coupled_layers = self._weight_inspector.get_shared_layers(layer_name)
                
                if not coupled_layers:
                    return {
                        'has_sharing': False,
                        'layer': layer_name,
                        'coupled_with': [],
                        'summary': f"Layer '{layer_name}' does not share weights with any other layer."
                    }
                
                implications = self._describe_weight_sharing_implications(
                    [layer_name] + coupled_layers
                )
                
                return {
                    'has_sharing': True,
                    'layer': layer_name,
                    'coupled_with': coupled_layers,
                    'num_coupled_layers': len(coupled_layers),
                    'summary': (
                        f"Layer '{layer_name}' shares weights with "
                        f"{len(coupled_layers)} other layer(s): {', '.join(coupled_layers)}"
                    ),
                    'implications': implications,
                    'warning': (
                        f"⚠️ Modifying '{layer_name}' will also affect: {', '.join(coupled_layers)}"
                    )
                }
            else:
                # Get all weight sharing
                sharing_info = self._detect_weight_sharing()
                
                if not sharing_info or not sharing_info.get('detected'):
                    return {
                        'has_sharing': False,
                        'summary': 'No weight sharing detected in this model.',
                        'note': 'All layers have independent weights.'
                    }
                
                return {
                    'has_sharing': True,
                    'num_groups': sharing_info['num_groups'],
                    'total_layers_affected': sharing_info['total_layers_affected'],
                    'groups': sharing_info['shared_groups'],
                    'summary': sharing_info['summary'],
                    'warning': sharing_info['warning']
                }
        
        except Exception as e:
            return {
                'has_sharing': False,
                'error': f"Could not query weight sharing: {e}"
            }

    def generate_diagram(self, format: str = 'text') -> str:
        """
        Generate an architectural diagram.
        
        Args:
            format: Output format ('text' or 'dot' for GraphViz)
        
        Returns:
            String representation of the architecture diagram
        
        Example:
            >>> diagram = navigator.generate_diagram('text')
            >>> print(diagram)
        """
        if format == 'text':
            return self._generate_text_diagram()
        elif format == 'dot':
            return self._generate_dot_diagram()
        else:
            return f"Unsupported format: {format}. Use 'text' or 'dot'."
    
    def compare_to_pattern(self, pattern: str) -> Dict[str, Any]:
        """
        Compare this architecture to a known pattern.
        
        Args:
            pattern: Architecture pattern name (e.g., 'transformer', 'resnet', 'bert')
        
        Returns:
            Dictionary containing:
            - pattern: Pattern name
            - matches: What matches the pattern
            - differences: What differs from the pattern
            - similarity_score: How similar (0-1)
            - explanation: Natural language comparison
        
        Example:
            >>> comparison = navigator.compare_to_pattern('transformer')
            >>> print(comparison['explanation'])
        """
        pattern_lower = pattern.lower()
        
        matches = []
        differences = []
        
        # Get model characteristics
        summary = self.get_architecture_summary()
        model_type = summary['model_type'].lower()
        layer_types = summary['layer_types']
        
        # Compare based on pattern
        if pattern_lower in ['transformer', 'gpt', 'bert', 'llama']:
            matches, differences = self._compare_transformer_pattern(
                layer_types, summary['structure_summary']
            )
        else:
            return {
                'pattern': pattern,
                'error': f"Unknown pattern '{pattern}'",
                'supported_patterns': ['transformer', 'gpt', 'bert', 'llama']
            }
        
        # Calculate similarity
        total_checks = len(matches) + len(differences)
        similarity = len(matches) / total_checks if total_checks > 0 else 0.0
        
        # Generate explanation
        explanation = self._generate_comparison_explanation(
            pattern, matches, differences, similarity
        )
        
        return {
            'pattern': pattern,
            'model_type': summary['model_type'],
            'matches': matches,
            'differences': differences,
            'similarity_score': similarity,
            'explanation': explanation
        }
    
    # ============================================================================
    # Private Helper Methods
    # ============================================================================
    
    def _detect_model_type(self) -> str:
        """Detect the type of model architecture."""
        if self._architecture_type:
            return self._architecture_type
        
        # Check config first
        if 'model_type' in self.config:
            self._architecture_type = self.config['model_type'].upper()
            return self._architecture_type
        
        # Analyze module names
        module_names = [name for name, _ in self.model.named_modules()]
        name_str = ' '.join(module_names).lower()
        
        if 'transformer' in name_str or 'attention' in name_str:
            if 'encoder' in name_str and 'decoder' in name_str:
                self._architecture_type = 'Transformer (Encoder-Decoder)'
            elif 'decoder' in name_str or 'causal' in name_str:
                self._architecture_type = 'Transformer (Decoder-only/GPT-style)'
            elif 'encoder' in name_str:
                self._architecture_type = 'Transformer (Encoder-only/BERT-style)'
            else:
                self._architecture_type = 'Transformer'
        elif 'conv' in name_str:
            self._architecture_type = 'CNN (Convolutional Neural Network)'
        elif 'lstm' in name_str or 'gru' in name_str or 'rnn' in name_str:
            self._architecture_type = 'RNN (Recurrent Neural Network)'
        else:
            self._architecture_type = 'Unknown Architecture'
        
        return self._architecture_type
    
    def _count_layer_types(self, modules: Dict[str, nn.Module]) -> Dict[str, int]:
        """Count occurrences of each layer type."""
        type_counts = defaultdict(int)
        
        for name, module in modules.items():
            module_type = type(module).__name__
            type_counts[module_type] += 1
        
        return dict(type_counts)
    
    def _detect_weight_sharing(self) -> Optional[Dict[str, Any]]:
        """
        Detect weight sharing using WeightInspector if available.
        
        Returns:
            Dictionary with weight sharing information, or None if no inspector
            or no sharing detected.
        """
        if not self._weight_inspector:
            return None
        
        try:
            shared_weights = self._weight_inspector.get_shared_weights()
            
            if not shared_weights:
                return None
            
            # Build structured information about weight sharing
            shared_groups = []
            for layer_names in shared_weights.values():
                if len(layer_names) < 2:
                    continue
                
                # Get tensor info from first layer
                first_layer = layer_names[0]
                try:
                    layer_info = self._weight_inspector.get_layer_weights(first_layer)
                    tensor_shape = layer_info['shape']
                    tensor_size = layer_info['num_parameters']
                except:
                    tensor_shape = None
                    tensor_size = None
                
                # Determine implications
                implications = self._describe_weight_sharing_implications(layer_names)
                
                shared_groups.append({
                    'layers': layer_names,
                    'tensor_shape': tensor_shape,
                    'tensor_size': tensor_size,
                    'implications': implications
                })
            
            if not shared_groups:
                return None
            
            # Generate summary
            total_shared = sum(len(g['layers']) for g in shared_groups)
            summary = (
                f"Detected {len(shared_groups)} group(s) of weight sharing "
                f"involving {total_shared} layer(s). "
                f"This is typically used for parameter efficiency (e.g., tied embeddings)."
            )
            
            return {
                'detected': True,
                'num_groups': len(shared_groups),
                'total_layers_affected': total_shared,
                'shared_groups': shared_groups,
                'summary': summary,
                'warning': (
                    "⚠️ Modifying any layer in a shared group will affect ALL layers "
                    "in that group, as they reference the same underlying tensor."
                )
            }
        
        except Exception as e:
            # Graceful degradation - don't break if detection fails
            return {
                'detected': False,
                'error': f"Could not detect weight sharing: {e}"
            }
    
    def _describe_weight_sharing_implications(self, layer_names: List[str]) -> str:
        """
        Describe the implications of weight sharing for specific layers.
        
        Args:
            layer_names: List of layer names that share weights
            
        Returns:
            Natural language description of implications
        """
        # Try to identify common patterns
        names_lower = [name.lower() for name in layer_names]
        
        # Check for embedding/output tying (common in language models)
        has_embed = any('embed' in name for name in names_lower)
        has_lm_head = any('lm_head' in name or 'output' in name for name in names_lower)
        
        if has_embed and has_lm_head:
            return (
                "Embedding and output head share weights (tied embeddings). "
                "This is a common optimization in language models that reduces "
                "parameters and can improve training. Modifying either layer "
                "affects both input embeddings and output predictions."
            )
        
        # Generic description
        return (
            f"These {len(layer_names)} layers share the same weight tensor. "
            f"Modifying any one of them will affect all {len(layer_names)} layers. "
            f"This architectural choice reduces parameters and enforces consistency."
        )
    
    def _generate_model_description(
        self, 
        model_type: str, 
        total_params: int,
        layer_types: Dict[str, int]
    ) -> str:
        """Generate a natural language description of the model."""
        desc_parts = [f"This is a {model_type} model"]
        
        # Add parameter count
        if total_params >= 1e9:
            desc_parts.append(f"with {total_params / 1e9:.2f}B parameters")
        elif total_params >= 1e6:
            desc_parts.append(f"with {total_params / 1e6:.2f}M parameters")
        else:
            desc_parts.append(f"with {total_params:,} parameters")
        
        # Add key components
        key_components = []
        if 'Linear' in layer_types:
            key_components.append(f"{layer_types['Linear']} linear layers")
        if 'LayerNorm' in layer_types:
            key_components.append(f"{layer_types['LayerNorm']} normalization layers")
        if 'Embedding' in layer_types:
            key_components.append(f"{layer_types['Embedding']} embedding layers")
        
        if key_components:
            desc_parts.append("consisting of " + ", ".join(key_components))
        
        return ". ".join(desc_parts) + "."
    
    def _extract_structure_info(self) -> Dict[str, Any]:
        """Extract high-level structural information."""
        structure = {}
        
        # From config
        if self.config:
            structure['hidden_size'] = self.config.get('hidden_size')
            structure['num_layers'] = self.config.get('num_hidden_layers') or self.config.get('num_layers')
            structure['num_attention_heads'] = self.config.get('num_attention_heads')
            structure['intermediate_size'] = self.config.get('intermediate_size')
            structure['vocab_size'] = self.config.get('vocab_size')
            structure['max_position_embeddings'] = self.config.get('max_position_embeddings')
        
        # Count transformer blocks if present
        module_names = [name for name, _ in self.model.named_modules()]
        transformer_blocks = len([n for n in module_names if 'layers.' in n and n.endswith('layers.' + n.split('.')[-1])])
        if transformer_blocks > 0:
            structure['transformer_blocks'] = transformer_blocks
        
        return {k: v for k, v in structure.items() if v is not None}
    
    def _get_parameter_details(self, module: nn.Module) -> Dict[str, Any]:
        """Get detailed parameter information for a module."""
        details = {}
        
        for name, param in module.named_parameters(recurse=False):
            details[name] = {
                'shape': list(param.shape),
                'dtype': str(param.dtype),
                'requires_grad': param.requires_grad,
                'numel': param.numel()
            }
        
        return details
    
    def _explain_layer_type(
        self, 
        layer_name: str, 
        module_type: str,
        param_details: Dict
    ) -> str:
        """Generate natural language explanation of a layer."""
        explanations = {
            'Linear': 'A fully connected linear transformation layer that applies matrix multiplication (y = xW + b).',
            'Embedding': 'An embedding layer that converts discrete tokens into continuous vector representations.',
            'LayerNorm': 'A layer normalization layer that normalizes activations to have mean 0 and variance 1.',
            'Dropout': 'A dropout layer that randomly zeros elements during training for regularization.',
            'GELU': 'A GELU (Gaussian Error Linear Unit) activation function that provides smooth non-linearity.',
            'ReLU': 'A ReLU (Rectified Linear Unit) activation function that zeros negative values.',
            'Softmax': 'A softmax activation that converts logits to probability distributions.',
            'Conv1d': 'A 1D convolutional layer that applies sliding filters to sequential data.',
            'Conv2d': 'A 2D convolutional layer that applies sliding filters to spatial data.',
        }
        
        base_explanation = explanations.get(module_type, f'A {module_type} layer.')
        
        # Add context based on location
        if 'attention' in layer_name.lower():
            if 'q_proj' in layer_name:
                base_explanation += ' This is the Query projection in self-attention.'
            elif 'k_proj' in layer_name:
                base_explanation += ' This is the Key projection in self-attention.'
            elif 'v_proj' in layer_name:
                base_explanation += ' This is the Value projection in self-attention.'
            elif 'o_proj' in layer_name or 'out_proj' in layer_name:
                base_explanation += ' This is the output projection of self-attention.'
        elif 'mlp' in layer_name.lower() or 'ffn' in layer_name.lower():
            if 'gate' in layer_name:
                base_explanation += ' This is part of the gated MLP mechanism.'
            elif 'up' in layer_name:
                base_explanation += ' This projects features up to a higher dimension.'
            elif 'down' in layer_name:
                base_explanation += ' This projects features down from the intermediate dimension.'
        elif 'embed' in layer_name.lower():
            if 'token' in layer_name:
                base_explanation += ' This embeds input tokens into the model\'s hidden space.'
            elif 'position' in layer_name:
                base_explanation += ' This adds positional information to token representations.'
        
        return base_explanation
    
    def _determine_layer_role(self, layer_name: str, module_type: str) -> str:
        """Determine the role of a layer in the architecture."""
        name_lower = layer_name.lower()
        
        if 'embed' in name_lower:
            return 'Input Processing - Converts tokens to vectors'
        elif 'attention' in name_lower:
            if any(proj in name_lower for proj in ['q_proj', 'k_proj', 'v_proj']):
                return 'Attention - Computes attention queries, keys, or values'
            else:
                return 'Attention - Enables the model to focus on relevant information'
        elif 'mlp' in name_lower or 'ffn' in name_lower:
            return 'Feed-Forward Network - Processes attended information'
        elif 'norm' in name_lower:
            return 'Normalization - Stabilizes training and activations'
        elif 'dropout' in name_lower:
            return 'Regularization - Prevents overfitting'
        elif 'lm_head' in name_lower or 'output' in name_lower:
            return 'Output - Projects to vocabulary for next token prediction'
        else:
            return 'Processing - Transforms representations'
    
    def _infer_layer_shapes(
        self, 
        module: nn.Module, 
        module_type: str
    ) -> Dict[str, Optional[Tuple]]:
        """Try to infer input/output shapes for a layer."""
        shapes = {'input': None, 'output': None}
        
        if module_type == 'Linear':
            if hasattr(module, 'in_features') and hasattr(module, 'out_features'):
                shapes['input'] = (None, module.in_features)  # None for batch size
                shapes['output'] = (None, module.out_features)
        elif module_type == 'Embedding':
            if hasattr(module, 'num_embeddings') and hasattr(module, 'embedding_dim'):
                shapes['input'] = (None,)  # Token indices
                shapes['output'] = (None, module.embedding_dim)
        elif module_type in ['LayerNorm', 'Dropout', 'GELU', 'ReLU']:
            # These preserve shape
            if hasattr(module, 'normalized_shape'):
                shapes['input'] = (None,) + tuple(module.normalized_shape)
                shapes['output'] = shapes['input']
        
        return shapes
    
    def _map_layer_connections(self, layer_name: str) -> Dict[str, Any]:
        """Map connections for a specific layer."""
        # Parse layer name to understand position
        parts = layer_name.split('.')
        
        # Find upstream and downstream layers (heuristic based on naming)
        upstream = []
        downstream = []
        
        # If it's part of a numbered layer block, find adjacent blocks
        for i, part in enumerate(parts):
            if part.isdigit():
                layer_num = int(part)
                if layer_num > 0:
                    upstream_name = '.'.join(parts[:i] + [str(layer_num - 1)] + parts[i+1:])
                    upstream.append(upstream_name)
                downstream_name = '.'.join(parts[:i] + [str(layer_num + 1)] + parts[i+1:])
                downstream.append(downstream_name)
                break
        
        # Determine connection type
        if 'attention' in layer_name.lower() and 'mlp' in layer_name.lower():
            conn_type = 'residual (skip connection)'
        elif any(x in layer_name.lower() for x in ['attention', 'mlp', 'ffn']):
            conn_type = 'sequential'
        else:
            conn_type = 'sequential'
        
        diagram = self._draw_layer_connection_diagram(layer_name, upstream, downstream)
        
        return {
            'layer': layer_name,
            'upstream': upstream,
            'downstream': downstream,
            'connection_type': conn_type,
            'diagram': diagram
        }
    
    def _map_overall_connections(self) -> Dict[str, Any]:
        """Map overall connection structure."""
        summary = self.get_architecture_summary()
        
        return {
            'overview': 'Overall architecture connection map',
            'flow': 'Input → Embedding → Transformer Blocks → Output',
            'structure': summary['structure_summary'],
            'diagram': self.generate_diagram('text')
        }
    
    def _draw_layer_connection_diagram(
        self, 
        layer: str, 
        upstream: List[str],
        downstream: List[str]
    ) -> str:
        """Draw a simple text diagram of layer connections."""
        lines = []
        
        if upstream:
            for up in upstream:
                lines.append(f"    {up}")
                lines.append("        ↓")
        
        lines.append(f"  ┌─[{layer}]")
        lines.append("  └→")
        
        if downstream:
            for down in downstream:
                lines.append("        ↓")
                lines.append(f"    {down}")
        
        return '\n'.join(lines)
    
    def _get_component_explanation(self, component: str) -> str:
        """Get explanation for a component type."""
        explanations = {
            'attention': 'Self-attention allows the model to weigh the importance of different parts of the input when processing each position. It computes queries, keys, and values to determine which tokens should influence each other.',
            'mlp': 'The Multi-Layer Perceptron (feed-forward network) processes each position independently after attention. It typically expands to a larger dimension, applies non-linearity, then projects back down.',
            'embedding': 'Embeddings convert discrete tokens (like words) into continuous vector representations that the model can process. They capture semantic relationships between tokens.',
            'layernorm': 'Layer Normalization stabilizes training by normalizing activations to have mean 0 and variance 1. This helps with gradient flow and model convergence.',
            'dropout': 'Dropout randomly zeros some activations during training to prevent overfitting and encourage the model to learn robust features.',
        }
        return explanations.get(component, f'The {component} component is part of the model architecture.')
    
    def _get_component_purpose(self, component: str) -> str:
        """Get the purpose of a component type."""
        purposes = {
            'attention': 'Enable the model to dynamically focus on relevant information and capture long-range dependencies.',
            'mlp': 'Process attended information through non-linear transformations to extract complex features.',
            'embedding': 'Convert tokens into a semantic space where similar meanings have similar representations.',
            'layernorm': 'Improve training stability and convergence speed.',
            'dropout': 'Regularize the model to prevent overfitting and improve generalization.',
        }
        return purposes.get(component, 'Part of the model\'s processing pipeline.')
    
    def _get_typical_structure(self, component: str) -> str:
        """Get typical structure for a component type."""
        structures = {
            'attention': 'Q = XW_Q, K = XW_K, V = XW_V\nAttention(Q,K,V) = softmax(QK^T / √d_k)V\nOutput = Attention·W_O',
            'mlp': 'x → Linear(expand) → Activation(GELU/ReLU) → Linear(contract)',
            'embedding': 'token_id → lookup_table[token_id] → embedding_vector',
            'layernorm': 'y = (x - mean(x)) / √(var(x) + ε)',
            'dropout': 'y = x * random_mask(p) / (1 - p)',
        }
        return structures.get(component, 'Varies by implementation.')
    
    def _answer_count_query(self, query: str) -> Dict[str, Any]:
        """Answer queries about counts (how many X?)."""
        summary = self.get_architecture_summary()
        
        if 'layer' in query:
            if 'attention' in query:
                count = summary['layer_types'].get('Linear', 0) // 4  # Rough estimate
                answer = f"There are approximately {count} attention layers in the model."
            else:
                count = summary['structure_summary'].get('num_layers', 'unknown')
                answer = f"The model has {count} transformer blocks/layers."
        elif 'parameter' in query:
            count = summary['total_parameters']
            answer = f"The model has {count:,} total parameters ({count / 1e9:.2f}B)."
        elif 'head' in query and 'attention' in query:
            count = summary['structure_summary'].get('num_attention_heads', 'unknown')
            answer = f"The model has {count} attention heads per layer."
        else:
            answer = "I'm not sure what count you're asking about. Try: 'How many layers?' or 'How many parameters?'"
        
        return {
            'query': query,
            'answer': answer,
            'details': summary['structure_summary']
        }
    
    def _answer_explanation_query(self, query: str) -> Dict[str, Any]:
        """Answer queries asking for explanations."""
        # Extract component name
        for component in ['attention', 'mlp', 'embedding', 'layernorm', 'dropout']:
            if component in query:
                return self.explain_component(component)
        
        return {
            'query': query,
            'answer': "Try asking about specific components like 'attention', 'mlp', 'embedding', etc."
        }
    
    def _answer_location_query(self, query: str) -> Dict[str, Any]:
        """Answer queries about locations."""
        modules = dict(self.model.named_modules())
        
        # Search for relevant modules
        search_terms = []
        if 'embed' in query:
            search_terms.append('embed')
        if 'attention' in query:
            search_terms.append('attn')
        if 'mlp' in query or 'feedforward' in query:
            search_terms.append('mlp')
        
        locations = []
        for name in modules.keys():
            if any(term in name.lower() for term in search_terms):
                locations.append(name)
        
        if locations:
            answer = f"Found {len(locations)} matching locations. First few: {', '.join(locations[:5])}"
        else:
            answer = "No matching locations found. Try being more specific."
        
        return {
            'query': query,
            'answer': answer,
            'locations': locations[:10]
        }
    
    def _answer_purpose_query(self, query: str) -> Dict[str, Any]:
        """Answer queries about purposes."""
        for component in ['attention', 'mlp', 'embedding', 'layernorm', 'dropout']:
            if component in query:
                info = self.explain_component(component)
                return {
                    'query': query,
                    'answer': info['purpose'],
                    'details': info
                }
        
        return {
            'query': query,
            'answer': "Try asking about specific components like 'Why use attention?' or 'Why use LayerNorm?'"
        }
    
    def _generate_text_diagram(self) -> str:
        """Generate a text-based architecture diagram."""
        summary = self.get_architecture_summary()
        structure = summary['structure_summary']
        
        lines = []
        lines.append("=" * 60)
        lines.append(f"  {summary['model_type']}")
        lines.append("=" * 60)
        lines.append("")
        lines.append("  INPUT (Token IDs)")
        lines.append("    ↓")
        lines.append("  ┌─────────────────────────────────┐")
        lines.append("  │  Token Embedding                │")
        lines.append("  │  Position Embedding             │")
        lines.append("  └─────────────────────────────────┘")
        lines.append("    ↓")
        
        num_layers = structure.get('num_layers', structure.get('transformer_blocks', '?'))
        for i in range(min(3, num_layers if isinstance(num_layers, int) else 3)):
            lines.append(f"  ┌─────────── Layer {i} ────────────┐")
            lines.append("  │  ┌─ Self-Attention ─────────┐  │")
            lines.append("  │  │  Q, K, V Projections     │  │")
            lines.append("  │  │  Attention Scores        │  │")
            lines.append("  │  └──────────────────────────┘  │")
            lines.append("  │  LayerNorm + Residual         │")
            lines.append("  │  ┌─ Feed Forward ──────────┐  │")
            lines.append("  │  │  Linear → GELU           │  │")
            lines.append("  │  │  Linear                  │  │")
            lines.append("  │  └──────────────────────────┘  │")
            lines.append("  │  LayerNorm + Residual         │")
            lines.append("  └─────────────────────────────────┘")
            lines.append("    ↓")
        
        if isinstance(num_layers, int) and num_layers > 3:
            lines.append(f"  ... ({num_layers - 3} more layers) ...")
            lines.append("    ↓")
        
        lines.append("  ┌─────────────────────────────────┐")
        lines.append("  │  Final Layer Norm               │")
        lines.append("  │  LM Head (to vocabulary)        │")
        lines.append("  └─────────────────────────────────┘")
        lines.append("    ↓")
        lines.append("  OUTPUT (Next Token Logits)")
        lines.append("")
        lines.append("=" * 60)
        
        return '\n'.join(lines)
    
    def _generate_dot_diagram(self) -> str:
        """Generate a GraphViz DOT diagram."""
        summary = self.get_architecture_summary()
        structure = summary['structure_summary']
        
        lines = ['digraph Architecture {']
        lines.append('    rankdir=TB;')
        lines.append('    node [shape=box, style=rounded];')
        lines.append('')
        lines.append('    input [label="Input Tokens"];')
        lines.append('    embed [label="Embedding Layer"];')
        lines.append('    input -> embed;')
        lines.append('')
        
        num_layers = structure.get('num_layers', 3)
        if isinstance(num_layers, int):
            for i in range(min(num_layers, 3)):
                lines.append(f'    layer{i} [label="Transformer Block {i}\\nAttention + FFN"];')
                if i == 0:
                    lines.append(f'    embed -> layer{i};')
                else:
                    lines.append(f'    layer{i-1} -> layer{i};')
            
            if num_layers > 3:
                lines.append(f'    more [label="... {num_layers - 3} more layers ...", shape=plaintext];')
                lines.append(f'    layer2 -> more;')
                lines.append(f'    more -> output;')
            else:
                lines.append(f'    layer{num_layers-1} -> output;')
        
        lines.append('')
        lines.append('    output [label="LM Head\\n(Output Logits)"];')
        lines.append('}')
        
        return '\n'.join(lines)
    
    def _compare_transformer_pattern(
        self, 
        layer_types: Dict[str, int],
        structure: Dict[str, Any]
    ) -> Tuple[List[str], List[str]]:
        """Compare against transformer pattern."""
        matches = []
        differences = []
        
        # Check for key transformer components
        if 'Linear' in layer_types and layer_types['Linear'] > 10:
            matches.append('Has many linear layers (typical of transformer)')
        else:
            differences.append('Fewer linear layers than typical transformer')
        
        if 'LayerNorm' in layer_types:
            matches.append('Uses layer normalization')
        else:
            differences.append('Missing layer normalization')
        
        if 'Embedding' in layer_types:
            matches.append('Has embedding layers')
        else:
            differences.append('Missing embedding layers')
        
        # Check structure
        if structure.get('num_attention_heads'):
            matches.append(f"Has multi-head attention ({structure['num_attention_heads']} heads)")
        else:
            differences.append('Multi-head attention count not detected')
        
        if structure.get('num_layers'):
            matches.append(f"Has stacked layers ({structure['num_layers']} layers)")
        
        return matches, differences
    
    def _generate_comparison_explanation(
        self,
        pattern: str,
        matches: List[str],
        differences: List[str],
        similarity: float
    ) -> str:
        """Generate explanation of architecture comparison."""
        lines = [f"Comparing this model to the {pattern} pattern:"]
        lines.append("")
        
        lines.append(f"Similarity: {similarity * 100:.1f}%")
        lines.append("")
        
        if matches:
            lines.append("✓ Matches:")
            for match in matches:
                lines.append(f"  • {match}")
            lines.append("")
        
        if differences:
            lines.append("✗ Differences:")
            for diff in differences:
                lines.append(f"  • {diff}")
        
        return '\n'.join(lines)
