"""
Model Configuration Presets

Pre-configured settings for different model sizes and hardware profiles.
Makes it easy to switch between models for experiments.

Author: AGI Self-Modification Research Team
Date: November 14, 2025
"""

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class ModelPreset:
    """Configuration preset for a specific model"""
    name: str
    huggingface_id: str
    parameters: int  # Total parameters
    context_length: int
    description: str
    
    # Memory requirements (GB VRAM)
    vram_fp16: float
    vram_4bit: float
    vram_8bit: float
    
    # Recommended settings
    recommended_batch_size: int
    recommended_max_tokens: int
    
    # Capabilities
    supports_flash_attention: bool = True
    supports_long_context: bool = False
    good_for_reasoning: bool = False
    good_for_coding: bool = False


# ============================================================================
# MODEL PRESETS
# ============================================================================

MODEL_PRESETS: Dict[str, ModelPreset] = {
    # Qwen 2.5 Family - Current recommendation
    "qwen2.5-3b": ModelPreset(
        name="Qwen 2.5 3B Instruct",
        huggingface_id="Qwen/Qwen2.5-3B-Instruct",
        parameters=3_090_000_000,
        context_length=32_768,
        description="Current baseline - efficient and capable",
        vram_fp16=6.0,
        vram_4bit=2.5,
        vram_8bit=3.5,
        recommended_batch_size=4,
        recommended_max_tokens=1000,
        supports_flash_attention=True,
        supports_long_context=False,
        good_for_reasoning=True,
        good_for_coding=True,
    ),
    
    "qwen2.5-7b": ModelPreset(
        name="Qwen 2.5 7B Instruct",
        huggingface_id="Qwen/Qwen2.5-7B-Instruct",
        parameters=7_610_000_000,
        context_length=128_000,  # 128K with YARN scaling!
        description="Recommended upgrade - significantly better reasoning, 4x longer context",
        vram_fp16=15.0,
        vram_4bit=4.5,
        vram_8bit=8.0,
        recommended_batch_size=2,
        recommended_max_tokens=2000,
        supports_flash_attention=True,
        supports_long_context=True,  # 128K context!
        good_for_reasoning=True,
        good_for_coding=True,
    ),
    
    "qwen2.5-1.5b": ModelPreset(
        name="Qwen 2.5 1.5B Instruct",
        huggingface_id="Qwen/Qwen2.5-1.5B-Instruct",
        parameters=1_540_000_000,
        context_length=32_768,
        description="Smaller/faster option for resource-constrained scenarios",
        vram_fp16=3.5,
        vram_4bit=1.5,
        vram_8bit=2.0,
        recommended_batch_size=8,
        recommended_max_tokens=1000,
        supports_flash_attention=True,
        supports_long_context=False,
        good_for_reasoning=False,
        good_for_coding=True,
    ),
    
    # Microsoft Phi Family
    "phi-3.5-mini": ModelPreset(
        name="Phi-3.5 Mini Instruct",
        huggingface_id="microsoft/Phi-3.5-mini-instruct",
        parameters=3_820_000_000,
        context_length=128_000,
        description="Microsoft's efficient small model with long context",
        vram_fp16=8.0,
        vram_4bit=3.0,
        vram_8bit=4.5,
        recommended_batch_size=4,
        recommended_max_tokens=1000,
        supports_flash_attention=True,
        supports_long_context=True,
        good_for_reasoning=True,
        good_for_coding=False,
    ),
    
    # Meta Llama Family
    "llama-3.2-3b": ModelPreset(
        name="Llama 3.2 3B Instruct",
        huggingface_id="meta-llama/Llama-3.2-3B-Instruct",
        parameters=3_210_000_000,
        context_length=128_000,
        description="Meta's latest small model with long context",
        vram_fp16=6.5,
        vram_4bit=2.8,
        vram_8bit=3.8,
        recommended_batch_size=4,
        recommended_max_tokens=1000,
        supports_flash_attention=True,
        supports_long_context=True,
        good_for_reasoning=True,
        good_for_coding=False,
    ),
    
    # Mistral Family
    "mistral-7b": ModelPreset(
        name="Mistral 7B Instruct v0.3",
        huggingface_id="mistralai/Mistral-7B-Instruct-v0.3",
        parameters=7_240_000_000,
        context_length=32_768,
        description="Strong general-purpose model with sliding window attention",
        vram_fp16=14.5,
        vram_4bit=4.0,
        vram_8bit=7.5,
        recommended_batch_size=2,
        recommended_max_tokens=1000,
        supports_flash_attention=True,
        supports_long_context=False,
        good_for_reasoning=True,
        good_for_coding=True,
    ),
    
    # DeepSeek Coder (specialized)
    "deepseek-coder-6.7b": ModelPreset(
        name="DeepSeek Coder 6.7B Instruct",
        huggingface_id="deepseek-ai/deepseek-coder-6.7b-instruct",
        parameters=6_700_000_000,
        context_length=16_384,
        description="Specialized for code understanding and generation",
        vram_fp16=13.5,
        vram_4bit=3.8,
        vram_8bit=7.0,
        recommended_batch_size=2,
        recommended_max_tokens=2000,
        supports_flash_attention=False,
        supports_long_context=False,
        good_for_reasoning=False,
        good_for_coding=True,
    ),
    
    # Google Gemma
    "gemma-2-2b": ModelPreset(
        name="Gemma 2 2B Instruct",
        huggingface_id="google/gemma-2-2b-it",
        parameters=2_600_000_000,
        context_length=8_192,
        description="Google's efficient small model",
        vram_fp16=5.5,
        vram_4bit=2.0,
        vram_8bit=3.0,
        recommended_batch_size=8,
        recommended_max_tokens=1000,
        supports_flash_attention=True,
        supports_long_context=False,
        good_for_reasoning=True,
        good_for_coding=False,
    ),
}


# ============================================================================
# HARDWARE PROFILES
# ============================================================================

@dataclass
class HardwareProfile:
    """Hardware-specific optimizations"""
    name: str
    vram_available_gb: float
    recommended_quantization: str  # "fp16", "8bit", "4bit"
    max_batch_size: int
    supports_flash_attention: bool
    recommended_models: list[str]  # Keys from MODEL_PRESETS


HARDWARE_PROFILES: Dict[str, HardwareProfile] = {
    "colab_free_t4": HardwareProfile(
        name="Google Colab Free (T4)",
        vram_available_gb=15.0,
        recommended_quantization="4bit",
        max_batch_size=2,
        supports_flash_attention=True,
        recommended_models=["qwen2.5-3b", "qwen2.5-7b", "phi-3.5-mini", "llama-3.2-3b"],
    ),
    
    "colab_pro_t4": HardwareProfile(
        name="Google Colab Pro (T4)",
        vram_available_gb=15.0,
        recommended_quantization="8bit",
        max_batch_size=4,
        supports_flash_attention=True,
        recommended_models=["qwen2.5-3b", "qwen2.5-7b", "phi-3.5-mini", "mistral-7b"],
    ),
    
    "colab_pro_plus_a100": HardwareProfile(
        name="Google Colab Pro+ (A100 40GB)",
        vram_available_gb=40.0,
        recommended_quantization="fp16",
        max_batch_size=8,
        supports_flash_attention=True,
        recommended_models=["qwen2.5-7b", "mistral-7b", "deepseek-coder-6.7b"],
    ),
    
    "l4_ada": HardwareProfile(
        name="NVIDIA L4 (24GB)",
        vram_available_gb=22.0,
        recommended_quantization="8bit",
        max_batch_size=4,
        supports_flash_attention=True,
        recommended_models=["qwen2.5-7b", "mistral-7b", "phi-3.5-mini"],
    ),
    
    "local_4090": HardwareProfile(
        name="NVIDIA RTX 4090 (24GB)",
        vram_available_gb=24.0,
        recommended_quantization="fp16",
        max_batch_size=4,
        supports_flash_attention=True,
        recommended_models=["qwen2.5-7b", "mistral-7b", "deepseek-coder-6.7b"],
    ),
    
    "local_3090": HardwareProfile(
        name="NVIDIA RTX 3090 (24GB)",
        vram_available_gb=24.0,
        recommended_quantization="8bit",
        max_batch_size=4,
        supports_flash_attention=True,
        recommended_models=["qwen2.5-7b", "mistral-7b", "phi-3.5-mini"],
    ),
}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_model_preset(preset_key: str) -> Optional[ModelPreset]:
    """Get model preset by key"""
    return MODEL_PRESETS.get(preset_key)


def get_hardware_profile(profile_key: str) -> Optional[HardwareProfile]:
    """Get hardware profile by key"""
    return HARDWARE_PROFILES.get(profile_key)


def recommend_model_for_hardware(hardware_key: str, prefer_reasoning: bool = True) -> str:
    """Recommend best model for given hardware profile"""
    profile = HARDWARE_PROFILES.get(hardware_key)
    if not profile:
        return "qwen2.5-3b"  # Default fallback
    
    # Filter recommended models
    candidates = profile.recommended_models
    
    if prefer_reasoning:
        # Prioritize models good for reasoning
        reasoning_models = [k for k in candidates if MODEL_PRESETS[k].good_for_reasoning]
        if reasoning_models:
            candidates = reasoning_models
    
    # Return first recommendation (they're ordered by preference)
    return candidates[0] if candidates else "qwen2.5-3b"


def list_available_models(max_vram_gb: Optional[float] = None, 
                          quantization: str = "4bit",
                          min_context: Optional[int] = None) -> list[str]:
    """
    List models that fit given constraints.
    
    Args:
        max_vram_gb: Maximum VRAM available
        quantization: Quantization level ("fp16", "8bit", "4bit")
        min_context: Minimum context length required
        
    Returns:
        List of model keys that meet the criteria
    """
    fitting_models = []
    
    for key, preset in MODEL_PRESETS.items():
        # Check VRAM
        if max_vram_gb is not None:
            if quantization == "fp16" and preset.vram_fp16 > max_vram_gb:
                continue
            elif quantization == "8bit" and preset.vram_8bit > max_vram_gb:
                continue
            elif quantization == "4bit" and preset.vram_4bit > max_vram_gb:
                continue
        
        # Check context length
        if min_context is not None and preset.context_length < min_context:
            continue
        
        fitting_models.append(key)
    
    return fitting_models


def print_model_comparison():
    """Print comparison table of all models"""
    print("\n" + "="*100)
    print("MODEL COMPARISON TABLE")
    print("="*100)
    print(f"{'Model':<25} {'Params':<10} {'Context':<10} {'VRAM (4bit)':<12} {'Reasoning':<12} {'Coding':<10}")
    print("-"*100)
    
    for key, preset in sorted(MODEL_PRESETS.items()):
        params = f"{preset.parameters / 1e9:.1f}B"
        context = f"{preset.context_length / 1000:.0f}K" if preset.context_length < 100_000 else f"{preset.context_length / 1000:.0f}K"
        vram = f"{preset.vram_4bit:.1f} GB"
        reasoning = "✅" if preset.good_for_reasoning else "❌"
        coding = "✅" if preset.good_for_coding else "❌"
        
        print(f"{preset.name:<25} {params:<10} {context:<10} {vram:<12} {reasoning:<12} {coding:<10}")
    
    print("="*100)


if __name__ == "__main__":
    print("🤖 AGI Self-Modification Research - Model Configurations\n")
    
    # Show comparison
    print_model_comparison()
    
    # Show hardware recommendations
    print("\n\n" + "="*100)
    print("HARDWARE RECOMMENDATIONS")
    print("="*100)
    
    for hw_key, hw_profile in HARDWARE_PROFILES.items():
        recommended = recommend_model_for_hardware(hw_key)
        recommended_preset = MODEL_PRESETS[recommended]
        print(f"\n📊 {hw_profile.name}")
        print(f"   VRAM: {hw_profile.vram_available_gb} GB")
        print(f"   Recommended: {recommended_preset.name}")
        print(f"   Quantization: {hw_profile.recommended_quantization}")
        print(f"   Other options: {', '.join([MODEL_PRESETS[k].name for k in hw_profile.recommended_models[1:3]])}")
    
    # Show what fits in 15GB (Colab Free)
    print("\n\n" + "="*100)
    print("MODELS THAT FIT IN 15GB (Colab Free T4)")
    print("="*100)
    fitting = list_available_models(max_vram_gb=15.0, quantization="4bit")
    for model_key in fitting:
        preset = MODEL_PRESETS[model_key]
        print(f"  ✅ {preset.name} ({preset.parameters / 1e9:.1f}B) - {preset.vram_4bit:.1f} GB")
    
    print("\n")
