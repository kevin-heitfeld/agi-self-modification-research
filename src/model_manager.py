"""
Model Management System
Downloads, loads, and manages the base model (Qwen2.5-3B-Instruct)
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer
from huggingface_hub import snapshot_download
from pathlib import Path
import logging
import os
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class ModelManager:
    """Manages model download, loading, and basic operations"""

    def __init__(self, model_name: str = "Qwen/Qwen2.5-3B-Instruct", cache_dir: Optional[Path] = None) -> None:
        self.model_name = model_name

        # Respect environment variables for cache directory (important for Colab!)
        if cache_dir is None:
            # Check HF_HOME and TRANSFORMERS_CACHE environment variables
            env_cache = os.environ.get('HF_HOME') or os.environ.get('TRANSFORMERS_CACHE')
            if env_cache:
                self.cache_dir = Path(env_cache)
                logger.info(f"Using cache directory from environment: {self.cache_dir}")
            else:
                self.cache_dir = Path("models")
                logger.info(f"Using default cache directory: {self.cache_dir}")
        else:
            self.cache_dir = cache_dir
            logger.info(f"Using explicit cache directory: {self.cache_dir}")

        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.model: Optional[PreTrainedModel] = None
        self.tokenizer: Optional[PreTrainedTokenizer] = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Detect GPU capabilities for optimal configuration
        self.gpu_name = None
        self.gpu_memory_gb = 0.0
        self.gpu_compute_capability = None
        if self.device == "cuda":
            self.gpu_name = torch.cuda.get_device_name(0)
            self.gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            # Get compute capability (e.g., 7.5 for T4, 8.0 for A100, 8.9 for L4)
            props = torch.cuda.get_device_properties(0)
            self.gpu_compute_capability = f"{props.major}.{props.minor}"
            logger.info(f"GPU detected: {self.gpu_name}")
            logger.info(f"GPU memory: {self.gpu_memory_gb:.1f} GB")
            logger.info(f"Compute capability: {self.gpu_compute_capability}")

        logger.info(f"ModelManager initialized for {model_name}")
        logger.info(f"Device: {self.device}")

    def get_optimal_limits(self) -> Dict[str, int]:
        """
        Get optimal token limits based on detected GPU capabilities.
        
        Returns:
            Dict with recommended limits for max_new_tokens, max_conversation_tokens, keep_recent_turns
        """
        # Default conservative limits (CPU or unknown GPU)
        limits = {
            "max_new_tokens": 400,
            "max_conversation_tokens": 1500,
            "keep_recent_turns": 2,
            "gpu_profile": "conservative"
        }
        
        if self.device != "cuda":
            logger.info("CPU detected - using conservative limits")
            return limits
        
        # Detect GPU tier based on memory and compute capability
        if "A100" in self.gpu_name or "A10" in self.gpu_name:
            # A100 (40-80 GB) or A10 (24 GB) - Ampere high-end
            # Conservative: 5000 * 5 = 25K cache (safer than 8000 * 5 = 40K)
            limits = {
                "max_new_tokens": 1000,  # More conservative than 1200
                "max_conversation_tokens": 5000,  # Safer than 8000 (prevents excessive cache)
                "keep_recent_turns": 5,  # Balanced retention
                "gpu_profile": "high_end_ampere"
            }
            logger.info(f"🚀 High-end GPU detected ({self.gpu_name}) - using generous limits")
            
        elif "L4" in self.gpu_name or (self.gpu_memory_gb >= 22 and float(self.gpu_compute_capability) >= 8.9):
            # L4 (24 GB) - Ada Lovelace
            # Conservative: cache is 5x conversation tokens (4000 * 5 = 20K cache, safe margin)
            # Previous 6000 tokens caused OOM (30K cache reached 22,985 tokens)
            limits = {
                "max_new_tokens": 850,  # Modest increase from 750
                "max_conversation_tokens": 4000,  # Safer than 6000 (prevents 30K cache OOM)
                "keep_recent_turns": 4,  # Balanced context retention
                "gpu_profile": "l4_ada"
            }
            logger.info(f"⚡ L4 GPU detected ({self.gpu_name}) - using optimized limits with Flash Attention")
            
        elif "T4" in self.gpu_name or (self.gpu_memory_gb >= 14 and float(self.gpu_compute_capability) >= 7.5):
            # T4 (16 GB) - Turing
            # Conservative: 2500 * 5 = 12.5K cache (safer than 3500 * 5 = 17.5K)
            limits = {
                "max_new_tokens": 600,  # Modest increase from 500
                "max_conversation_tokens": 2500,  # Safer than 3500
                "keep_recent_turns": 3,  # Balanced retention
                "gpu_profile": "t4_turing"
            }
            logger.info(f"✓ T4 GPU detected ({self.gpu_name}) - using balanced limits")
            
        elif "V100" in self.gpu_name or (self.gpu_memory_gb >= 14 and float(self.gpu_compute_capability) >= 7.0):
            # V100 (16-32 GB) - Volta
            limits = {
                "max_new_tokens": 600,
                "max_conversation_tokens": 2500,
                "keep_recent_turns": 3,
                "gpu_profile": "v100_volta"
            }
            logger.info(f"✓ V100 GPU detected ({self.gpu_name}) - using moderate limits")
            
        elif self.gpu_memory_gb >= 10:
            # Other GPU with decent memory (P100, etc.)
            limits = {
                "max_new_tokens": 450,
                "max_conversation_tokens": 1800,
                "keep_recent_turns": 2,
                "gpu_profile": "moderate"
            }
            logger.info(f"✓ GPU detected ({self.gpu_name}, {self.gpu_memory_gb:.1f} GB) - using moderate limits")
            
        else:
            # Small GPU or unknown
            logger.warning(f"⚠ Small GPU detected ({self.gpu_name}, {self.gpu_memory_gb:.1f} GB) - using conservative limits")
        
        logger.info(f"  Recommended limits: max_new_tokens={limits['max_new_tokens']}, "
                   f"max_conversation_tokens={limits['max_conversation_tokens']}, "
                   f"keep_recent_turns={limits['keep_recent_turns']}")
        
        return limits

    def download_model(self, use_auth_token: Optional[str] = None) -> bool:
        """
        Download model files from HuggingFace without loading into memory.
        
        This is more efficient than load_model() for just downloading,
        as it doesn't load the checkpoint shards into memory.

        Args:
            use_auth_token: HuggingFace authentication token (optional for Qwen models)

        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Downloading model: {self.model_name}")
            logger.info(f"Cache directory: {self.cache_dir}")
            logger.info("This may take several minutes for a ~6GB model...")

            # Use snapshot_download to only download files without loading into memory
            logger.info("Downloading model files (no loading into memory)...")
            snapshot_download(
                repo_id=self.model_name,
                cache_dir=str(self.cache_dir),
                token=use_auth_token,
                ignore_patterns=["*.msgpack", "*.h5", "*.ot"],  # Skip unnecessary files
                resume_download=True
            )

            logger.info(f"✓ Model files downloaded successfully to {self.cache_dir}")
            logger.info("Note: Model not loaded into memory. Use load_model() when ready to use it.")
            return True

        except Exception as e:
            logger.error(f"✗ Failed to download model: {e}")
            return False

    def load_model(self, use_auth_token: Optional[str] = None, use_flash_attention: bool = True) -> bool:
        """
        Load model from cache (downloads if not present)

        Args:
            use_auth_token: HuggingFace authentication token
            use_flash_attention: Use Flash Attention 2 for memory/speed optimization (default: True)

        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if already loaded
            if self.model is not None and self.tokenizer is not None:
                logger.info("Model already loaded")
                return True

            logger.info(f"Loading model: {self.model_name}")

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=str(self.cache_dir),
                token=use_auth_token,
                trust_remote_code=True
            )

            # Determine attention implementation
            # Flash Attention 2: Faster + more memory efficient, but needs flash-attn package
            # Eager: Required for output_attentions=True (activation inspection)
            if use_flash_attention:
                attn_impl = "flash_attention_2"
                logger.info("Attempting to use Flash Attention 2 for memory/speed optimization")
            else:
                attn_impl = "eager"
                logger.info("Using eager attention (required for activation inspection)")

            # Load model with memory optimizations
            # Flash Attention 2: Reduces attention memory from O(n²) to O(n)
            # Note: Flash Attention doesn't support output_attentions=True
            # If activation inspection is needed, will need to use eager
            flash_attention_failed = False
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    cache_dir=str(self.cache_dir),
                    token=use_auth_token,
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True,
                    attn_implementation=attn_impl
                )
                logger.info(f"✓ Model loaded with {attn_impl} attention")
                            
            except Exception as e:
                if use_flash_attention and "flash" in str(e).lower():
                    logger.warning(f"⚠ Flash Attention 2 load failed: {e}")
                    flash_attention_failed = True
                else:
                    raise
            
            # Fallback to eager if Flash Attention failed at load time
            if flash_attention_failed:
                logger.warning("⚠ Falling back to eager attention")
                logger.info("  Reloading model with eager attention...")
                
                # Clean up failed model
                if self.model is not None:
                    del self.model
                    if self.device == "cuda":
                        torch.cuda.empty_cache()
                
                # Reload with eager
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    cache_dir=str(self.cache_dir),
                    token=use_auth_token,
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True,
                    attn_implementation="eager"
                )
                logger.info("✓ Model loaded with eager attention (fallback)")
                attn_impl = "eager"  # Update for logging

            # Validate model loaded correctly
            if self.model is None:
                raise RuntimeError("Model loading returned None")

            # Verify model has parameters (checkpoint shards loaded)
            try:
                param_count = sum(p.numel() for p in self.model.parameters())
                if param_count == 0:
                    raise RuntimeError("Model has no parameters - checkpoint loading may have failed")
                logger.info(f"✓ Model has {param_count:,} parameters")
            except Exception as e:
                raise RuntimeError(f"Failed to verify model parameters: {e}")

            # Move to device
            if self.device == "cuda":
                self.model = self.model.to(self.device)
                logger.info(f"✓ Model moved to GPU")
                
                # Test Flash Attention compatibility after moving to GPU
                # Flash Attention has runtime GPU requirements (Ampere+) not checked at load time
                if attn_impl == "flash_attention_2":
                    logger.info("  Testing Flash Attention 2 compatibility on GPU...")
                    try:
                        test_input = self.tokenizer("test", return_tensors="pt")
                        test_input = {k: v.to(self.device) for k, v in test_input.items()}
                        with torch.no_grad():
                            _ = self.model.generate(**test_input, max_new_tokens=2)
                        logger.info("  ✓ Flash Attention 2 working correctly on this GPU")
                    except RuntimeError as runtime_err:
                        if "FlashAttention only supports Ampere GPUs" in str(runtime_err):
                            logger.warning(f"  ⚠ Flash Attention 2 incompatible: {runtime_err}")
                            logger.warning("  ⚠ T4 GPU detected - Flash Attention requires Ampere (A100, A10G) or newer")
                            logger.warning("  ⚠ Reloading model with eager attention...")
                            
                            # Clean up
                            del self.model
                            torch.cuda.empty_cache()
                            
                            # Reload with eager
                            self.model = AutoModelForCausalLM.from_pretrained(
                                self.model_name,
                                cache_dir=str(self.cache_dir),
                                token=use_auth_token,
                                torch_dtype=torch.float16,
                                low_cpu_mem_usage=True,
                                trust_remote_code=True,
                                attn_implementation="eager"
                            )
                            self.model = self.model.to(self.device)
                            logger.info("  ✓ Model reloaded with eager attention (fallback)")
                        else:
                            raise
            else:
                logger.warning("⚠ Running on CPU - this will be EXTREMELY slow. Enable GPU in Colab: Runtime → Change runtime type → GPU")

            logger.info(f"✓ Model loaded successfully")
            return True

        except Exception as e:
            logger.error(f"✗ Failed to load model: {e}")
            return False

    def generate(self, prompt: str, max_length: int = 100, **kwargs) -> str:
        """
        Generate text from prompt

        Args:
            prompt: Input text prompt
            max_length: Maximum length of generated text
            **kwargs: Additional generation parameters

        Returns:
            Generated text
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt")
        if self.device == "cuda":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                **kwargs
            )

        # Decode
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        if self.model is None:
            return {"status": "not_loaded"}

        info = {
            "model_name": self.model_name,
            "device": self.device,
            "num_parameters": sum(p.numel() for p in self.model.parameters()),
            "trainable_parameters": sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            "dtype": str(next(self.model.parameters()).dtype),
        }

        if self.device == "cuda":
            info["gpu_memory_allocated"] = f"{torch.cuda.memory_allocated() / 1e9:.2f} GB"
            info["gpu_memory_reserved"] = f"{torch.cuda.memory_reserved() / 1e9:.2f} GB"

        return info

    def unload_model(self):
        """Unload model from memory"""
        if self.model is not None:
            del self.model
            self.model = None

        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None

        if self.device == "cuda":
            torch.cuda.empty_cache()

        logger.info("Model unloaded from memory")


if __name__ == "__main__":
    # Test the model manager
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    print("=" * 60)
    print("MODEL MANAGER TEST")
    print("=" * 60)

    # Note: Qwen2.5 is an open model and doesn't require authentication
    print("\nTo download Qwen2.5-3B-Instruct:")
    print("1. No HuggingFace account required (fully open model)")
    print("2. Model page: https://huggingface.co/Qwen/Qwen2.5-3B-Instruct")
    print("\nThen run:")
    print("  manager = ModelManager()")
    print("  manager.download_model()  # No token needed!")
