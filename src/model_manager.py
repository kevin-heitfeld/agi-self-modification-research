"""
Model Management System

Downloads, loads, and manages the base model (Qwen2.5-3B-Instruct).

Features:
- Automatic GPU detection and optimal limit calculation
- Flash Attention 2 support (memory and speed optimization)
- 8-bit/4-bit model quantization (50-75% memory savings)
- 8-bit KV cache quantization (50% cache memory savings)

Quantization Usage:
    # Default: float16 (no quantization)
    manager = ModelManager()
    manager.load_model()  # ~14 GB for 7B model
    
    # 8-bit quantization (recommended)
    manager.load_model(quantize_model="8bit")  # ~7 GB for 7B model
    # - 50% memory savings
    # - Minimal quality loss (~1% perplexity)
    # - Slightly slower inference (~10-20%)
    # - Fits comfortably on L4 GPU (24 GB)
    
    # 4-bit quantization (aggressive)
    manager.load_model(quantize_model="4bit")  # ~3.5 GB for 7B model
    # - 75% memory savings
    # - Small quality loss (~2-3% perplexity)
    # - Slower inference (~20-30%)
    # - Can fit 14B+ models on L4 GPU (24 GB)
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
            # NEW: Check for MODEL_CACHE_DIR first (Nextcloud persistent storage)
            # Then fall back to HF_HOME/TRANSFORMERS_CACHE (local symlink cache)
            env_cache = (os.environ.get('MODEL_CACHE_DIR') or 
                        os.environ.get('HF_HOME') or 
                        os.environ.get('TRANSFORMERS_CACHE'))
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

    def get_optimal_limits(self, quantization: Optional[str] = None) -> Dict[str, int]:
        """
        Get optimal token limits based on detected GPU capabilities, model size, and quantization.
        
        Args:
            quantization: Quantization mode ('4bit', '8bit', or None)
        
        Returns:
            Dict with recommended limits for max_new_tokens, max_conversation_tokens
        """
        # Detect model size from name
        model_size_b = 7.0  # Default to 7B
        if "3B" in self.model_name or "2.5B" in self.model_name:
            model_size_b = 3.0
        elif "1.5B" in self.model_name or "1B" in self.model_name:
            model_size_b = 1.5
        elif "0.5B" in self.model_name:
            model_size_b = 0.5
        elif "14B" in self.model_name or "13B" in self.model_name:
            model_size_b = 14.0
        elif "32B" in self.model_name or "33B" in self.model_name or "30B" in self.model_name:
            model_size_b = 32.0
        elif "70B" in self.model_name or "72B" in self.model_name:
            model_size_b = 70.0
        
        # Calculate scaling factor with 7B as 1.0x baseline (current limits tuned for 7B)
        # Smaller models get more headroom: 3B = 2.33x, 1.5B = 4.67x
        # Larger models get tighter limits: 14B = 0.5x, 32B = 0.22x, 70B = 0.1x
        size_scale = 7.0 / model_size_b
        
        # Quantization gives us more VRAM headroom for KV cache
        # 8-bit: ~50% model memory savings → 1.5x more conversation tokens
        # 4-bit: ~75% model memory savings → 2.0x more conversation tokens
        quant_scale = 1.0
        if quantization == "8bit":
            quant_scale = 1.5
            logger.info(f"Model size: {model_size_b}B, scale factor: {size_scale:.2f}x, quantization: 8-bit (1.5x memory headroom)")
        elif quantization == "4bit":
            quant_scale = 2.0
            logger.info(f"Model size: {model_size_b}B, scale factor: {size_scale:.2f}x, quantization: 4-bit (2.0x memory headroom)")
        else:
            logger.info(f"Model size: {model_size_b}B, scale factor: {size_scale:.2f}x")
        
        # Default conservative limits (CPU or unknown GPU)
        limits = {
            "max_new_tokens": int(400 * size_scale * quant_scale),
            "max_conversation_tokens": int(1500 * size_scale * quant_scale),
            "gpu_profile": "conservative",
            "model_size_b": model_size_b,
            "quantization": quantization
        }
        
        if self.device != "cuda":
            logger.info("CPU detected - using conservative limits")
            return limits
        
        # Detect GPU tier based on memory and compute capability
        if "A100" in self.gpu_name or "A10" in self.gpu_name:
            # A100 (40-80 GB) or A10 (24 GB) - Ampere high-end
            # With 8-bit KV cache quantization: 50% memory savings
            limits = {
                "max_new_tokens": int(1200 * size_scale * quant_scale),
                "max_conversation_tokens": int(8000 * size_scale * quant_scale),
                "gpu_profile": "high_end_ampere",
                "model_size_b": model_size_b,
                "quantization": quantization
            }
            logger.info(f"🚀 High-end GPU detected ({self.gpu_name}) + {model_size_b}B model - using generous limits with 8-bit KV cache quantization")
            
        elif "L4" in self.gpu_name or (self.gpu_memory_gb >= 22 and float(self.gpu_compute_capability) >= 8.9):
            # L4 (24 GB) - Ada Lovelace
            # With 8-bit KV cache quantization: 50% memory savings
            # Reduced from 1500/9000 to 1200/7000 after OOM at iteration 19 with 14B 8-bit
            limits = {
                "max_new_tokens": int(1200 * size_scale * quant_scale),
                "max_conversation_tokens": int(7000 * size_scale * quant_scale),
                "gpu_profile": "l4_ada",
                "model_size_b": model_size_b,
                "quantization": quantization
            }
            logger.info(f"⚡ L4 GPU detected ({self.gpu_name}) + {model_size_b}B model - using optimized limits with 8-bit KV cache + Flash Attention")
            
        elif "T4" in self.gpu_name or (self.gpu_memory_gb >= 14 and float(self.gpu_compute_capability) >= 7.5):
            # T4 (16 GB) - Turing
            # With 8-bit KV cache quantization: 50% memory savings
            limits = {
                "max_new_tokens": int(700 * size_scale * quant_scale),
                "max_conversation_tokens": int(3500 * size_scale * quant_scale),
                "gpu_profile": "t4_turing",
                "model_size_b": model_size_b,
                "quantization": quantization
            }
            logger.info(f"✓ T4 GPU detected ({self.gpu_name}) + {model_size_b}B model - using balanced limits with 8-bit KV cache quantization")
            
        elif "V100" in self.gpu_name or (self.gpu_memory_gb >= 14 and float(self.gpu_compute_capability) >= 7.0):
            # V100 (16-32 GB) - Volta
            # With 8-bit KV cache quantization: 50% memory savings
            limits = {
                "max_new_tokens": int(700 * size_scale * quant_scale),
                "max_conversation_tokens": int(3500 * size_scale * quant_scale),
                "gpu_profile": "v100_volta",
                "model_size_b": model_size_b,
                "quantization": quantization
            }
            logger.info(f"✓ V100 GPU detected ({self.gpu_name}) + {model_size_b}B model - using moderate limits with 8-bit KV cache quantization")
            
        elif self.gpu_memory_gb >= 10:
            # Other GPU with decent memory (P100, etc.)
            limits = {
                "max_new_tokens": int(450 * size_scale * quant_scale),
                "max_conversation_tokens": int(1800 * size_scale * quant_scale),
                "gpu_profile": "moderate",
                "model_size_b": model_size_b,
                "quantization": quantization
            }
            logger.info(f"✓ GPU detected ({self.gpu_name}, {self.gpu_memory_gb:.1f} GB) + {model_size_b}B model - using moderate limits")
            
        else:
            # Small GPU or unknown
            logger.warning(f"⚠ Small GPU detected ({self.gpu_name}, {self.gpu_memory_gb:.1f} GB) + {model_size_b}B model - using conservative limits")
        
        logger.info(f"  Recommended limits: max_new_tokens={limits['max_new_tokens']}, "
                   f"max_conversation_tokens={limits['max_conversation_tokens']}")
        
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

    def load_model(
        self, 
        use_auth_token: Optional[str] = None, 
        use_flash_attention: bool = True,
        quantize_model: Optional[str] = None
    ) -> bool:
        """
        Load model from cache (downloads if not present)

        Args:
            use_auth_token: HuggingFace authentication token
            use_flash_attention: Use Flash Attention 2 for memory/speed optimization (default: True)
            quantize_model: Model weight quantization (None, "8bit", or "4bit")
                - None: Load in float16 (default, ~14 GB for 7B model)
                - "8bit": 8-bit quantization (~7 GB for 7B, minimal quality loss)
                - "4bit": 4-bit quantization (~3.5 GB for 7B, small quality loss)

        Returns:
            True if successful, False otherwise
        """
        # Configure CUDA memory allocator to prevent fragmentation
        # This is especially important for long-running sessions (like Colab)
        # See: https://pytorch.org/docs/stable/notes/cuda.html#environment-variables
        import os
        os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
        
        try:
            # Check if already loaded
            if self.model is not None and self.tokenizer is not None:
                logger.info("Model already loaded")
                return True

            logger.info(f"Loading model: {self.model_name}")

            # Check if model exists in local_dir format (downloaded via snapshot_download with local_dir)
            # This is used when models are downloaded directly to Nextcloud to avoid symlinks
            model_dir_name = self.model_name.replace("/", "--")
            local_dir_path = self.cache_dir / model_dir_name
            
            # Determine loading strategy
            if local_dir_path.exists() and (local_dir_path / "config.json").exists():
                # Load from local_dir (direct files, no symlinks)
                logger.info(f"Loading from local_dir format: {local_dir_path}")
                load_path = str(local_dir_path)
                use_local_files_only = True
            else:
                # Load using HuggingFace cache (may download if not present)
                logger.info(f"Loading from HuggingFace cache: {self.model_name}")
                load_path = self.model_name
                use_local_files_only = False

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                load_path,
                cache_dir=str(self.cache_dir) if not use_local_files_only else None,
                local_files_only=use_local_files_only,
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

            # Prepare quantization configuration if requested
            quantization_config = None
            if quantize_model == "8bit":
                logger.info("Using 8-bit model quantization (saves ~50% VRAM, minimal quality loss)")
                quantization_config = "8bit"  # Signal to use load_in_8bit
            elif quantize_model == "4bit":
                logger.info("Using 4-bit model quantization (saves ~75% VRAM, small quality loss)")
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",  # Normal Float 4-bit
                    bnb_4bit_use_double_quant=True,  # Nested quantization for better accuracy
                    bnb_4bit_compute_dtype=torch.float16
                )
            elif quantize_model is not None:
                logger.warning(f"Unknown quantization mode '{quantize_model}', loading in float16")
                quantization_config = None

            # Load model with memory optimizations
            # Flash Attention 2: Reduces attention memory from O(n²) to O(n)
            # Model quantization: Reduces model memory by 50-75%
            # Note: Flash Attention doesn't support output_attentions=True
            # If activation inspection is needed, will need to use eager
            flash_attention_failed = False
            try:
                # Build kwargs for model loading
                load_kwargs = {
                    "cache_dir": str(self.cache_dir) if not use_local_files_only else None,
                    "local_files_only": use_local_files_only,
                    "token": use_auth_token,
                    "low_cpu_mem_usage": True,
                    "trust_remote_code": True,
                }
                
                # Add quantization config if specified
                if quantization_config == "8bit":
                    load_kwargs["load_in_8bit"] = True
                    load_kwargs["device_map"] = "auto"  # Required for quantization
                    load_kwargs["llm_int8_enable_fp32_cpu_offload"] = True  # Allow CPU offload if needed
                elif isinstance(quantization_config, object) and hasattr(quantization_config, 'load_in_4bit'):
                    load_kwargs["quantization_config"] = quantization_config
                    load_kwargs["device_map"] = "auto"  # Required for quantization
                    load_kwargs["llm_int8_enable_fp32_cpu_offload"] = True  # Allow CPU offload if needed
                else:
                    # No quantization, use manual dtype and device placement
                    load_kwargs["torch_dtype"] = torch.float16
                
                # Add attention implementation (only if not quantizing, as quantization controls dtype)
                if quantization_config is None:
                    load_kwargs["attn_implementation"] = attn_impl
                
                self.model = AutoModelForCausalLM.from_pretrained(
                    load_path,
                    **load_kwargs
                )
                
                if quantize_model:
                    logger.info(f"✓ Model loaded with {quantize_model} quantization")
                else:
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
                
                # Rebuild kwargs with eager attention
                load_kwargs = {
                    "cache_dir": str(self.cache_dir) if not use_local_files_only else None,
                    "local_files_only": use_local_files_only,
                    "token": use_auth_token,
                    "low_cpu_mem_usage": True,
                    "trust_remote_code": True,
                    "attn_implementation": "eager"
                }
                
                # Re-add quantization if specified
                if quantization_config == "8bit":
                    load_kwargs["load_in_8bit"] = True
                    load_kwargs["device_map"] = "auto"
                    load_kwargs["llm_int8_enable_fp32_cpu_offload"] = True  # Allow CPU offload if needed
                    del load_kwargs["attn_implementation"]  # Quantization controls dtype
                elif isinstance(quantization_config, object) and hasattr(quantization_config, 'load_in_4bit'):
                    load_kwargs["quantization_config"] = quantization_config
                    load_kwargs["device_map"] = "auto"
                    load_kwargs["llm_int8_enable_fp32_cpu_offload"] = True  # Allow CPU offload if needed
                    del load_kwargs["attn_implementation"]  # Quantization controls dtype
                else:
                    load_kwargs["torch_dtype"] = torch.float16
                
                # Reload with eager
                self.model = AutoModelForCausalLM.from_pretrained(
                    load_path,
                    **load_kwargs
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

            # Move to device (skip if using quantization, which uses device_map="auto")
            if self.device == "cuda" and quantize_model is None:
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
                if quantize_model:
                    logger.info(f"✓ Quantized model loaded with device_map='auto'")
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
