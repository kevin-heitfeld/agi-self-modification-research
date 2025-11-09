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

        logger.info(f"ModelManager initialized for {model_name}")
        logger.info(f"Device: {self.device}")

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

    def load_model(self, use_auth_token: Optional[str] = None) -> bool:
        """
        Load model from cache (downloads if not present)

        Args:
            use_auth_token: HuggingFace authentication token

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

            # Load model
            # Use float16 on both GPU and CPU to avoid dtype conversion overhead
            # (CPU can handle float16, and it matches the cached model dtype)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                cache_dir=str(self.cache_dir),
                token=use_auth_token,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )

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
