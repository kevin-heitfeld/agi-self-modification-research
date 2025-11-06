"""
Model Management System
Downloads, loads, and manages the base model (Llama 3.2 3B)
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class ModelManager:
    """Manages model download, loading, and basic operations"""

    def __init__(self, model_name: str = "meta-llama/Llama-3.2-3B", cache_dir: Optional[Path] = None):
        self.model_name = model_name
        self.cache_dir = cache_dir or Path("models")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.model: Optional[Any] = None
        self.tokenizer: Optional[Any] = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info(f"ModelManager initialized for {model_name}")
        logger.info(f"Device: {self.device}")

    def download_model(self, use_auth_token: Optional[str] = None) -> bool:
        """
        Download model and tokenizer from HuggingFace

        Args:
            use_auth_token: HuggingFace authentication token (required for Llama models)

        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Downloading model: {self.model_name}")
            logger.info(f"Cache directory: {self.cache_dir}")
            logger.info("This may take several minutes for a ~6GB model...")

            # Download tokenizer
            logger.info("Downloading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=str(self.cache_dir),
                token=use_auth_token,
                trust_remote_code=True
            )

            # Download model
            logger.info("Downloading model weights...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                cache_dir=str(self.cache_dir),
                token=use_auth_token,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )

            logger.info(f"✓ Model downloaded successfully to {self.cache_dir}")
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
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                cache_dir=str(self.cache_dir),
                token=use_auth_token,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )

            # Move to device
            if self.device == "cuda":
                self.model = self.model.to(self.device)
                logger.info(f"✓ Model moved to GPU")

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

    # Note: You'll need a HuggingFace token to download Llama models
    print("\nTo download Llama 3.2 3B, you need:")
    print("1. HuggingFace account")
    print("2. Accept Llama 3.2 license at: https://huggingface.co/meta-llama/Llama-3.2-3B")
    print("3. Create access token at: https://huggingface.co/settings/tokens")
    print("\nThen run:")
    print("  manager = ModelManager()")
    print("  manager.download_model(use_auth_token='your_token_here')")
