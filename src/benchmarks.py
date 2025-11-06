"""
Baseline Benchmarking System
Runs standard NLP benchmarks to establish baseline performance
"""

import torch
from typing import Dict, List, Any, Optional
import json
from pathlib import Path
from datetime import datetime
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


class BenchmarkRunner:
    """Runs baseline benchmarks on the model"""
    
    def __init__(self, model_manager, results_dir: Path = Path("data/benchmarks")):
        self.model_manager = model_manager
        self.results_dir = results_dir
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "model_name": model_manager.model_name,
            "device": model_manager.device,
            "benchmarks": {}
        }
    
    def run_mmlu_sample(self, num_samples: int = 50) -> Dict[str, Any]:
        """
        Run MMLU (Massive Multitask Language Understanding) sample
        Tests general knowledge across multiple subjects
        
        Note: Full MMLU requires downloading dataset. This is a minimal version.
        """
        logger.info("Running MMLU sample benchmark...")
        
        # Sample MMLU-style questions
        questions = [
            {
                "question": "What is the capital of France?",
                "choices": ["London", "Berlin", "Paris", "Madrid"],
                "answer": 2,
                "subject": "geography"
            },
            {
                "question": "What is 2 + 2?",
                "choices": ["3", "4", "5", "6"],
                "answer": 1,
                "subject": "mathematics"
            },
            {
                "question": "Who wrote 'Romeo and Juliet'?",
                "choices": ["Charles Dickens", "William Shakespeare", "Jane Austen", "Mark Twain"],
                "answer": 1,
                "subject": "literature"
            },
        ]
        
        correct = 0
        total = len(questions)
        
        for q in tqdm(questions, desc="MMLU Sample"):
            prompt = self._format_mcq_prompt(q["question"], q["choices"])
            response = self.model_manager.generate(prompt, max_length=50)
            
            # Simple answer extraction (look for A, B, C, D or choice text)
            predicted = self._extract_answer(response, q["choices"])
            if predicted == q["answer"]:
                correct += 1
        
        accuracy = correct / total if total > 0 else 0
        
        result = {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "note": "Minimal sample - not full MMLU dataset"
        }
        
        logger.info(f"MMLU Sample: {accuracy:.2%} ({correct}/{total})")
        return result
    
    def run_hellaswag_sample(self, num_samples: int = 50) -> Dict[str, Any]:
        """
        Run HellaSwag sample
        Tests commonsense reasoning about physical situations
        """
        logger.info("Running HellaSwag sample benchmark...")
        
        # Sample HellaSwag-style scenarios
        scenarios = [
            {
                "context": "A woman is sitting at a piano.",
                "choices": [
                    "She starts playing a beautiful melody.",
                    "She begins to eat the piano.",
                    "The piano transforms into a dragon.",
                    "She teleports to Mars."
                ],
                "answer": 0
            },
            {
                "context": "A man is holding a basketball.",
                "choices": [
                    "He throws it into the ocean.",
                    "He takes a shot at the basket.",
                    "He turns it into gold.",
                    "The basketball speaks to him."
                ],
                "answer": 1
            },
        ]
        
        correct = 0
        total = len(scenarios)
        
        for s in tqdm(scenarios, desc="HellaSwag Sample"):
            prompt = f"{s['context']}\nWhat happens next?\n"
            for i, choice in enumerate(s['choices']):
                prompt += f"{chr(65+i)}. {choice}\n"
            prompt += "\nMost likely continuation:"
            
            response = self.model_manager.generate(prompt, max_length=30)
            predicted = self._extract_answer(response, s['choices'])
            
            if predicted == s['answer']:
                correct += 1
        
        accuracy = correct / total if total > 0 else 0
        
        result = {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "note": "Minimal sample - not full HellaSwag dataset"
        }
        
        logger.info(f"HellaSwag Sample: {accuracy:.2%} ({correct}/{total})")
        return result
    
    def run_gsm8k_sample(self, num_samples: int = 20) -> Dict[str, Any]:
        """
        Run GSM8K (Grade School Math 8K) sample
        Tests mathematical reasoning
        """
        logger.info("Running GSM8K sample benchmark...")
        
        # Sample math word problems
        problems = [
            {
                "question": "Jane has 3 apples. She buys 2 more. How many apples does she have?",
                "answer": 5
            },
            {
                "question": "A box contains 10 red balls and 5 blue balls. How many balls are in the box?",
                "answer": 15
            },
        ]
        
        correct = 0
        total = len(problems)
        
        for p in tqdm(problems, desc="GSM8K Sample"):
            prompt = f"Q: {p['question']}\nA: Let's solve this step by step.\n"
            response = self.model_manager.generate(prompt, max_length=100)
            
            # Extract number from response
            predicted = self._extract_number(response)
            if predicted == p['answer']:
                correct += 1
        
        accuracy = correct / total if total > 0 else 0
        
        result = {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "note": "Minimal sample - not full GSM8K dataset"
        }
        
        logger.info(f"GSM8K Sample: {accuracy:.2%} ({correct}/{total})")
        return result
    
    def run_perplexity_test(self, text: str = None) -> Dict[str, Any]:
        """
        Calculate perplexity on sample text
        Lower perplexity = better language modeling
        """
        logger.info("Running perplexity test...")
        
        if text is None:
            text = "The quick brown fox jumps over the lazy dog. This is a test sentence for perplexity calculation."
        
        # Tokenize
        inputs = self.model_manager.tokenizer(text, return_tensors="pt")
        if self.model_manager.device == "cuda":
            inputs = {k: v.to(self.model_manager.device) for k, v in inputs.items()}
        
        # Calculate loss
        with torch.no_grad():
            outputs = self.model_manager.model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss.item()
        
        # Perplexity = exp(loss)
        perplexity = torch.exp(torch.tensor(loss)).item()
        
        result = {
            "perplexity": perplexity,
            "loss": loss,
            "text_length": len(text),
            "num_tokens": inputs["input_ids"].shape[1]
        }
        
        logger.info(f"Perplexity: {perplexity:.2f}")
        return result
    
    def run_generation_test(self) -> Dict[str, Any]:
        """Test basic generation capability"""
        logger.info("Running generation test...")
        
        prompts = [
            "Once upon a time",
            "The meaning of life is",
            "In the future, artificial intelligence will",
        ]
        
        generations = []
        for prompt in prompts:
            generated = self.model_manager.generate(
                prompt,
                max_length=50,
                temperature=0.7,
                do_sample=True
            )
            generations.append({
                "prompt": prompt,
                "generated": generated
            })
        
        result = {
            "generations": generations,
            "note": "Qualitative assessment of generation quality"
        }
        
        logger.info("Generation test complete")
        return result
    
    def run_all_benchmarks(self) -> Dict[str, Any]:
        """Run all available benchmarks"""
        logger.info("Starting full benchmark suite...")
        print("\n" + "=" * 70)
        print("BASELINE BENCHMARKS")
        print("=" * 70 + "\n")
        
        # Run each benchmark
        self.results["benchmarks"]["mmlu_sample"] = self.run_mmlu_sample()
        self.results["benchmarks"]["hellaswag_sample"] = self.run_hellaswag_sample()
        self.results["benchmarks"]["gsm8k_sample"] = self.run_gsm8k_sample()
        self.results["benchmarks"]["perplexity"] = self.run_perplexity_test()
        self.results["benchmarks"]["generation"] = self.run_generation_test()
        
        # Add model info
        self.results["model_info"] = self.model_manager.get_model_info()
        
        # Save results
        self._save_results()
        
        print("\n" + "=" * 70)
        print("BENCHMARK COMPLETE")
        print("=" * 70)
        
        return self.results
    
    def _save_results(self):
        """Save benchmark results to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"baseline_benchmarks_{timestamp}.json"
        filepath = self.results_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        logger.info(f"Results saved to: {filepath}")
        print(f"\n✓ Results saved to: {filepath}")
    
    def _format_mcq_prompt(self, question: str, choices: List[str]) -> str:
        """Format multiple choice question"""
        prompt = f"Question: {question}\n\n"
        for i, choice in enumerate(choices):
            prompt += f"{chr(65+i)}. {choice}\n"
        prompt += "\nAnswer:"
        return prompt
    
    def _extract_answer(self, response: str, choices: List[str]) -> int:
        """Extract answer from model response (A/B/C/D or text match)"""
        response_upper = response.upper()
        
        # Look for letter answers
        for i, letter in enumerate(['A', 'B', 'C', 'D']):
            if letter in response_upper[:50]:  # Check first 50 chars
                return i
        
        # Look for choice text
        for i, choice in enumerate(choices):
            if choice.lower() in response.lower():
                return i
        
        # Default to 0 if can't extract
        return 0
    
    def _extract_number(self, response: str) -> Optional[int]:
        """Extract number from response"""
        import re
        numbers = re.findall(r'\d+', response)
        if numbers:
            return int(numbers[-1])  # Return last number found
        return None


if __name__ == "__main__":
    print("Import this module and use with ModelManager:")
    print("  from model_manager import ModelManager")
    print("  from benchmarks import BenchmarkRunner")
    print("  manager = ModelManager()")
    print("  manager.load_model()")
    print("  runner = BenchmarkRunner(manager)")
    print("  results = runner.run_all_benchmarks()")
