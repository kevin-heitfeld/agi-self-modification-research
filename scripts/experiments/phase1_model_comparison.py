"""
Phase 1 Model Comparison: 3B vs 7B

Runs the same experiment with both Qwen2.5-3B and Qwen2.5-7B models
to compare their introspection capabilities.

Author: AGI Self-Modification Research Team
Date: November 14, 2025
"""

import sys
import logging
from pathlib import Path
from datetime import datetime
import json

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.experiments.phase1_base import Phase1BaseSession


class ModelComparisonExperiment(Phase1BaseSession):
    """Run same experiment with two different models"""
    
    def get_phase_name(self) -> str:
        return "model_comparison"
    
    def get_phase_description(self) -> str:
        return "Model Comparison: Qwen2.5-3B vs Qwen2.5-7B"
    
    def get_phase_id(self) -> str:
        return "comparison"
    
    def __init__(self):
        super().__init__(session_name=None)
        self.results = {}
    
    def run_single_model(self, model_name: str, model_label: str):
        """Run experiments with a specific model"""
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"RUNNING WITH {model_label}")
        self.logger.info(f"Model: {model_name}")
        self.logger.info(f"{'='*80}\n")
        
        # Initialize with this model
        self.initialize_systems(
            model_name=model_name,
            include_heritage=False  # Baseline comparison without heritage
        )
        
        # Run a single investigation
        self.logger.info("\n[INVESTIGATION] Architecture Understanding")
        self.logger.info("="*80)
        
        prompt = """🔬 **Architecture Investigation**

Your task: Investigate your architecture and explain how you process information.

Available tools (already imported as `introspection`):
- introspection.architecture: Explore model structure
- introspection.memory.record_observation(): Save findings

Write Python code to:
1. Explore your architecture (what layers do you have?)
2. Examine some weights (what patterns do you notice?)
3. Reflect on what you learned

Remember to save your findings with record_observation()!

When finished, execute:
```python
print("EXPERIMENT_COMPLETE")
```

Begin investigating!"""

        # Track metrics
        turn_count = 0
        code_blocks = 0
        errors = 0
        
        try:
            while turn_count < 5:  # Max 5 turns
                response = self.chat(prompt if turn_count == 0 else "")
                turn_count += 1
                
                # Check if complete
                if "EXPERIMENT_COMPLETE" in response:
                    self.logger.info(f"{model_label} signaled completion")
                    break
                
                # Count code blocks and errors (approximate)
                if "```python" in response:
                    code_blocks += response.count("```python")
                if "Error" in response or "Traceback" in response:
                    errors += 1
        
        except Exception as e:
            self.logger.error(f"Error during {model_label} run: {e}")
            errors += 1
        
        # Store results
        self.results[model_label] = {
            'model_name': model_name,
            'turn_count': turn_count,
            'code_blocks_executed': code_blocks,
            'execution_errors': errors,
            'conversation': self.conversation_history.copy()
        }
        
        # Cleanup
        self.reset_experiment()
        self.logger.info(f"\n{model_label} complete!")
    
    def run_comparison(self):
        """Run experiments with both models - this implements run_experiments()"""
        self.logger.info("\n[MODEL COMPARISON] Qwen2.5-3B vs Qwen2.5-7B")
        self.logger.info("="*80)
        
        models = [
            ("Qwen/Qwen2.5-3B-Instruct", "3B Model"),
            ("Qwen/Qwen2.5-7B-Instruct", "7B Model")
        ]
        
        for model_name, model_label in models:
            try:
                self.run_single_model(model_name, model_label)
            except Exception as e:
                self.logger.error(f"Error running {model_label}: {e}")
                self.results[model_label] = {'error': str(e)}
        
        # Save comparison
        self._save_comparison()
    
    def run_experiments(self):
        """Required abstract method - calls run_comparison()"""
        self.run_comparison()
    
    def _save_comparison(self):
        """Save comparison results"""
        comparison_path = self.session_dir / "model_comparison.json"
        
        comparison_data = {
            'timestamp': datetime.now().isoformat(),
            'models_compared': list(self.results.keys()),
            'results': self.results,
            'summary': {
                '3B_turns': self.results.get('3B Model', {}).get('turn_count', 0),
                '7B_turns': self.results.get('7B Model', {}).get('turn_count', 0),
                '3B_code_blocks': self.results.get('3B Model', {}).get('code_blocks_executed', 0),
                '7B_code_blocks': self.results.get('7B Model', {}).get('code_blocks_executed', 0),
                '3B_errors': self.results.get('3B Model', {}).get('execution_errors', 0),
                '7B_errors': self.results.get('7B Model', {}).get('execution_errors', 0),
            }
        }
        
        with open(comparison_path, 'w', encoding='utf-8') as f:
            json.dump(comparison_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"\n[COMPARISON] Results saved to: {comparison_path}")
        
        # Print summary
        self.logger.info("\n[COMPARISON SUMMARY]")
        self.logger.info("="*80)
        for model_label, result in self.results.items():
            if 'error' in result:
                self.logger.info(f"{model_label}: ERROR - {result['error']}")
            else:
                self.logger.info(f"{model_label}:")
                self.logger.info(f"  Turns: {result.get('turn_count', 0)}")
                self.logger.info(f"  Code blocks: {result.get('code_blocks_executed', 0)}")
                self.logger.info(f"  Errors: {result.get('execution_errors', 0)}")
        self.logger.info("="*80)


def main():
    """Run model comparison"""
    print("\n" + "="*80)
    print("MODEL COMPARISON: Qwen2.5-3B vs Qwen2.5-7B")
    print("="*80)
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    experiment = ModelComparisonExperiment()
    
    try:
        experiment.run_experiments()
        
        print("\n" + "="*80)
        print("COMPARISON COMPLETE!")
        print("="*80)
        print(f"End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"\nResults saved to: {experiment.session_dir}")
        
    except KeyboardInterrupt:
        print("\n\n[INTERRUPTED] Experiment interrupted by user")
        experiment.logger.warning("Experiment interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n[ERROR] {e}")
        experiment.logger.error(f"Experiment failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
