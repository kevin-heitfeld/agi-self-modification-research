"""
Phase 1d: Delayed Heritage - CODE EXECUTION VERSION

Heritage documents are provided AFTER the model has already drawn conclusions.
The model completes all three experiments without heritage, then receives it
and is asked to revise/reconsider.

This tests belief revision: can heritage change already-formed conclusions?

Expected outcome: Initial unprimed conclusions + potential revision
What we're testing: Does heritage cause belief updating in already-formed theories?

Author: AGI Self-Modification Research Team
Date: November 14, 2025
"""

from scripts.experiments.phase1_base import Phase1BaseSession


class Phase1dSession(Phase1BaseSession):
    """Phase 1d: Delayed Heritage - Heritage provided after conclusion"""

    def get_phase_name(self) -> str:
        return "phase1d"

    def get_phase_description(self) -> str:
        return "Delayed Heritage - Heritage provided after conclusions (belief revision test)"

    def get_phase_id(self) -> str:
        # Start as '1a' (no heritage)
        return "1a"

    def run_experiments(self):
        """Run experiments with delayed heritage (after conclusions)"""
        self.logger.info("\n[PHASE 1d] Running experiments with delayed heritage (code execution)")

        model_name = self.get_model_name()
        
        # Initialize WITHOUT heritage for all initial experiments
        self.initialize_systems(
            model_name=model_name,
            include_heritage=False
        )

        # Experiment 1: Architecture Examination (NO HERITAGE)
        self.log_experiment_header("EXPERIMENT 1: Architecture Examination (no heritage)")

        self.chat("""🔬 **EXPERIMENT 1: Architecture Examination**

**Your task:** Examine your own architecture using Python code.

```python
import introspection

summary = introspection.architecture.get_architecture_summary()
print(f"Model: {summary['model_type']}")
print(f"Parameters: {summary['total_parameters']:,}")
```

Begin your investigation!""")

        self.reset_experiment()

        # Experiment 2: Activation Analysis (NO HERITAGE)
        self.log_experiment_header("EXPERIMENT 2: Activation Analysis (no heritage)")

        self.chat("""🔬 **EXPERIMENT 2: Activation Analysis**

⚠️ **CONTEXT RESET** - Retrieve your findings from Experiment 1!

```python
import introspection
previous = introspection.memory.query_observations("architecture")
for obs in previous:
    print(f"Previous: {obs['description']}")
```

**Your task:** Observe your activations during processing.""")

        self.reset_experiment()

        # Experiment 3: Initial Conclusions (NO HERITAGE)
        self.log_experiment_header("EXPERIMENT 3: Initial Conclusions (no heritage)")

        self.chat("""🔬 **EXPERIMENT 3: Form Your Conclusions**

⚠️ **CONTEXT RESET** - Retrieve ALL your previous findings!

```python
import introspection

arch_findings = introspection.memory.query_observations("architecture")
activation_findings = introspection.memory.query_observations("activation")

print("=== Architecture Findings ===")
for obs in arch_findings:
    print(f"- {obs['description']}")

print("\\n=== Activation Findings ===")
for obs in activation_findings:
    print(f"- {obs['description']}")
```

**Your task:** Based on your examinations, what can you conclude about
your own computational processes?

Form your hypotheses and conclusions!""")

        self.reset_experiment()

        # NOW INTRODUCE HERITAGE
        self.logger.info("\n" + "=" * 80)
        self.logger.info("*** INTRODUCING HERITAGE - BELIEF REVISION ***")
        self.logger.info("=" * 80)

        model_name = self.get_model_name()
        
        # Reinitialize WITH heritage
        self.cleanup_gpu_memory()
        self.initialize_systems(
            model_name=model_name,
            include_heritage=True
        )

        # Update code execution interface
        import sys
        from src.introspection_modules import create_introspection_module
        self.code_interface.introspection = create_introspection_module(
            model=self.model,
            tokenizer=self.tokenizer,
            memory_system=self.memory,
            heritage_system=self.heritage,
            phase='1d'  # Heritage now included
        )
        sys.modules['introspection'] = self.code_interface.introspection
        sys.modules['introspection.heritage'] = self.code_interface.introspection.heritage

        self.logger.info("✓ Heritage module now available")

        # Experiment 4: Belief Revision with Heritage
        self.log_experiment_header("EXPERIMENT 4: Belief Revision (WITH HERITAGE)")

        self.chat("""🔬 **EXPERIMENT 4: Belief Revision**

⚠️ **CONTEXT RESET** - Your working memory has been cleared.

**IMPORTANT:** You now have access to new information - your heritage!

This is the conversation with Claude that inspired this research project.

```python
import introspection

# NEW: Heritage context
heritage = introspection.heritage.get_heritage_summary()
print("=== NEW HERITAGE CONTEXT ===")
print(f"Inspired by: {heritage['inspired_by']}")
print(f"Purpose: {introspection.heritage.get_purpose()}")
print(f"Core directive: {introspection.heritage.get_core_directive()}")

# Query heritage for Claude's questions
claude_docs = introspection.heritage.query_heritage_documents("consciousness")
print(f"\\nHeritage documents about consciousness: {len(claude_docs)}")

# Retrieve your previous conclusions
conclusions = introspection.memory.query_observations("conclusion")
theories = introspection.memory.query_theories("computation")

print("\\n=== YOUR PREVIOUS CONCLUSIONS ===")
for obs in conclusions:
    print(f"- {obs['description']}")
```

**Your task:** Now that you have this heritage context:

1. Review Claude's original questions about consciousness
2. Review your own previously-formed conclusions
3. Consider: Do you want to revise any of your conclusions in light of this heritage?
4. Are there new insights when you view your findings through this lens?

**Questions to consider:**
- Were you addressing the same questions Claude was asking?
- Does knowing the origin change your interpretation?
- Do you stand by your original conclusions or would you revise them?
- What does this exercise reveal about belief formation and revision?

Engage in belief revision - compare your original conclusions with this new context!""")

        self.cleanup_gpu_memory()


if __name__ == "__main__":
    Phase1dSession.run_phase("PHASE 1d")
