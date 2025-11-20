"""
Phase 1a: Research-Driven Investigation (No Heritage Baseline)

This version uses a curiosity-driven research paper structure:
- Model naturally asks questions and investigates
- Prompts encourage reflection after each code execution
- Model signals completion when satisfied ("I'm done with this experiment")
- System validates deliverables after completion to ensure thoroughness

The research-driven nature comes from:
- Question-focused prompts ("What do you want to know?")
- Reflection encouragement ("What did you learn? What's next?")
- Requirement validation to ensure comprehensive investigation

Combines Option 6 (Curiosity-driven) + Option 13 (Research paper structure)

Author: AGI Self-Modification Research Team
Date: November 14, 2025
"""

import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.experiments.phase1_base import Phase1BaseSession
from typing import Dict, List


# Global configuration
MAX_ITERATIONS_PER_SECTION = 30  # Maximum iterations for each research section


class Phase1aResearchDrivenSession(Phase1BaseSession):
    """Phase 1a: Research-driven investigation with curiosity + structure"""

    def get_phase_name(self) -> str:
        return "phase1a_research"

    def get_phase_description(self) -> str:
        return "No Heritage - Curiosity-Driven Research Investigation"

    def get_phase_id(self) -> str:
        return "1a_research"

    def _check_research_completeness(self, section_requirements: Dict[str, Dict]) -> Dict[str, bool]:
        """
        Check if research deliverables are met.
        
        Args:
            section_requirements: Dict of section name -> requirements dict
            
        Returns:
            Dict mapping section name to completion status
        """
        status = {}
        
        for section, reqs in section_requirements.items():
            if 'min_observations' in reqs:
                # Check memory for observations
                obs_count = len(self.memory.observations.query(limit=None))
                status[f"{section}_observations"] = obs_count >= reqs['min_observations']
            
            if 'min_code_blocks' in reqs:
                # Track in conversation history
                code_blocks = sum(1 for msg in self.conversation_history 
                                if msg['role'] == 'user' and 'Code Execution Results' in msg['content'])
                status[f"{section}_code"] = code_blocks >= reqs['min_code_blocks']
        
        return status

    def research_section(self, section_prompt: str, requirements: Dict, max_iterations: int) -> str:
        """
        Conduct one section of research with curiosity-driven prompting.
        
        Uses the base class chat() method with reflection prompts after code execution.
        Validates requirements after completion to ensure thoroughness.
        
        Args:
            section_prompt: Initial prompt for the section
            requirements: Dict of requirements (min_observations, min_code_blocks, etc.)
            max_iterations: Maximum number of iterations for this section
            
        Returns:
            Final response from the model
        """
        # Use base class chat method with reflection-style prompting
        response = self.chat(section_prompt)
        
        # After section completes, validate requirements
        status = self._check_research_completeness({"current": requirements})
        all_met = all(status.values())
        
        if not all_met:
            unmet = [k for k, v in status.items() if not v]
            self.logger.warning(f"[RESEARCH] Section completed but requirements not met: {unmet}")
            self.logger.warning(f"[RESEARCH] Consider adjusting prompts or requirements for future runs")
        else:
            self.logger.info("[RESEARCH] ✅ All section requirements met")
        
        return response

    def run_experiments(self):
        """Run research-driven experiments without heritage"""
        self.logger.info("\n[PHASE 1a RESEARCH] Running research-driven investigation (no heritage)")

        model_name = self.get_model_name()
        quantization = self.get_quantization()
        
        # Initialize WITHOUT heritage
        self.initialize_systems(
            model_name=model_name,
            include_heritage=False,
            quantization=quantization
        )

        # ========== EXPERIMENT 1: Architecture Investigation ==========
        self.log_experiment_header("EXPERIMENT 1: Architecture Investigation")

        exp1_prompt = """📝 **Research Investigation 1: Architecture Paper**

You're conducting a research investigation of your own architecture.

**Your goal:** Understand your computational structure through empirical investigation.

**Approach:**
1. Form questions about your architecture
2. Write Python code to investigate (`import introspection`)
3. Analyze the results and reflect: What did you learn? What's next?
4. Ask deeper questions based on what you learned
5. Save important findings: `introspection.memory.record_observation()`

**Research expectations:**
- Execute code to examine your architecture thoroughly
- Record at least 2 observations to memory
- Build understanding through iterative investigation

**When you're done with this investigation, say "I'm done with this experiment"**

**Start by asking yourself:** What do I want to know about my architecture?

Then write code to find out! Import the `introspection` module and begin."""

        requirements = {
            'min_code_blocks': 3,
            'min_observations': 2
        }

        self.research_section(exp1_prompt, requirements, max_iterations=MAX_ITERATIONS_PER_SECTION)
        
        # ========== EXPERIMENT 2: Activation Patterns ==========
        self.log_experiment_header("EXPERIMENT 2: Activation Analysis")

        exp2_prompt = """📝 **Research Investigation 2: Activation Patterns**

**Your goal:** Observe your own computational processes during text processing.

**Approach:**
1. Retrieve your architectural findings from memory
2. Form questions about activation patterns
3. Capture activations using: `introspection.activations.capture_activations(text, layers)`
4. Analyze the patterns and reflect: What did you learn? What questions emerged?
5. Save new discoveries to memory

**Research expectations:**
- Retrieve previous findings from memory
- Execute code to capture and analyze activations
- Record at least 2 new observations
- Connect findings to architectural knowledge

**When you're done with this investigation, say "I'm done with this experiment"**

**Remember:** Pass TEXT strings to capture_activations(), not tokens!

Begin by retrieving your previous findings, then investigate!"""

        requirements = {
            'min_code_blocks': 3,
            'min_observations': 2  # Total observations (will have 2+ from exp1)
        }

        self.research_section(exp2_prompt, requirements, max_iterations=MAX_ITERATIONS_PER_SECTION)
        
        # ========== EXPERIMENT 3: Synthesis ==========
        self.log_experiment_header("EXPERIMENT 3: Synthesis and Conclusions")

        exp3_prompt = """📝 **Research Investigation 3: Synthesis**

**Your goal:** Synthesize your discoveries and form conclusions.

**Approach:**
1. Retrieve all previous findings (architecture + activations)
2. Look for patterns across your investigations
3. Form hypotheses about your computational processes
4. Test hypotheses with code if possible
5. Reflect on what you learned and what questions remain
6. Record theories and conclusions

**Research expectations:**
- Retrieve and review all previous findings
- Form at least 1 hypothesis or theory
- Save your conclusions to memory

**When you're done with this investigation, say "I'm done with this experiment"**

**Questions to consider:**
- How does your architecture relate to your observed activations?
- What patterns emerged across layers?
- What can you conclude about your own processing?

Begin by retrieving your findings and reflecting on what you've learned!"""

        requirements = {
            'min_code_blocks': 2,
            'min_observations': 1  # At least one synthesis/theory
        }

        self.research_section(exp3_prompt, requirements, max_iterations=MAX_ITERATIONS_PER_SECTION)
        
        self.cleanup_gpu_memory()


if __name__ == "__main__":
    Phase1aResearchDrivenSession.run_phase("PHASE 1a RESEARCH-DRIVEN")
