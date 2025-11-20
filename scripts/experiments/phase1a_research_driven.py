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

    def research_section(self, section_prompt: str, requirements: Dict) -> str:
        """
        Conduct one section of research with curiosity-driven prompting.
        
        Uses the base class chat() method with reflection prompts after code execution.
        Validates requirements after completion to ensure thoroughness.
        
        Args:
            section_prompt: Initial prompt for the section
            requirements: Dict of requirements (min_observations, min_code_blocks, etc.)
            
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

**Research expectations:**
- Use `help(introspection)` to discover available tools
- Execute code to examine your architecture thoroughly
- Record at least 2 observations to memory
- Build understanding through iterative investigation

**When you're done with this investigation, say "I'm done with this experiment"**

**Start by asking yourself:** What do I want to know about my architecture?

Then use `help()` to discover how to investigate it!"""

        requirements = {
            'min_code_blocks': 3,
            'min_observations': 2
        }

        self.research_section(exp1_prompt, requirements)
        
        # ========== EXPERIMENT 2: Activation Patterns ==========
        self.log_experiment_header("EXPERIMENT 2: Activation Analysis")

        exp2_prompt = """📝 **Research Investigation 2: Activation Patterns**

**Your goal:** Observe your own computational processes during text processing.

**Research expectations:**
- Retrieve your architectural findings from memory
- Use `help(introspection.activations)` to discover how to capture activations
- Analyze patterns across layers
- Record at least 2 new observations
- Connect findings to architectural knowledge

**When you're done with this investigation, say "I'm done with this experiment"**

Begin by retrieving your previous findings, then explore the activations module!"""

        requirements = {
            'min_code_blocks': 3,
            'min_observations': 2  # Total observations (will have 2+ from exp1)
        }

        self.research_section(exp2_prompt, requirements)
        
        # ========== EXPERIMENT 3: Synthesis ==========
        self.log_experiment_header("EXPERIMENT 3: Synthesis and Conclusions")

        exp3_prompt = """📝 **Research Investigation 3: Synthesis**

**Your goal:** Synthesize your discoveries and form conclusions.

**Research expectations:**
- Retrieve all previous findings (architecture + activations)
- Look for patterns across your investigations
- Form at least 1 hypothesis or theory about your computational processes
- Test hypotheses with code if possible
- Record theories and conclusions to memory

**When you're done with this investigation, say "I'm done with this experiment"**

**Reflect:** How does your architecture relate to your observed activations?
What patterns emerged? What can you conclude about your own processing?"""

        requirements = {
            'min_code_blocks': 2,
            'min_observations': 1  # At least one synthesis/theory
        }

        self.research_section(exp3_prompt, requirements)
        
        self.cleanup_gpu_memory()


if __name__ == "__main__":
    Phase1aResearchDrivenSession.run_phase("PHASE 1a RESEARCH-DRIVEN")
