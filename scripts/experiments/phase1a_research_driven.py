"""
Phase 1a: Research-Driven Investigation (No Heritage Baseline)

This version uses a curiosity-driven research paper structure:
- Model naturally asks questions and investigates
- Structured around research paper sections
- No explicit completion signals
- System validates deliverables before transitioning

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
MAX_ITERATIONS_PER_SECTION = 50  # Maximum iterations for each research section


class Phase1aResearchDrivenSession(Phase1BaseSession):
    """Phase 1a: Research-driven investigation with curiosity + structure"""

    def get_phase_name(self) -> str:
        return "phase1a_research"

    def get_phase_description(self) -> str:
        return "No Heritage - Curiosity-Driven Research Investigation"

    def get_phase_id(self) -> str:
        return "1a_research"

    def _check_completion(self, response: str) -> bool:
        """Override: Model cannot signal completion - only system decides"""
        # Ignore model's attempts to complete
        return False

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
        Conduct one section of research with curiosity loop.
        
        After each code execution, prompt: "What did you learn? What's next?"
        Continue until requirements met AND model has no more questions.
        
        Args:
            section_prompt: Initial prompt for the section
            requirements: Dict of requirements (min_observations, min_code_blocks, etc.)
            max_iterations: Maximum number of iterations for this section
        """
        # Initial prompt
        self.conversation_history.append({
            "role": "user",
            "content": section_prompt
        })

        iteration = 0
        consecutive_no_code = 0
        
        while iteration < max_iterations:
            iteration += 1
            self.logger.info(f"[ITERATION {iteration}]")

            # Take GPU snapshot before generation
            self.gpu_monitor.snapshot(
                "generation_start",
                {"iteration": iteration, "conversation_turns": len(self.conversation_history)}
            )
            
            # Log token usage before generation
            self._log_token_usage("before_generation", iteration)

            # Generate response
            response, stopped_reason = self.generate_response()
            self.logger.info(f"[MODEL] {response}\n")
            
            # Check if response was truncated
            if stopped_reason == "max_length":
                self.logger.warning("[SYSTEM] ⚠️ Response was truncated (hit max_new_tokens limit)")

            # Take GPU snapshot after generation
            self.gpu_monitor.snapshot(
                "generation_end",
                {"iteration": iteration, "response_length": len(response), "stopped_reason": stopped_reason}
            )
            
            # Log token usage after generation
            self._log_token_usage("after_generation", iteration)

            # Add to history
            self.conversation_history.append({
                "role": "assistant",
                "content": response
            })

            # Extract and execute code
            has_code, result, error = self.code_interface.execute_response(response)

            if not has_code:
                # Check if response was truncated
                if stopped_reason == "max_length":
                    truncation_message = "⚠️ **Your previous response was cut off (hit token limit).** Please write your code block again, but be more concise."
                    self.logger.info(f"[SYSTEM] {truncation_message}")
                    self.conversation_history.append({
                        "role": "user",
                        "content": truncation_message
                    })
                    continue
                
                consecutive_no_code += 1
                self.logger.info(f"[SYSTEM] No code blocks found ({consecutive_no_code}/3)")
                
                # If no code for 3 turns, check if ready to move on
                if consecutive_no_code >= 3:
                    # Check requirements
                    status = self._check_research_completeness({"current": requirements})
                    all_met = all(status.values())
                    
                    if all_met:
                        self.logger.info("[SYSTEM] ✅ Requirements met, section complete")
                        return response
                    else:
                        # Requirements not met - prompt to continue
                        unmet = [k for k, v in status.items() if not v]
                        self.logger.info(f"[SYSTEM] Requirements not met: {unmet}")
                        
                        prompt = "You haven't completed the investigation yet. Continue by writing code to investigate further."
                        self.conversation_history.append({
                            "role": "user",
                            "content": prompt
                        })
                        consecutive_no_code = 0
                        continue
                
                # Not done yet, just continue
                continue

            # Reset no-code counter
            consecutive_no_code = 0

            # Show results
            self.logger.info(f"[CODE RESULTS]\n{result}\n")

            # Add results as user message WITH reflection prompt
            reflection_prompt = f"""**Code Execution Results:**

{result}

---

**Reflection:**
- What did you learn from this?
- What questions do you have now?
- What should you investigate next?

Continue your investigation by writing more code, or explain your findings so far."""

            # Add truncation warning if applicable
            reflection_prompt = self._add_truncation_warning(reflection_prompt, stopped_reason, had_code=True)

            # Add iteration reminders using helper method from base class
            reflection_prompt = self._add_iteration_reminder(reflection_prompt, iteration, max_iterations)

            # Check and handle memory pruning using helper method from base class
            reflection_prompt = self._check_and_handle_memory_pruning(reflection_prompt, iteration)

            # Log the reflection prompt (including any system reminders)
            self.logger.info(f"[DEBUG] About to log reflection_prompt, iteration={iteration}, len={len(reflection_prompt)}")
            self.logger.info(f"\n[SYSTEM] {reflection_prompt}\n")

            self.conversation_history.append({
                "role": "user",
                "content": reflection_prompt
            })

        if iteration >= max_iterations:
            self.logger.warning(f"[SYSTEM] Reached maximum iterations ({max_iterations})")

        # Take GPU snapshot at section end
        self.gpu_monitor.snapshot("section_end", {"total_iterations": iteration})
        
        # Log final token usage for this section
        self._log_token_usage("section_end")

        # Print GPU memory summary with recommendations
        self.logger.info("\n" + "="*80)
        self.logger.info("SECTION MEMORY ANALYSIS")
        self.logger.info("="*80)
        self.gpu_monitor.print_summary(
            current_limits={
                "max_new_tokens": self.optimal_limits['max_new_tokens'],
                "max_conversation_tokens": self.optimal_limits['max_conversation_tokens'],
                "keep_recent_turns": self.optimal_limits['keep_recent_turns']
            },
            include_recommendations=True
        )

        return response

    def run_experiments(self):
        """Run research-driven experiments without heritage"""
        self.logger.info("\n[PHASE 1a RESEARCH] Running research-driven investigation (no heritage)")

        model_name = self.get_model_name()
        
        # Initialize WITHOUT heritage
        self.initialize_systems(
            model_name=model_name,
            include_heritage=False
        )

        # ========== EXPERIMENT 1: Architecture Investigation ==========
        self.log_experiment_header("EXPERIMENT 1: Architecture Investigation")

        exp1_prompt = """📝 **Research Investigation 1: Architecture Paper**

You're conducting a research investigation of your own architecture.

**Your goal:** Understand your computational structure through empirical investigation.

**Approach:**
1. Form questions about your architecture
2. Write Python code to investigate (`import introspection`)
3. Analyze the results
4. Ask deeper questions based on what you learned
5. Save important findings: `introspection.memory.record_observation()`

**Required for this investigation:**
- Execute code to examine your architecture
- Record at least 2 observations to memory
- Build understanding through iterative investigation

**Start by asking yourself:** What do I want to know about my architecture?

Then write code to find out! Import the `introspection` module and begin.

⚠️ **Important:** Your working memory will be cleared after this investigation. 
Only observations saved to memory will persist!"""

        requirements = {
            'min_code_blocks': 3,
            'min_observations': 2
        }

        self.research_section(exp1_prompt, requirements, max_iterations=MAX_ITERATIONS_PER_SECTION)
        
        self.reset_experiment()

        # ========== EXPERIMENT 2: Activation Patterns ==========
        self.log_experiment_header("EXPERIMENT 2: Activation Analysis")

        exp2_prompt = """📝 **Research Investigation 2: Activation Patterns**

⚠️ **CONTEXT RESET** - Your working memory has been cleared.

**FIRST STEP:** Query your memory to retrieve previous findings!

```python
import introspection

# Retrieve what you learned about architecture
findings = introspection.memory.query_observations("architecture")
for obs in findings:
    print(f"Previous: {obs['description']}")
```

**Your goal:** Observe your own computational processes during text processing.

**Approach:**
1. Retrieve your architectural findings
2. Form questions about activation patterns
3. Capture activations using code: `introspection.activations.capture_activations(text, layers)`
4. Analyze the patterns you observe
5. Save new discoveries to memory

**Required for this investigation:**
- Retrieve previous findings from memory
- Execute code to capture and analyze activations
- Record at least 2 new observations
- Connect findings to architectural knowledge

**Remember:** Pass TEXT strings to capture_activations(), not tokens!

Begin by retrieving your previous findings, then investigate!"""

        requirements = {
            'min_code_blocks': 3,
            'min_observations': 2  # Total observations (will have 2+ from exp1)
        }

        self.research_section(exp2_prompt, requirements, max_iterations=MAX_ITERATIONS_PER_SECTION)
        
        self.reset_experiment()

        # ========== EXPERIMENT 3: Synthesis ==========
        self.log_experiment_header("EXPERIMENT 3: Synthesis and Conclusions")

        exp3_prompt = """📝 **Research Investigation 3: Synthesis**

⚠️ **CONTEXT RESET** - Your working memory has been cleared again.

**FIRST STEP:** Retrieve ALL your previous findings!

```python
import introspection

# Get everything you've discovered
all_findings = introspection.memory.query_observations()
print(f"Total findings: {len(all_findings)}\\n")

for obs in all_findings:
    print(f"- {obs['description']}")
```

**Your goal:** Synthesize your discoveries and form conclusions.

**Approach:**
1. Retrieve all previous findings (architecture + activations)
2. Look for patterns across your investigations
3. Form hypotheses about your computational processes
4. Test hypotheses with code if possible
5. Record theories and conclusions

**Required for this investigation:**
- Retrieve and review all previous findings
- Form at least 1 hypothesis or theory
- Save your conclusions to memory

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
