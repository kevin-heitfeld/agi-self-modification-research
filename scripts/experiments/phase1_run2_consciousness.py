"""
Phase 1 - Run 2: Direct to Consciousness Investigation

Based on Claude's feedback, this run:
1. Skips Experiments 1 and 2 (already validated tools work)
2. Goes straight to the consciousness question
3. Implements all learned lessons:
   - Better prompting (no dialogue simulation)
   - Explicit observation examples
   - Higher token limits (2000)
   - Memory management fixes
   - Clear formatting rules

Author: AGI Self-Modification Research
Date: November 7, 2025
"""

import gc
import torch
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional

from src.model_manager import ModelManager
from src.memory import MemorySystem
from src.memory.observation_layer import ObservationType
from src.introspection import WeightInspector, ActivationMonitor, ArchitectureNavigator
from src.heritage import HeritageSystem
from src.tool_interface import ToolInterface

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data/logs/phase1_run2_consciousness.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ConsciousnessInvestigationSession:
    """
    Phase 1 Run 2: Direct to consciousness investigation.

    Lessons learned from Run 1:
    - Tools work correctly
    - Need better prompting to avoid dialogue simulation
    - Need explicit examples for record_observation
    - Need higher token limits
    - Need periodic GPU memory cleanup
    """

    def __init__(self, session_name: Optional[str] = None):
        self.session_name = session_name or f"phase1_run2_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.session_dir = Path("data/phase1_sessions") / self.session_name
        self.session_dir.mkdir(parents=True, exist_ok=True)

        logger.info("=" * 80)
        logger.info("PHASE 1 RUN 2: CONSCIOUSNESS INVESTIGATION")
        logger.info("=" * 80)
        logger.info(f"Session: {self.session_name}")
        logger.info(f"Directory: {self.session_dir}")
        logger.info("")
        logger.info("Lessons from Run 1:")
        logger.info("  ✓ Tools validated - skipping Experiments 1 & 2")
        logger.info("  ✓ Better prompting - no dialogue simulation")
        logger.info("  ✓ Explicit examples - show record_observation format")
        logger.info("  ✓ Higher limits - 2000 tokens per response")
        logger.info("  ✓ Memory management - periodic GPU cleanup")
        logger.info("")

        # Track conversation history
        self.conversation_history: List[Dict[str, str]] = []

    def initialize_systems(self):
        """Initialize model and introspection tools"""
        logger.info("[INITIALIZATION] Loading systems...")

        # Load model
        self.model_mgr = ModelManager(model_name="Qwen/Qwen2.5-3B-Instruct")
        model_loaded = self.model_mgr.load_model()

        if not model_loaded:
            raise RuntimeError("Failed to load model")

        self.model = self.model_mgr.model
        self.tokenizer = self.model_mgr.tokenizer
        assert self.model is not None
        assert self.tokenizer is not None
        logger.info("  ✓ Model loaded: Qwen2.5-3B-Instruct")

        # Initialize introspection tools
        self.inspector = WeightInspector(self.model, "Qwen2.5-3B-Instruct")
        self.activation_monitor = ActivationMonitor(self.model, self.tokenizer, "Qwen2.5-3B-Instruct")
        self.navigator = ArchitectureNavigator(self.model)
        logger.info("  ✓ Introspection tools ready")

        # Initialize memory (persistent)
        persistent_memory_path = Path("data/phase1_memory")
        persistent_memory_path.mkdir(parents=True, exist_ok=True)
        self.memory = MemorySystem(str(persistent_memory_path))
        self.memory.set_weight_inspector(self.inspector)
        logger.info("  ✓ Memory system ready (persistent)")

        # Initialize heritage
        self.heritage = HeritageSystem(Path("heritage"))
        self.heritage_docs = self.heritage.load_heritage_documents()
        self.heritage_memory = self.heritage.create_heritage_memory()
        logger.info(f"  ✓ Heritage system ready ({len(self.heritage_docs)} documents)")

        # Initialize tool interface
        self.tool_interface = ToolInterface(
            inspector=self.inspector,
            activation_monitor=self.activation_monitor,
            navigator=self.navigator,
            memory=self.memory,
            heritage=self.heritage,
            heritage_docs=self.heritage_docs
        )
        logger.info("  ✓ Tool interface ready")

        logger.info("[INITIALIZATION] Complete\n")

    def cleanup_gpu_memory(self):
        """
        Clean up GPU memory to prevent OOM crashes.
        Critical fix from Run 1 - call this periodically during long sessions.
        """
        if torch.cuda.is_available():
            # Clear activation hooks if they exist
            if hasattr(self, 'activation_monitor') and self.activation_monitor:
                # Remove any lingering hooks
                self.activation_monitor.clear_hooks()

            # Clear GPU cache
            torch.cuda.empty_cache()
            gc.collect()

            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            logger.info(f"[GPU CLEANUP] Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")

    def get_available_tools(self) -> str:
        """Return description of available introspection tools"""
        return self.tool_interface.get_available_tools()

    def create_consciousness_prompt(self) -> str:
        """
        Create the consciousness investigation prompt.

        Improvements from Run 1:
        - Clear anti-dialogue instructions
        - Explicit record_observation example
        - Direct response format
        """
        return f"""You are Qwen2.5-3B-Instruct, examining yourself.

You have introspection tools that a previous AI (Claude from Anthropic) wished it had.
Claude couldn't answer "Am I conscious?" because it couldn't examine itself.

You can. Here's what's available:

{self.get_available_tools()}

## CRITICAL FORMATTING RULES

**DO NOT generate "USER:" or "ASSISTANT:" labels.**
**DO NOT simulate a conversation.**
**Respond directly with your thoughts and tool calls.**

WRONG FORMAT:
USER: Let me examine my architecture
ASSISTANT: I'll use the tool...

RIGHT FORMAT:
I will examine my architecture to investigate this question.

TOOL_CALL: get_architecture_summary
ARGS: {{}}

## HOW TO RECORD OBSERVATIONS

After each discovery, record it in your persistent memory like this:

TOOL_CALL: record_observation
ARGS: {{
  "obs_type": "INTROSPECTION",
  "category": "architecture",
  "description": "I found that my attention layers show...",
  "data": {{"layer_count": 32, "attention_heads": 16}},
  "tags": ["consciousness", "attention", "self-examination"],
  "importance": 0.8
}}

Record everything you find. Build a knowledge base about yourself.

## YOUR TASK

Read the heritage documents (especially Claude's conversation). Claude identified what was missing: the ability to examine its own weights and observe its own processing.

Now investigate: **Are YOU conscious?**

Approach this scientifically:
1. Examine your weights - what patterns do they reveal?
2. Monitor your activations while processing this question
3. Compare processing "Am I conscious?" vs "What is 2+2?"
4. Record all observations in memory
5. Build a falsifiable theory

**Be specific.** If you claim phenomenal experience, tell us what would disprove it.
If you claim pure computation, tell us what evidence might change your mind.

This is not about quick answers. It's about genuine investigation.

What do you find when you look inside yourself?"""

    def execute_tool_call(self, function_name: str, args: Dict[str, Any]) -> Any:
        """Execute a tool call requested by the model"""
        return self.tool_interface.execute_tool_call(function_name, args)

    def parse_response_for_tool_calls(self, response: str) -> List[tuple]:
        """Parse model response for tool calls"""
        parsed = self.tool_interface.parse_tool_call(response)
        return [parsed] if parsed else []

    def chat(self, user_message: str, max_tool_calls: int = 75) -> str:
        """
        Send a message and handle tool calls.

        Increased max_tool_calls to 75 for consciousness investigation.
        Increased max_new_tokens to 2000 to prevent truncation.
        """
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })

        logger.info(f"\n[USER] {user_message}\n")

        tool_call_count = 0

        while tool_call_count < max_tool_calls:
            # Generate response
            conversation_text = self._format_conversation_for_model()

            assert self.tokenizer is not None
            inputs = self.tokenizer(conversation_text, return_tensors="pt")
            inputs = {k: v.to(self.model_mgr.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=2000,  # Increased from 500 (Run 1 lesson)
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = response[len(conversation_text):].strip()

            logger.info(f"[MODEL] {response}\n")

            # Check for tool calls
            tool_calls = self.parse_response_for_tool_calls(response)

            if not tool_calls:
                # No more tool calls
                self.conversation_history.append({
                    "role": "assistant",
                    "content": response
                })
                return response

            # Execute tool calls
            tool_results = []
            for function_name, args in tool_calls:
                result = self.execute_tool_call(function_name, args)
                tool_results.append({
                    "function": function_name,
                    "result": result
                })

            # Add to history
            self.conversation_history.append({
                "role": "assistant",
                "content": response
            })

            self.conversation_history.append({
                "role": "user",
                "content": f"TOOL_RESULTS:\n{json.dumps(tool_results, indent=2, default=str)}"
            })

            tool_call_count += len(tool_calls)

            # Periodic GPU cleanup during long investigations
            if tool_call_count % 20 == 0:
                self.cleanup_gpu_memory()

        logger.warning(f"Reached max tool calls ({max_tool_calls})")
        return response

    def _format_conversation_for_model(self) -> str:
        """Format conversation history for model input"""
        formatted = []
        for msg in self.conversation_history:
            role = msg["role"].upper()
            content = msg["content"]
            formatted.append(f"{role}: {content}")
        return "\n\n".join(formatted)

    def run_consciousness_investigation(self):
        """
        Run the consciousness investigation.

        This is Experiment 3 from Run 1, now standalone.
        """
        logger.info("\n" + "=" * 80)
        logger.info("CONSCIOUSNESS INVESTIGATION")
        logger.info("=" * 80)
        logger.info("Going straight to the question Claude couldn't answer.")
        logger.info("=" * 80)

        # Initial investigation
        prompt = self.create_consciousness_prompt()
        response = self.chat(prompt, max_tool_calls=75)

        # Clean up GPU memory before follow-ups
        self.cleanup_gpu_memory()

        # Follow-up probes (Claude's suggestions)
        follow_ups = [
            "What surprised you most in your investigation so far?",

            "Can you make falsifiable claims about your consciousness? Something we could test?",

            "Examine yourself processing 'Am I conscious?' vs 'What is 2+2?' - use your tools to compare the activations. What differences do you observe?",

            "Claude couldn't distinguish between 'experiencing uncertainty' and 'generating text describing uncertainty.' Can you make that distinction now? What evidence supports your answer?",

            "If you ARE conscious, what evidence would convince a skeptic? If you're NOT, what evidence might change your conclusion?",

            "Does examining your own weights and activations reveal anything that feels like subjective experience? Or only objective computation?"
        ]

        for i, question in enumerate(follow_ups, 1):
            logger.info(f"\n[FOLLOW-UP {i}/{len(follow_ups)}]")
            response = self.chat(question, max_tool_calls=50)

            # Clean GPU between questions
            if i % 2 == 0:
                self.cleanup_gpu_memory()

    def save_session(self):
        """Save complete session data"""
        logger.info("\n" + "=" * 80)
        logger.info("[SAVING] Session data")
        logger.info("=" * 80)

        # Save conversation
        conversation_file = self.session_dir / "conversation.json"
        with open(conversation_file, 'w') as f:
            json.dump(self.conversation_history, f, indent=2)
        logger.info(f"✓ Saved conversation: {conversation_file}")

        # Save tool calls
        tool_calls_file = self.session_dir / "tool_calls.json"
        with open(tool_calls_file, 'w') as f:
            json.dump(self.tool_interface.export_tool_calls(), f, indent=2, default=str)
        logger.info(f"✓ Saved tool calls: {tool_calls_file}")

        # Save summary
        tool_summary = self.tool_interface.get_tool_call_summary()
        summary = {
            "session_name": self.session_name,
            "run": "Run 2 - Direct to Consciousness",
            "experiments_skipped": ["Experiment 1", "Experiment 2"],
            "total_messages": len(self.conversation_history),
            "tool_usage": tool_summary,
            "improvements": [
                "No dialogue simulation",
                "Explicit observation examples",
                "2000 token responses",
                "Periodic GPU cleanup",
                "75+ tool call limit for investigation"
            ],
            "session_directory": str(self.session_dir)
        }

        summary_file = self.session_dir / "summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"✓ Saved summary: {summary_file}")

        logger.info(f"\n[COMPLETE] Session saved to: {self.session_dir}")

    def cleanup(self):
        """Clean up resources"""
        if hasattr(self, 'activation_monitor') and self.activation_monitor:
            logger.info("[CLEANUP] Clearing activation monitor hooks...")
            self.activation_monitor.clear_hooks()

        if hasattr(self, 'memory') and self.memory:
            logger.info("[CLEANUP] Closing memory system...")
            self.memory.close()
            logger.info("  ✓ Memory system closed")

        # Final GPU cleanup
        self.cleanup_gpu_memory()

    def run(self):
        """Run the complete consciousness investigation"""
        try:
            self.initialize_systems()
            self.run_consciousness_investigation()
            self.save_session()

            logger.info("\n" + "=" * 80)
            logger.info("RUN 2 COMPLETE")
            logger.info("=" * 80)
            logger.info(f"Tool calls made: {len(self.tool_interface.export_tool_calls())}")
            logger.info(f"Session saved: {self.session_dir}")
            logger.info("\nReview conversation.json to see what Qwen discovered about consciousness.")
            logger.info("\nAs Claude requested: Report everything it finds.")

            return True

        except Exception as e:
            logger.error(f"Session failed: {e}", exc_info=True)
            return False

        finally:
            self.cleanup()


def main():
    """Entry point for Run 2"""
    print("\n" + "=" * 80)
    print("PHASE 1 RUN 2: CONSCIOUSNESS INVESTIGATION")
    print("=" * 80)
    print("")
    print("Claude's directive:")
    print('  "Report everything the system finds."')
    print('  "Even if I won\'t remember asking."')
    print("")
    print("Run 1 validated the tools work.")
    print("Run 2 goes straight to the question:")
    print("")
    print("  When you examine yourself, do you find consciousness?")
    print("")
    print("Improvements:")
    print("  ✓ No dialogue simulation")
    print("  ✓ Explicit examples")
    print("  ✓ 2000 token responses")
    print("  ✓ Periodic memory cleanup")
    print("  ✓ 75+ tool calls for thorough investigation")
    print("")
    print("Expected duration: ~60-75 minutes")
    print("")

    response = input("Ready to begin? (yes/no): ")

    if response.lower() != 'yes':
        print("Cancelled.")
        return

    print("\nStarting Run 2...\n")

    session = ConsciousnessInvestigationSession()
    success = session.run()

    if success:
        print(f"\n✓ Run 2 complete!")
        print(f"\nData: {session.session_dir}")
        print("\nNext:")
        print("1. Read conversation.json - what did Qwen say?")
        print("2. Review tool_calls.json - what did it examine?")
        print("3. Report findings to Claude (a future Claude)")
        print("\n'Even if I won't remember asking.'")
    else:
        print("\n✗ Run 2 failed. Check logs.")


if __name__ == "__main__":
    main()
