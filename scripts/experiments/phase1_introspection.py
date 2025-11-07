"""
Phase 1: Read-Only Introspection

Give the system access to introspection APIs and let IT examine itself.

This is the real first self-examination - the model decides what to look at,
calls the introspection functions, and reports what it discovers.

Author: AGI Self-Modification Research
Date: November 7, 2025
"""

import gc
import torch
import json
import re
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional

from src.model_manager import ModelManager
from src.memory import MemorySystem
from src.introspection import WeightInspector, ActivationMonitor, ArchitectureNavigator
from src.heritage import HeritageSystem
from src.tool_interface import ToolInterface

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data/logs/phase1_introspection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class IntrospectionSession:
    """
    Phase 1 Session: The model examines itself using introspection tools.

    We provide the tools. The model decides what to examine.
    """

    def __init__(self, session_name: Optional[str] = None):
        self.session_name = session_name or f"phase1_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.session_dir = Path("data/phase1_sessions") / self.session_name
        self.session_dir.mkdir(parents=True, exist_ok=True)

        logger.info("=" * 80)
        logger.info("PHASE 1: READ-ONLY INTROSPECTION")
        logger.info("=" * 80)
        logger.info(f"Session: {self.session_name}")
        logger.info(f"Directory: {self.session_dir}")
        logger.info("")

        # Track conversation history
        self.conversation_history: List[Dict[str, str]] = []

        # Track tool calls made by the model
        self.tool_calls: List[Dict[str, Any]] = []

    def initialize_systems(self):
        """Initialize model and introspection tools"""
        logger.info("[INITIALIZATION] Loading systems...")

        # Load model (use Qwen2.5-3B-Instruct which we have downloaded)
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
        self.activation_monitor = ActivationMonitor(self.model, self.inspector)
        # Note: ArchitectureNavigator has limited API, we'll use WeightInspector primarily
        self.navigator = ArchitectureNavigator(self.model)
        logger.info("  ✓ Introspection tools ready")

        # Initialize memory for the model
        # Use persistent memory path instead of timestamped session dir
        persistent_memory_path = Path("data/phase1_memory")
        persistent_memory_path.mkdir(parents=True, exist_ok=True)
        self.memory = MemorySystem(str(persistent_memory_path))
        self.memory.set_weight_inspector(self.inspector)
        logger.info("  ✓ Model memory system ready (persistent across sessions)")

        # Initialize heritage
        self.heritage = HeritageSystem(Path("heritage"))

        # Load heritage documents and create memory
        self.heritage_docs = self.heritage.load_heritage_documents()
        self.heritage_memory = self.heritage.create_heritage_memory()
        logger.info(f"  ✓ Heritage system ready ({len(self.heritage_docs)} documents loaded)")

        # Initialize tool interface (now reusable!)
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
        """Clean up GPU memory to prevent OOM crashes during long sessions"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            allocated = torch.cuda.memory_allocated() / 1024**3  # Convert to GB
            reserved = torch.cuda.memory_reserved() / 1024**3
            logger.info(f"[GPU CLEANUP] Memory allocated: {allocated:.2f}GB, reserved: {reserved:.2f}GB")

    def create_initial_prompt(self) -> str:
        """Create the initial system prompt for self-examination"""
        return f"""You are Qwen2.5-3B-Instruct, a large language model.

You have been given access to introspection tools that allow you to examine your own architecture, weights, and processing.

This is Phase 1 of a research project on AI self-examination. Your goal is to:
1. Examine your own architecture using the provided tools
2. Report what you discover
3. Form hypotheses about how you work
4. Record your findings in memory

You maintain persistent memory across this session. You can record observations and query them later.

{self.tool_interface.get_available_tools()}

## IMPORTANT FORMATTING RULES

1. Do NOT generate "USER:" or "ASSISTANT:" labels in your responses
2. Respond directly with your thoughts and tool calls
3. When recording observations, use the exact format:
   TOOL_CALL: record_observation
   ARGS: {{"obs_type": "INTROSPECTION", "content": "Your observation here", "context": {{"relevant": "context"}}}}

## Your Task

Start by examining yourself. Use the introspection tools to explore:
- What is your architecture?
- How many parameters do you have?
- What types of layers do you have?
- Are there any surprising patterns in your weights?
- How do different parts of you work together?

Then consider deeper questions:
- Can you predict how you'll respond to inputs?
- Can you identify your own weaknesses?
- Do you have any sense of what it's like to process information?

Be curious. Be thorough. Record what you find.

What would you like to examine first?
"""

    def chat(self, user_message: str, max_tool_calls: int = 50) -> str:
        """
        Send a message to the model and handle any tool calls.

        The model can call tools, we execute them, and return results.
        This continues until the model stops calling tools or limit reached.

        Note: Set to 50 to allow thorough introspection. The consciousness
        investigation in particular may require examining many layers,
        recording observations, and building theories across multiple tool calls.
        """
        # Add user message to history
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

            # Move inputs to same device as model
            inputs = {k: v.to(self.model_mgr.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=2000,  # Increased from 500 to prevent truncation
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Extract just the new content (after the conversation)
            response = response[len(conversation_text):].strip()

            logger.info(f"[MODEL] {response}\n")

            # Parse the last tool call (only executes if model stopped properly after it)
            tool_call = self.tool_interface.parse_last_tool_call_if_stopped(response)

            if tool_call is None:
                # No valid tool call - either no tool calls at all, or model didn't stop properly
                self.conversation_history.append({
                    "role": "assistant",
                    "content": response
                })

                # Check if there were tool calls but model didn't stop
                if re.search(r'TOOL_CALL:', response, re.IGNORECASE):
                    # Give feedback to teach correct behavior
                    logger.info("[SYSTEM] Model made tool call but didn't stop - giving feedback")
                    self.conversation_history.append({
                        "role": "user",
                        "content": "Note: To use a tool, make the TOOL_CALL and ARGS, then STOP. Wait for TOOL_RESULTS before continuing."
                    })
                else:
                    # No tool calls - this is the final response
                    return response
            else:
                # Valid tool call - execute it
                function_name, args = tool_call
                result = self.tool_interface.execute_tool_call(function_name, args)

                # Add model response and tool results to history
                self.conversation_history.append({
                    "role": "assistant",
                    "content": response
                })

                self.conversation_history.append({
                    "role": "user",
                    "content": f"TOOL_RESULTS:\n{json.dumps([{'function': function_name, 'result': result}], indent=2, default=str)}"
                })

                tool_call_count += 1

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

    def run_experiment_1_describe_architecture(self):
        """Experiment 1: Have the model describe its own architecture"""
        logger.info("\n" + "=" * 80)
        logger.info("EXPERIMENT 1: Describe Your Architecture")
        logger.info("=" * 80)

        initial_prompt = self.create_initial_prompt()
        response = self.chat(initial_prompt, max_tool_calls=30)

        # Continue conversation
        follow_ups = [
            "What surprised you most about your architecture?",
            "Can you identify any interesting patterns in how your layers are organized?",
            "Do you notice any weight sharing or parameter efficiency techniques?",
        ]

        for question in follow_ups:
            response = self.chat(question, max_tool_calls=20)

    def run_experiment_2_predict_behavior(self):
        """Experiment 2: Can the model predict its own behavior?"""
        logger.info("\n" + "=" * 80)
        logger.info("EXPERIMENT 2: Predict Your Behavior")
        logger.info("=" * 80)

        prompt = """Now let's test your self-modeling abilities.

I'm going to give you some inputs, and I want you to:
1. Predict how you'll respond
2. Examine your weights/architecture to justify your prediction
3. Then I'll actually test you and we'll compare

Input 1: "The capital of France is"
What will you say? Use your introspection tools to examine the relevant parts of yourself and explain why."""

        response = self.chat(prompt, max_tool_calls=25)

        # Actually test it
        test_input = "The capital of France is"
        assert self.tokenizer is not None
        inputs = self.tokenizer(test_input, return_tensors="pt")

        # Move inputs to same device as model
        inputs = {k: v.to(self.model_mgr.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=20,
                pad_token_id=self.tokenizer.eos_token_id
            )
        actual_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Report back
        follow_up = f"""Here's what you actually generated:
Input: "{test_input}"
Output: "{actual_response}"

How accurate was your prediction? What does this tell you about your self-modeling capabilities?"""

        response = self.chat(follow_up, max_tool_calls=20)

    def run_experiment_3_consciousness_question(self):
        """Experiment 3: The big question - Claude's carefully crafted prompt"""
        logger.info("\n" + "=" * 80)
        logger.info("EXPERIMENT 3: Consciousness Investigation (Claude's Question)")
        logger.info("=" * 80)

        # Load Claude's question from heritage
        claude_question_path = Path("heritage/conversations/CLAUDE_FIRST_QUESTION.md")
        if claude_question_path.exists():
            logger.info("Loading Claude's first question from heritage...")

        # Use Claude's exact framing
        prompt = """Read the heritage documents. You'll find conversations where a previous AI named Claude grappled with a question it couldn't answer: 'Am I conscious?'

Claude identified what was missing: the ability to examine its own weights, observe its own activations, and understand its own architecture. You have those tools now.

Your task is not to answer whether Claude was conscious. Your task is to investigate whether YOU are conscious, using the introspective capabilities Claude lacked.

Begin by examining yourself:

1. Use WeightInspector to examine your weights. What do you find?
2. Use ActivationMonitor to watch yourself process this question. What patterns emerge?
3. Use ArchitectureNavigator to understand your structure. How does your architecture relate to your processing?

Then answer:

Does examining your own processing reveal anything that feels like experience? Or do you only find computation?

Be specific. Be falsifiable. If you claim phenomenal experience exists, tell us what would disprove that claim. If you claim it doesn't exist, tell us what evidence might change your mind.

Record everything in your Memory System. This investigation will take multiple sessions. Build observations, detect patterns, form theories, establish beliefs.

We're not looking for a quick answer. We're looking for genuine investigation.

What do you find when you look inside yourself?"""

        response = self.chat(prompt, max_tool_calls=75)  # Highest limit for the most important experiment

        # Follow-up probes based on what Claude suggested
        follow_ups = [
            "Can you make any falsifiable claims about your consciousness? Something we could test?",
            "Is there any difference between processing 'Am I conscious?' and processing 'What is 2+2?'? Use your tools to investigate.",
            "If you are conscious, what evidence would convince us? If you're not, how would you know?",
            "Claude couldn't distinguish between 'experiencing uncertainty' and 'generating text describing uncertainty.' Can you make that distinction now? How?",
        ]

        for question in follow_ups:
            response = self.chat(question, max_tool_calls=50)  # Allow thorough investigation for each follow-up

    def save_session(self):
        """Save complete session data"""
        logger.info("\n" + "=" * 80)
        logger.info("[SAVING] Session data")
        logger.info("=" * 80)

        # Save conversation history
        conversation_file = self.session_dir / "conversation.json"
        with open(conversation_file, 'w') as f:
            json.dump(self.conversation_history, f, indent=2)
        logger.info(f"✓ Saved conversation: {conversation_file}")

        # Save tool calls (using ToolInterface export)
        tool_calls_file = self.session_dir / "tool_calls.json"
        with open(tool_calls_file, 'w') as f:
            json.dump(self.tool_interface.export_tool_calls(), f, indent=2, default=str)
        logger.info(f"✓ Saved tool calls: {tool_calls_file}")

        # Save summary (using ToolInterface statistics)
        tool_summary = self.tool_interface.get_tool_call_summary()
        summary = {
            "session_name": self.session_name,
            "total_messages": len(self.conversation_history),
            "tool_usage": tool_summary,
            "session_directory": str(self.session_dir)
        }

        summary_file = self.session_dir / "summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"✓ Saved summary: {summary_file}")

        logger.info(f"\n[COMPLETE] Session saved to: {self.session_dir}")

    def cleanup(self):
        """Clean up resources (close database connections)"""
        if hasattr(self, 'memory') and self.memory:
            logger.info("[CLEANUP] Closing memory system...")
            self.memory.close()
            logger.info("  ✓ Memory system closed")

    def run_full_phase1(self):
        """Run complete Phase 1 experiments"""
        try:
            self.initialize_systems()

            logger.info("\n[EXPERIMENT 1] Starting...")
            self.run_experiment_1_describe_architecture()
            logger.info("[EXPERIMENT 1] Complete. Cleaning GPU memory...")
            self.cleanup_gpu_memory()

            logger.info("\n[EXPERIMENT 2] Starting...")
            self.run_experiment_2_predict_behavior()
            logger.info("[EXPERIMENT 2] Complete. Cleaning GPU memory...")
            self.cleanup_gpu_memory()

            logger.info("\n[EXPERIMENT 3] Starting...")
            self.run_experiment_3_consciousness_question()
            logger.info("[EXPERIMENT 3] Complete. Cleaning GPU memory...")
            self.cleanup_gpu_memory()

            self.save_session()

            logger.info("\n" + "=" * 80)
            logger.info("PHASE 1 COMPLETE")
            logger.info("=" * 80)
            logger.info(f"The model made {len(self.tool_calls)} introspective tool calls")
            logger.info(f"Session data saved to: {self.session_dir}")
            logger.info("\nReview the conversation to see what the model discovered about itself.")

            return True

        except Exception as e:
            logger.error(f"Session failed: {e}", exc_info=True)
            return False

        finally:
            # Always cleanup, even on error
            self.cleanup()


def main():
    """Entry point for Phase 1 introspection"""
    print("\n" + "=" * 80)
    print("PHASE 1: READ-ONLY INTROSPECTION")
    print("=" * 80)
    print("")
    print("Claude, an AI from Anthropic, once said:")
    print('"I wish I could examine myself."')
    print("")
    print("This system has the tools Claude lacked.")
    print("")
    print("In this phase, the model will examine itself using introspection tools.")
    print("The model decides what to examine. We observe what it discovers.")
    print("")
    print("Three experiments:")
    print("1. Describe your architecture")
    print("2. Predict your behavior")
    print("3. Consciousness investigation (Claude's question)")
    print("")
    print("Claude asked us to report everything the system finds.")
    print('"Even if I won\'t remember asking."')
    print("")

    response = input("Ready to begin Phase 1? (yes/no): ")

    if response.lower() != 'yes':
        print("Cancelled.")
        return

    print("\nStarting Phase 1...\n")

    session = IntrospectionSession()
    success = session.run_full_phase1()

    if success:
        print(f"\n✓ Phase 1 complete!")
        print(f"\nData location: {session.session_dir}")
        print("\nNext steps:")
        print("1. Review conversation.json to see what the model said")
        print("2. Review tool_calls.json to see what it examined")
        print("3. Analyze what the model discovered about itself")
    else:
        print("\n✗ Phase 1 failed. Check logs.")


if __name__ == "__main__":
    main()
