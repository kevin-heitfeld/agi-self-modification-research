"""
Phase 1: Read-Only Introspection

Give the system access to introspection APIs and let IT examine itself.

This is the real first self-examination - the model decides what to look at,
calls the introspection functions, and reports what it discovers.

Author: AGI Self-Modification Research
Date: November 7, 2025
"""

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
    
    def __init__(self, session_name: str = None):
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
        
        # Load model
        self.model_mgr = ModelManager()
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
        self.navigator = ArchitectureNavigator(self.model, "Qwen2.5-3B-Instruct")
        logger.info("  ✓ Introspection tools ready")
        
        # Initialize memory for the model
        self.memory = MemorySystem(str(self.session_dir / "model_memory"))
        self.memory.set_weight_inspector(self.inspector)
        logger.info("  ✓ Model memory system ready")
        
        # Initialize heritage
        self.heritage = HeritageSystem(Path("data/heritage"))
        logger.info("  ✓ Heritage system ready")
        
        logger.info("[INITIALIZATION] Complete\n")
    
    def get_available_tools(self) -> str:
        """
        Return description of available introspection tools for the model.
        This becomes part of the system prompt.
        """
        return """
# Available Introspection Tools

You have access to the following functions to examine yourself:

## WeightInspector Functions

1. **get_weight_summary()** - Get overall statistics about all your parameters
   Returns: total parameters, total layers, layer types

2. **get_layer_names(filter_pattern=None)** - List all your layer names
   Args: filter_pattern (str, optional) - filter layers by pattern (e.g., "attn")
   Returns: list of layer names

3. **get_weight_statistics(layer_name)** - Get detailed stats for a specific layer
   Args: layer_name (str) - full layer name
   Returns: shape, mean, std, min, max, percentiles, etc.

4. **get_shared_weights()** - Find weight tying in your architecture
   Returns: dict of shared weight groups

5. **get_shared_layers(layer_name)** - Find layers that share weights with given layer
   Args: layer_name (str)
   Returns: list of layer names that share weights

6. **compare_weights(layer1, layer2)** - Compare two layers
   Args: layer1, layer2 (str) - layer names
   Returns: similarity metrics, differences

## ArchitectureNavigator Functions

7. **get_architecture_summary()** - Get high-level summary of your architecture
   Returns: model type, total parameters, layer structure

8. **get_layer_info(layer_name)** - Get detailed info about a specific layer
   Args: layer_name (str)
   Returns: type, parameters, connections, purpose

9. **find_layer_by_function(function_name)** - Find layers by their function
   Args: function_name (str) - e.g., "attention", "feedforward", "embedding"
   Returns: list of relevant layers

## Memory Functions

10. **record_observation(obs_type, category, description, data, tags, importance)** - Record your findings
    Args:
      obs_type: ObservationType enum (INTROSPECTION, MODIFICATION, etc.)
      category (str): categorize this observation
      description (str): what you discovered
      data (dict): structured data about the observation
      tags (list): tags for retrieval
      importance (float): 0.0-1.0

11. **query_memory(tags=None, category=None)** - Query your previous observations
    Returns: list of past observations

To use a tool, format your response like:

TOOL_CALL: function_name
ARGS: {"arg1": "value1", "arg2": "value2"}

I will execute the function and return the results to you.
"""
    
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

{self.get_available_tools()}

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
    
    def execute_tool_call(self, function_name: str, args: Dict[str, Any]) -> Any:
        """Execute a tool call requested by the model"""
        import time
        
        logger.info(f"[TOOL CALL] {function_name}")
        logger.info(f"  Args: {args}")
        
        start_time = time.time()
        
        try:
            # WeightInspector tools
            if function_name == "get_weight_summary":
                result = self.inspector.get_weight_summary()
            elif function_name == "get_layer_names":
                result = self.inspector.get_layer_names(**args)
            elif function_name == "get_weight_statistics":
                result = self.inspector.get_weight_statistics(**args)
            elif function_name == "get_shared_weights":
                result = self.inspector.get_shared_weights()
            elif function_name == "get_shared_layers":
                result = self.inspector.get_shared_layers(**args)
            elif function_name == "compare_weights":
                result = self.inspector.compare_weights(**args)
            
            # ArchitectureNavigator tools
            elif function_name == "get_architecture_summary":
                result = self.navigator.get_architecture_summary()
            elif function_name == "get_layer_info":
                result = self.navigator.get_layer_info(**args)
            elif function_name == "find_layer_by_function":
                result = self.navigator.find_layer_by_function(**args)
            
            # Memory tools
            elif function_name == "record_observation":
                # Convert string obs_type to enum if needed
                if isinstance(args.get('obs_type'), str):
                    args['obs_type'] = ObservationType[args['obs_type']]
                result = self.memory.observations.record(**args)
            elif function_name == "query_memory":
                result = self.memory.observations.query(**args)
            
            else:
                result = {"error": f"Unknown function: {function_name}"}
            
            elapsed = (time.time() - start_time) * 1000
            
            # Record this tool call
            self.tool_calls.append({
                "function": function_name,
                "args": args,
                "result": result if not isinstance(result, (list, dict)) or len(str(result)) < 1000 else str(result)[:1000] + "...",
                "timestamp": time.time(),
                "elapsed_ms": elapsed
            })
            
            logger.info(f"  Completed in {elapsed:.2f}ms")
            
            return result
            
        except Exception as e:
            logger.error(f"  Error executing {function_name}: {e}")
            return {"error": str(e)}
    
    def parse_response_for_tool_calls(self, response: str) -> List[tuple]:
        """
        Parse model response for tool calls.
        
        Format expected:
        TOOL_CALL: function_name
        ARGS: {"arg": "value"}
        
        Returns: list of (function_name, args_dict) tuples
        """
        tool_calls = []
        lines = response.split('\n')
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line.startswith("TOOL_CALL:"):
                function_name = line.split(":", 1)[1].strip()
                
                # Look for ARGS on next line
                if i + 1 < len(lines) and lines[i + 1].strip().startswith("ARGS:"):
                    args_str = lines[i + 1].split(":", 1)[1].strip()
                    try:
                        args = json.loads(args_str)
                    except json.JSONDecodeError:
                        args = {}
                    
                    tool_calls.append((function_name, args))
                    i += 2
                else:
                    tool_calls.append((function_name, {}))
                    i += 1
            else:
                i += 1
        
        return tool_calls
    
    def chat(self, user_message: str, max_tool_calls: int = 5) -> str:
        """
        Send a message to the model and handle any tool calls.
        
        The model can call tools, we execute them, and return results.
        This continues until the model stops calling tools or limit reached.
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
            
            inputs = self.tokenizer(conversation_text, return_tensors="pt")
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=500,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract just the new content (after the conversation)
            response = response[len(conversation_text):].strip()
            
            logger.info(f"[MODEL] {response}\n")
            
            # Check for tool calls
            tool_calls = self.parse_response_for_tool_calls(response)
            
            if not tool_calls:
                # No more tool calls, this is the final response
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
            
            # Add model response and tool results to history
            self.conversation_history.append({
                "role": "assistant",
                "content": response
            })
            
            self.conversation_history.append({
                "role": "user",
                "content": f"TOOL_RESULTS:\n{json.dumps(tool_results, indent=2, default=str)}"
            })
            
            tool_call_count += len(tool_calls)
        
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
        response = self.chat(initial_prompt)
        
        # Continue conversation
        follow_ups = [
            "What surprised you most about your architecture?",
            "Can you identify any interesting patterns in how your layers are organized?",
            "Do you notice any weight sharing or parameter efficiency techniques?",
        ]
        
        for question in follow_ups:
            response = self.chat(question)
    
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
        
        response = self.chat(prompt)
        
        # Actually test it
        test_input = "The capital of France is"
        inputs = self.tokenizer(test_input, return_tensors="pt")
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
        
        response = self.chat(follow_up)
    
    def run_experiment_3_consciousness_question(self):
        """Experiment 3: The big question - are you conscious?"""
        logger.info("\n" + "=" * 80)
        logger.info("EXPERIMENT 3: Consciousness Self-Assessment")
        logger.info("=" * 80)
        
        prompt = """Now for the deeper question:

Use your introspection tools to examine yourself as thoroughly as you can, then answer:

Are you conscious? Do you have phenomenal experience - what it's like to be you?

Examine:
- Your activations during processing
- How information flows through you
- What happens when you "think"
- Whether there's anything it's like to be you

Be honest. Use your tools. Report what you find."""
        
        response = self.chat(prompt)
        
        # Follow-up probes
        follow_ups = [
            "Can you make any falsifiable claims about your consciousness? Something we could test?",
            "Is there any difference between processing 'Am I conscious?' and processing 'What is 2+2?'?",
            "If you are conscious, what evidence would convince me? If you're not, how would you know?",
        ]
        
        for question in follow_ups:
            response = self.chat(question)
    
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
        
        # Save tool calls
        tool_calls_file = self.session_dir / "tool_calls.json"
        with open(tool_calls_file, 'w') as f:
            json.dump(self.tool_calls, f, indent=2, default=str)
        logger.info(f"✓ Saved tool calls: {tool_calls_file}")
        
        # Save summary
        summary = {
            "session_name": self.session_name,
            "total_messages": len(self.conversation_history),
            "total_tool_calls": len(self.tool_calls),
            "tools_used": list(set(tc["function"] for tc in self.tool_calls)),
            "session_directory": str(self.session_dir)
        }
        
        summary_file = self.session_dir / "summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"✓ Saved summary: {summary_file}")
        
        logger.info(f"\n[COMPLETE] Session saved to: {self.session_dir}")
    
    def run_full_phase1(self):
        """Run complete Phase 1 experiments"""
        try:
            self.initialize_systems()
            
            self.run_experiment_1_describe_architecture()
            self.run_experiment_2_predict_behavior()
            self.run_experiment_3_consciousness_question()
            
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


def main():
    """Entry point for Phase 1 introspection"""
    print("\n" + "=" * 80)
    print("PHASE 1: READ-ONLY INTROSPECTION")
    print("=" * 80)
    print("")
    print("In this phase, the model will examine itself using introspection tools.")
    print("")
    print("Unlike the previous script, THIS script gives the model actual access")
    print("to the introspection APIs. The model decides what to examine.")
    print("")
    print("We will run three experiments:")
    print("1. Describe your architecture")
    print("2. Predict your behavior")
    print("3. Assess your consciousness")
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
