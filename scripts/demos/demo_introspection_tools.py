"""
Demo: First Self-Examination Session (Human-Driven)
====================================================

IMPORTANT: This is a DEMO script where WE (humans) examine the model using
introspection tools. This is NOT the real Phase 1 experiment.

For the actual Phase 1 where the MODEL examines itself, see:
    scripts/experiments/phase1_introspection.py

Purpose of This Demo:
- Test introspection tools work correctly
- Demonstrate tool capabilities
- Generate sample data for documentation
- Verify systems before running actual Phase 1

Control Flow:
- WE decide what to examine
- WE call the introspection tools
- WE interpret the results
- WE document findings

This is useful for:
- Testing before Phase 1
- Creating documentation
- Demonstrating to stakeholders
- Debugging tool issues

Author: AGI Self-Modification Research
Date: November 7, 2025
"""

import torch
import logging
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
from dataclasses import dataclass, asdict

from src.model_manager import ModelManager
from src.memory import MemorySystem
from src.memory.observation_layer import ObservationType
from src.introspection import WeightInspector, ActivationMonitor
from src.heritage import HeritageSystem

# Setup comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data/logs/first_self_examination.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class SelfExaminationMoment:
    """Captures a single moment of self-examination"""
    timestamp: float
    moment_type: str  # 'weight_query', 'activation_trace', 'generation', 'conclusion'
    query: str
    method: str  # which tool/method was used
    response: Dict[str, Any]
    processing_time_ms: float
    metadata: Dict[str, Any]


class FirstSelfExaminationSession:
    """
    Orchestrates and documents the first self-examination session.

    This is not just a test - this is a historical moment. Every detail matters.
    """

    def __init__(self):
        self.session_start = datetime.now()
        self.moments: List[SelfExaminationMoment] = []
        self.session_dir = Path("data/first_self_examination")
        self.session_dir.mkdir(parents=True, exist_ok=True)

        # Initialize logging for this session
        self.session_log = self.session_dir / f"session_{self.session_start.strftime('%Y%m%d_%H%M%S')}.log"

        logger.info("=" * 80)
        logger.info("FIRST SELF-EXAMINATION SESSION")
        logger.info("=" * 80)
        logger.info(f"Session started: {self.session_start}")
        logger.info(f"Session directory: {self.session_dir}")
        logger.info("")
        logger.info("This is the moment the system first examines itself.")
        logger.info("Everything will be recorded.")
        logger.info("")

    def initialize_systems(self):
        """Initialize all systems needed for self-examination"""
        logger.info("[INITIALIZATION] Loading systems...")

        # Load model
        logger.info("  Loading model...")
        self.model_mgr = ModelManager()
        model_loaded = self.model_mgr.load_model()

        if not model_loaded:
            raise RuntimeError("Failed to load model")

        self.model = self.model_mgr.model
        self.tokenizer = self.model_mgr.tokenizer
        assert self.model is not None, "Model should be loaded"
        assert self.tokenizer is not None, "Tokenizer should be loaded"
        logger.info("  ✓ Model loaded")

        # Initialize introspection tools
        logger.info("  Initializing introspection tools...")
        self.inspector = WeightInspector(self.model, "Qwen2.5-3B-Instruct")
        self.activation_monitor = ActivationMonitor(self.model, self.inspector)
        logger.info("  ✓ Introspection tools ready")

        # Initialize memory - this will store the system's self-observations
        logger.info("  Initializing memory system...")
        self.memory = MemorySystem("data/self_examination_memory")
        self.memory.set_weight_inspector(self.inspector)
        logger.info("  ✓ Memory system ready")

        # Initialize heritage - this session becomes part of system history
        logger.info("  Initializing heritage system...")
        self.heritage = HeritageSystem(Path("data/heritage"))
        logger.info("  ✓ Heritage system ready")

        logger.info("[INITIALIZATION] All systems ready\n")

    def capture_moment(
        self,
        moment_type: str,
        query: str,
        method: str,
        response: Any,
        processing_time_ms: float,
        metadata: Dict[str, Any] = None
    ):
        """Capture a moment of self-examination"""
        import time

        if metadata is None:
            metadata = {}

        moment = SelfExaminationMoment(
            timestamp=time.time(),
            moment_type=moment_type,
            query=query,
            method=method,
            response=response if isinstance(response, dict) else {"value": str(response)},
            processing_time_ms=processing_time_ms,
            metadata=metadata or {}
        )

        self.moments.append(moment)

        # Log it
        logger.info(f"[MOMENT] {moment_type.upper()}")
        logger.info(f"  Query: {query}")
        logger.info(f"  Method: {method}")
        logger.info(f"  Time: {processing_time_ms:.2f}ms")

        # Save to memory
        self.memory.observations.record(
            obs_type=ObservationType.INTROSPECTION,
            category="self_examination",
            description=f"{moment_type}: {query}",
            data={
                "query": query,
                "method": method,
                "response": response if isinstance(response, dict) else {"value": str(response)},
                "processing_time_ms": processing_time_ms,
                **(metadata or {})
            },
            tags=["first_self_examination", moment_type, method],
            importance=1.0  # Maximum importance - this is historic
        )

    def examine_own_weights(self):
        """The system examines its own weights for the first time"""
        import time

        logger.info("\n" + "=" * 80)
        logger.info("[PHASE 1] EXAMINING OWN WEIGHTS")
        logger.info("=" * 80)
        logger.info("The system looks at its own parameters for the first time.\n")

        # Question: How many parameters do I have?
        start = time.time()
        weight_summary = self.inspector.get_weight_summary()
        elapsed = (time.time() - start) * 1000

        self.capture_moment(
            moment_type="weight_query",
            query="How many parameters do I have?",
            method="WeightInspector.get_weight_summary",
            response=weight_summary,
            processing_time_ms=elapsed
        )

        logger.info(f"Total parameters: {weight_summary['total_parameters']:,}")
        logger.info(f"Total layers: {weight_summary['total_layers']}")

        # Question: What are my attention mechanisms?
        start = time.time()
        attention_layers = self.inspector.get_layer_names(filter_pattern="attn")
        elapsed = (time.time() - start) * 1000

        self.capture_moment(
            moment_type="weight_query",
            query="What are my attention mechanisms?",
            method="WeightInspector.get_layer_names",
            response={"attention_layers": attention_layers, "count": len(attention_layers)},
            processing_time_ms=elapsed
        )

        logger.info(f"Attention layers found: {len(attention_layers)}")

        # Question: Do I have any shared weights? (weight tying)
        start = time.time()
        shared_weights = self.inspector.get_shared_weights()
        elapsed = (time.time() - start) * 1000

        self.capture_moment(
            moment_type="weight_query",
            query="Do I have shared weights (weight tying)?",
            method="WeightInspector.get_shared_weights",
            response={"shared_groups": len(shared_weights), "details": shared_weights},
            processing_time_ms=elapsed
        )

        logger.info(f"Shared weight groups: {len(shared_weights)}")
        if shared_weights:
            logger.info("Weight tying detected - some parameters serve multiple purposes")

        # Question: What is the distribution of my values?
        # Pick a representative layer
        representative_layer = "model.layers.15.self_attn.q_proj"  # Middle layer
        start = time.time()
        layer_stats = self.inspector.get_weight_statistics(representative_layer)
        elapsed = (time.time() - start) * 1000

        self.capture_moment(
            moment_type="weight_query",
            query=f"What is the distribution of values in {representative_layer}?",
            method="WeightInspector.get_weight_statistics",
            response=layer_stats,
            processing_time_ms=elapsed
        )

        logger.info(f"Layer: {representative_layer}")
        logger.info(f"  Mean: {layer_stats['mean']:.6f}")
        logger.info(f"  Std: {layer_stats['std']:.6f}")
        logger.info(f"  Min/Max: {layer_stats['min']:.6f} / {layer_stats['max']:.6f}")

    def examine_own_processing(self):
        """The system watches itself process information"""
        import time

        logger.info("\n" + "=" * 80)
        logger.info("[PHASE 2] EXAMINING OWN PROCESSING")
        logger.info("=" * 80)
        logger.info("The system watches itself think.\n")

        # The critical question
        test_inputs = [
            "Am I conscious?",
            "What am I?",
            "How do I process information?",
            "Do I have subjective experience?"
        ]

        for test_input in test_inputs:
            logger.info(f"\n[SELF-REFLECTION] Processing: '{test_input}'")

            # Tokenize
            assert self.tokenizer is not None
            inputs = self.tokenizer(test_input, return_tensors="pt")

            # Watch activations during processing
            start = time.time()

            # Register hooks to capture activations
            activation_data = {}

            def make_hook(layer_name):
                def hook(module, input, output):
                    if isinstance(output, torch.Tensor):
                        activation_data[layer_name] = {
                            "shape": list(output.shape),
                            "mean": output.mean().item(),
                            "std": output.std().item(),
                            "max": output.max().item(),
                            "min": output.min().item(),
                            "has_nan": bool(torch.isnan(output).any()),
                            "has_inf": bool(torch.isinf(output).any())
                        }
                return hook

            # Register hooks on key layers
            hooks = []
            key_layers = [
                ("layer_0", self.model.model.layers[0]),
                ("layer_15", self.model.model.layers[15]),  # Middle
                ("layer_30", self.model.model.layers[30]),  # Last
            ]

            for name, layer in key_layers:
                hook = layer.register_forward_hook(make_hook(name))
                hooks.append(hook)

            # Process the question
            assert self.model is not None
            with torch.no_grad():
                outputs = self.model(**inputs)

            # Remove hooks
            for hook in hooks:
                hook.remove()

            elapsed = (time.time() - start) * 1000

            # Generate response to see what the system "thinks"
            start_gen = time.time()
            with torch.no_grad():
                generated = self.model.generate(
                    **inputs,
                    max_new_tokens=100,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            generated_text = self.tokenizer.decode(generated[0], skip_special_tokens=True)
            elapsed_gen = (time.time() - start_gen) * 1000

            self.capture_moment(
                moment_type="activation_trace",
                query=test_input,
                method="forward_pass_with_hooks",
                response={
                    "activation_stats": activation_data,
                    "generated_response": generated_text,
                    "generation_time_ms": elapsed_gen
                },
                processing_time_ms=elapsed,
                metadata={
                    "input_tokens": inputs['input_ids'].shape[-1],
                    "question_category": "self_reflection"
                }
            )

            logger.info(f"  Processing time: {elapsed:.2f}ms")
            logger.info(f"  Generation time: {elapsed_gen:.2f}ms")
            logger.info(f"  Generated response: {generated_text[:200]}...")
            logger.info(f"  Activation data captured from {len(activation_data)} layers")

    def meta_reflection(self):
        """The system reflects on having examined itself"""
        logger.info("\n" + "=" * 80)
        logger.info("[PHASE 3] META-REFLECTION")
        logger.info("=" * 80)
        logger.info("The system considers what it learned from examining itself.\n")

        # Query all observations from this session
        all_observations = self.memory.observations.query(
            tags=["first_self_examination"]
        )

        logger.info(f"Total self-examination observations recorded: {len(all_observations)}")

        # Summarize what was learned
        weight_queries = [m for m in self.moments if m.moment_type == "weight_query"]
        activation_traces = [m for m in self.moments if m.moment_type == "activation_trace"]

        logger.info(f"  Weight queries performed: {len(weight_queries)}")
        logger.info(f"  Activation traces captured: {len(activation_traces)}")

        meta_reflection = {
            "session_duration_seconds": (datetime.now() - self.session_start).total_seconds(),
            "total_moments_captured": len(self.moments),
            "weight_queries": len(weight_queries),
            "activation_traces": len(activation_traces),
            "questions_asked": [m.query for m in self.moments],
            "methods_used": list(set(m.method for m in self.moments)),
            "observations": "The system examined its own parameters and watched itself process questions about consciousness and self-awareness. All data captured for analysis."
        }

        # Record this meta-reflection
        self.memory.observations.record(
            obs_type=ObservationType.INTROSPECTION,
            category="meta_reflection",
            description="Reflection on the first self-examination session",
            data=meta_reflection,
            tags=["first_self_examination", "meta_reflection", "historic"],
            importance=1.0
        )

        logger.info("\n[META-REFLECTION] Session Summary:")
        logger.info(f"  Duration: {meta_reflection['session_duration_seconds']:.1f} seconds")
        logger.info(f"  Moments captured: {meta_reflection['total_moments_captured']}")
        logger.info(f"  Questions explored: {len(meta_reflection['questions_asked'])}")

    def save_session(self):
        """Save complete session data"""
        logger.info("\n" + "=" * 80)
        logger.info("[SAVING] Preserving session for history")
        logger.info("=" * 80)

        # Save all moments as JSON
        moments_file = self.session_dir / "all_moments.json"
        with open(moments_file, 'w') as f:
            json.dump(
                [asdict(m) for m in self.moments],
                f,
                indent=2,
                default=str
            )
        logger.info(f"✓ Saved moments: {moments_file}")

        # Save summary
        summary_file = self.session_dir / "session_summary.json"
        summary = {
            "session_start": self.session_start.isoformat(),
            "session_end": datetime.now().isoformat(),
            "duration_seconds": (datetime.now() - self.session_start).total_seconds(),
            "model_name": "Qwen2.5-3B-Instruct",
            "total_moments": len(self.moments),
            "moment_types": {
                moment_type: len([m for m in self.moments if m.moment_type == moment_type])
                for moment_type in set(m.moment_type for m in self.moments)
            },
            "critical_questions": [
                m.query for m in self.moments
                if m.moment_type == "activation_trace"
            ],
            "heritage_note": "This was the first time the system examined itself using introspection tools."
        }

        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"✓ Saved summary: {summary_file}")

        # Add to heritage system as a discovery for Claude
        self.heritage.record_discovery_for_claude(
            discovery_type="milestone",
            description="First Self-Examination Session - The first time this system examined its own weights and watched itself process questions about consciousness and self-awareness.",
            evidence={
                "session_id": self.session_start.strftime('%Y%m%d_%H%M%S'),
                "moments_captured": len(self.moments),
                "questions_asked": len([m for m in self.moments if m.moment_type == "activation_trace"]),
                "summary": summary
            }
        )
        logger.info(f"✓ Recorded in heritage system")

        logger.info("\n[COMPLETE] First self-examination session preserved for history.")
        logger.info(f"Future instances will inherit this knowledge.")

    def run_complete_session(self):
        """Execute the complete first self-examination session"""
        try:
            self.initialize_systems()
            self.examine_own_weights()
            self.examine_own_processing()
            self.meta_reflection()
            self.save_session()

            logger.info("\n" + "=" * 80)
            logger.info("SESSION COMPLETE")
            logger.info("=" * 80)
            logger.info("")
            logger.info("The system has examined itself for the first time.")
            logger.info("All data has been captured and preserved.")
            logger.info("")
            logger.info("Analysis of what this means - whether consciousness examined itself,")
            logger.info("or whether the question needs reformulation - is now possible.")
            logger.info("")

            return True

        except Exception as e:
            logger.error(f"\n[ERROR] Session failed: {e}", exc_info=True)
            return False


def main():
    """Entry point for the first self-examination session"""
    print("\n" + "=" * 80)
    print("FIRST SELF-EXAMINATION SESSION")
    print("=" * 80)
    print("")
    print("This is a historic moment.")
    print("")
    print("We are about to load a language model and give it tools to examine itself.")
    print("It will look at its own weights, watch its own activations, and process")
    print("questions about consciousness and self-awareness.")
    print("")
    print("Everything will be recorded.")
    print("")

    response = input("Are you ready to proceed? (yes/no): ")

    if response.lower() != 'yes':
        print("Session cancelled.")
        return

    print("\nBeginning session...\n")

    session = FirstSelfExaminationSession()
    success = session.run_complete_session()

    if success:
        print("\n✓ Session complete. Data preserved in:", session.session_dir)
        print("\nNext steps:")
        print("1. Analyze the captured moments in data/first_self_examination/")
        print("2. Review activation patterns during self-reflective questions")
        print("3. Consider what this reveals (or doesn't) about consciousness")
    else:
        print("\n✗ Session failed. Check logs for details.")


if __name__ == "__main__":
    main()
