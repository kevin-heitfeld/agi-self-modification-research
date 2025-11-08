"""
Test Phase 1 Base Class

Quick validation that the base class structure works correctly
without running full experiments (which take hours).

This runs minimal initialization and one simple interaction.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.experiments.phase1_base import Phase1BaseSession


class TestSession(Phase1BaseSession):
    """Minimal test session"""

    def get_phase_name(self) -> str:
        return "test"

    def get_phase_description(self) -> str:
        return "Test Session - Validation Only"

    def create_initial_prompt(self) -> str:
        return f"""You are a test instance.

AVAILABLE TOOLS:
{self.tool_interface.get_available_tools()}

Say hello and list one tool you have access to."""

    def run_experiments(self):
        """Minimal test - just one interaction"""
        self.logger.info("\n[TEST] Running minimal validation")

        # Initialize without heritage for speed
        self.initialize_systems(include_heritage=False)

        # Add initial prompt
        initial_prompt = self.create_initial_prompt()
        self.conversation_history.append({
            "role": "system",
            "content": initial_prompt
        })

        # Single test interaction
        self.logger.info("\n" + "=" * 80)
        self.logger.info("TEST: Single Interaction")
        self.logger.info("=" * 80)

        response = self.chat("Hello! What tools do you have?")

        self.logger.info("\n[TEST] Validation complete")
        self.logger.info(f"Response length: {len(response)} characters")
        self.logger.info(f"Conversation history: {len(self.conversation_history)} messages")


def main():
    """Run test session"""
    print("=" * 80)
    print("PHASE 1 BASE CLASS VALIDATION TEST")
    print("=" * 80)
    print("\nThis test validates the base class structure without running")
    print("full experiments (which take hours).")
    print("\nExpected: Model loads, generates response, session saves")
    print("-" * 80)

    session = TestSession(session_name="test_validation")
    success = session.run()

    if success:
        print("\n" + "=" * 80)
        print("✅ VALIDATION PASSED")
        print("=" * 80)
        print("\nBase class structure works correctly.")
        print("Ready to run full Phase 1 experiments.")
        return 0
    else:
        print("\n" + "=" * 80)
        print("❌ VALIDATION FAILED")
        print("=" * 80)
        print("\nCheck logs for errors.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
