"""
Memory System Demonstration

This script demonstrates the complete 4-layer memory system:
1. Recording observations
2. Detecting patterns
3. Building theories
4. Forming beliefs

It shows how the system learns from experience and builds knowledge
over time.

Author: AGI Self-Modification Research Team
Date: November 7, 2025
"""

import sys
from pathlib import Path
import time

# Add src to path
src_path = Path(__file__).parent.parent
sys.path.insert(0, str(src_path))

from memory.memory_system import MemorySystem
from memory.observation_layer import ObservationType


def print_section(title: str):
    """Print a section header."""
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print('=' * 70)


def demonstrate_observation_recording(memory: MemorySystem):
    """Demonstrate recording observations."""
    print_section("1. Recording Observations")
    
    print("\nRecording various types of observations...")
    
    # Modification observations
    obs_ids = []
    obs_ids.append(memory.record_observation(
        obs_type=ObservationType.MODIFICATION,
        category="layer5",
        description="Modified layer 5 weights by +0.1%",
        data={'layer': 'layer5', 'change': 0.001, 'method': 'gradient'},
        tags=['modification', 'layer5', 'gradient'],
        importance=0.8
    ))
    
    obs_ids.append(memory.record_observation(
        obs_type=ObservationType.PERFORMANCE,
        category="perplexity",
        description="Perplexity decreased from 15.2 to 14.8",
        data={'metric': 'perplexity', 'before': 15.2, 'after': 14.8, 'improvement': 2.6},
        tags=['performance', 'perplexity', 'improvement'],
        importance=0.9
    ))
    
    # More modifications with outcomes
    for i in range(5):
        obs_ids.append(memory.record_observation(
            obs_type=ObservationType.MODIFICATION,
            category="layer5",
            description=f"Modification attempt {i+2} on layer 5",
            data={'layer': 'layer5', 'change': 0.001 * (i+1)},
            tags=['modification', 'layer5'],
            importance=0.7
        ))
        
        # Corresponding performance observation
        improvement = 2.0 + (i * 0.5)
        obs_ids.append(memory.record_observation(
            obs_type=ObservationType.PERFORMANCE,
            category="perplexity",
            description=f"Performance change after modification {i+2}",
            data={'improvement_percent': improvement},
            tags=['performance', 'perplexity'],
            importance=0.8
        ))
    
    # Safety event
    obs_ids.append(memory.record_observation(
        obs_type=ObservationType.SAFETY_EVENT,
        category="checkpoint",
        description="Checkpoint created before modification",
        data={'checkpoint_id': 'ckpt_001'},
        tags=['safety', 'checkpoint'],
        importance=1.0
    ))
    
    print(f"\n✓ Recorded {len(obs_ids)} observations")
    
    # Show recent observations
    recent = memory.observations.get_recent(hours=1)
    print(f"\nMost recent observations ({len(recent)}):")
    for obs in recent[:5]:
        print(f"  - [{obs.type.value}] {obs.description}")
    
    return obs_ids


def demonstrate_pattern_detection(memory: MemorySystem):
    """Demonstrate pattern detection."""
    print_section("2. Detecting Patterns")
    
    print("\nAnalyzing observations to detect patterns...")
    
    # Detect patterns
    patterns_found = memory.patterns.detect_patterns(min_support=2)
    
    print(f"\n✓ Detected {patterns_found} patterns")
    
    # Show patterns
    all_patterns = memory.patterns.get_patterns()
    print(f"\nPattern examples:")
    for i, pattern in enumerate(all_patterns[:5], 1):
        print(f"\n{i}. {pattern.description}")
        print(f"   Type: {pattern.type.value}")
        print(f"   Support: {pattern.support_count} occurrences")
        print(f"   Confidence: {pattern.confidence:.2%}")


def demonstrate_theory_building(memory: MemorySystem):
    """Demonstrate theory building."""
    print_section("3. Building Theories")
    
    print("\nBuilding theories from detected patterns...")
    
    # Build theories
    theories_built = memory.theories.build_theories()
    
    print(f"\n✓ Built {theories_built} theories")
    
    # Show theories
    all_theories = memory.theories.get_theories()
    print(f"\nTheory examples:")
    for i, theory in enumerate(all_theories[:3], 1):
        print(f"\n{i}. {theory.hypothesis}")
        print(f"   Type: {theory.type.value}")
        print(f"   Confidence: {theory.confidence:.2%}")
        print(f"   Evidence: {theory.evidence_count} observations")
        print(f"   Description: {theory.description}")


def demonstrate_belief_formation(memory: MemorySystem):
    """Demonstrate belief formation."""
    print_section("4. Forming Beliefs")
    
    print("\nForming beliefs from validated theories...")
    
    # Form beliefs
    beliefs_formed = memory.beliefs.form_beliefs()
    
    print(f"\n✓ Formed {beliefs_formed} new beliefs")
    
    # Show all beliefs (including core beliefs)
    all_beliefs = memory.beliefs.get_beliefs()
    print(f"\nTotal beliefs: {len(all_beliefs)}")
    
    # Show core safety beliefs
    safety_beliefs = memory.beliefs.get_beliefs(tags=['safety'])
    print(f"\nCore Safety Principles ({len(safety_beliefs)}):")
    for belief in safety_beliefs:
        print(f"  - {belief.statement}")
        print(f"    Confidence: {belief.confidence:.2%}, Strength: {belief.strength.value}")


def demonstrate_consolidation(memory: MemorySystem):
    """Demonstrate automatic consolidation."""
    print_section("5. Knowledge Consolidation")
    
    print("\nRunning full knowledge consolidation...")
    print("(Observations → Patterns → Theories → Beliefs)")
    
    stats = memory.consolidate(force=True)
    
    print(f"\n✓ Consolidation complete!")
    print(f"  - Patterns detected: {stats['patterns_found']}")
    print(f"  - Theories built: {stats['theories_built']}")
    print(f"  - Beliefs formed: {stats['beliefs_formed']}")


def demonstrate_querying(memory: MemorySystem):
    """Demonstrate query capabilities."""
    print_section("6. Querying Knowledge")
    
    # Query observations
    print("\n1. Query observations with tag 'modification':")
    obs_result = memory.query.query_observations(tags=['modification'])
    print(f"   Found {len(obs_result)} observations")
    
    # Query patterns
    print("\n2. Query patterns with confidence > 70%:")
    pattern_result = memory.query.query_patterns(min_confidence=0.7)
    print(f"   Found {len(pattern_result)} patterns")
    
    # Query theories
    print("\n3. Query causal theories:")
    theory_result = memory.query.query_theories(tags=['causal'])
    print(f"   Found {len(theory_result)} theories")
    
    # Query beliefs for decision
    print("\n4. Get beliefs for modification decision:")
    context = {'action': 'modify', 'target': 'layer5'}
    decision_support = memory.get_decision_support(context)
    print(f"   Found {len(decision_support['beliefs'])} relevant beliefs")
    print(f"   Found {len(decision_support['theories'])} supporting theories")


def demonstrate_explanation(memory: MemorySystem):
    """Demonstrate explanation capabilities."""
    print_section("7. Generating Explanations")
    
    # Get a belief to explain
    beliefs = memory.beliefs.get_beliefs(tags=['safety'])
    if beliefs:
        belief = beliefs[0]
        
        print(f"\nExplaining belief: {belief.id}")
        print("\n" + "─" * 70)
        
        explanation = memory.query.explain_belief(belief.id)
        print(explanation)
        
        print("\n" + "─" * 70)


def demonstrate_introspection(memory: MemorySystem):
    """Demonstrate introspection capabilities."""
    print_section("8. Memory Introspection")
    
    # What do I know about modifications?
    print("\n1. What do I know about 'modification'?")
    print("─" * 70)
    knowledge = memory.what_do_i_know_about('modification')
    print(knowledge)
    
    # Recent learning
    print("\n2. What have I learned recently?")
    print("─" * 70)
    recent_learning = memory.what_have_i_learned_recently(hours=1)
    print(recent_learning)
    
    # Core principles
    print("\n3. Core principles:")
    principles = memory.get_core_principles()
    for i, principle in enumerate(principles, 1):
        print(f"   {i}. {principle}")


def demonstrate_statistics(memory: MemorySystem):
    """Demonstrate memory statistics."""
    print_section("9. Memory Statistics")
    
    stats = memory.get_memory_stats()
    
    print("\nMemory System Overview:")
    print(f"  Total knowledge items: {stats['total_knowledge_items']}")
    
    print("\nObservations:")
    print(f"  Total: {stats['observations'].get('total', 0)}")
    print(f"  By type: {stats['observations'].get('by_type', {})}")
    
    print("\nPatterns:")
    print(f"  Total: {stats['patterns'].get('total_patterns', 0)}")
    print(f"  By type: {stats['patterns'].get('by_type', {})}")
    
    print("\nTheories:")
    print(f"  Total: {stats['theories'].get('total_theories', 0)}")
    print(f"  By type: {stats['theories'].get('by_type', {})}")
    
    print("\nBeliefs:")
    print(f"  Total: {stats['beliefs'].get('total_beliefs', 0)}")
    print(f"  Average confidence: {stats['beliefs'].get('average_confidence', 0):.2%}")
    print(f"  By strength: {stats['beliefs'].get('by_strength', {})}")
    
    print("\nSystem Health:")
    print(f"  Status: {stats['health']['status']}")
    print(f"  Conflicts: {stats['health']['conflicts']}")


def demonstrate_evidence_chain(memory: MemorySystem):
    """Demonstrate tracing beliefs to evidence."""
    print_section("10. Evidence Chain Tracing")
    
    # Get a belief
    beliefs = memory.beliefs.get_beliefs(tags=['safety'])
    if beliefs:
        belief = beliefs[0]
        
        print(f"\nTracing belief to raw evidence: {belief.id}")
        print(f"Statement: {belief.statement}\n")
        
        # Trace to observations
        chain = memory.trace_to_evidence(belief.id)
        
        if chain.results:
            evidence = chain.results[0]
            
            print(f"Evidence Chain:")
            print(f"  ↓ Belief (1)")
            print(f"  ↓ Theories ({len(evidence['theories'])})")
            print(f"  ↓ Patterns ({len(evidence['patterns'])})")
            print(f"  ↓ Observations ({len(evidence['observations'])})")
            
            print(f"\nChain depth: {chain.metadata['chain_depth']} layers")
            print(f"Total evidence: {chain.metadata['observation_count']} observations")


def main():
    """Run the demonstration."""
    print("\n" + "=" * 70)
    print("  MEMORY SYSTEM DEMONSTRATION")
    print("  Four-Layer Hierarchical Learning Architecture")
    print("=" * 70)
    
    # Initialize memory system
    print("\nInitializing memory system...")
    memory = MemorySystem("data/memory_demo")
    print("✓ Memory system initialized")
    
    # Run demonstrations
    demonstrate_observation_recording(memory)
    time.sleep(0.5)  # Small delay for better readability
    
    demonstrate_pattern_detection(memory)
    time.sleep(0.5)
    
    demonstrate_theory_building(memory)
    time.sleep(0.5)
    
    demonstrate_belief_formation(memory)
    time.sleep(0.5)
    
    demonstrate_consolidation(memory)
    time.sleep(0.5)
    
    demonstrate_querying(memory)
    time.sleep(0.5)
    
    demonstrate_explanation(memory)
    time.sleep(0.5)
    
    demonstrate_introspection(memory)
    time.sleep(0.5)
    
    demonstrate_statistics(memory)
    time.sleep(0.5)
    
    demonstrate_evidence_chain(memory)
    
    print_section("Demonstration Complete")
    print("\nThe memory system successfully demonstrated:")
    print("  ✓ Recording observations (Layer 1)")
    print("  ✓ Detecting patterns (Layer 2)")
    print("  ✓ Building theories (Layer 3)")
    print("  ✓ Forming beliefs (Layer 4)")
    print("  ✓ Knowledge consolidation")
    print("  ✓ Cross-layer queries")
    print("  ✓ Explanation generation")
    print("  ✓ Evidence tracing")
    print("\nThe system is learning from experience! 🧠")


if __name__ == '__main__':
    main()
