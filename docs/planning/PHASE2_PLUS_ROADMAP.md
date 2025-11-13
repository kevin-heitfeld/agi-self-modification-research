# Memory System Roadmap: Phase 2+ Planning

**Version:** 1.0  
**Date:** 2025-01-XX  
**Status:** DESIGN PHASE  
**Prerequisites:** Phase 1 Lifecycle Management (Complete)

## Executive Summary

This document outlines the evolution of the memory system beyond Phase 1's lifecycle management. The roadmap focuses on enabling advanced epistemic reasoning capabilities:

- **Phase 2:** Git-like branching for hypothesis testing
- **Phase 3:** Auto-cleanup with retention policies  
- **Phase 4:** Confidence propagation and lazy revalidation
- **Phase 5:** Epistemic metrics and learning analytics
- **Phase 6:** Cross-layer versioning and consolidation v2

Each phase builds on previous capabilities to create a robust, self-aware knowledge management system suitable for AGI development.

---

## Phase 2: Git-Like Branching for Hypothesis Testing

### Motivation

**Current limitation:**
- Model can only work in single timeline
- Testing hypothesis requires committing to main knowledge base
- Can't compare alternative interpretations
- Risk: Wrong hypothesis pollutes main memory

**Scientific method needs:**
1. Form multiple competing hypotheses
2. Test each in isolation
3. Compare evidence for each
4. Commit winning hypothesis

**Git analogy:**
- `main` branch = accepted knowledge
- Feature branches = hypotheses being tested
- Merge = hypothesis confirmation
- Branch comparison = evidence evaluation

### Design Overview

#### Data Model

**New Table: branches**
```sql
CREATE TABLE branches (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    created_at REAL NOT NULL,
    created_from TEXT,  -- Parent branch ID
    description TEXT,
    status TEXT DEFAULT 'active',  -- active | merged | abandoned
    merged_into TEXT,
    merged_at REAL
)
```

**New Table: observation_branches**
```sql
CREATE TABLE observation_branches (
    observation_id TEXT,
    branch_id TEXT,
    PRIMARY KEY (observation_id, branch_id),
    FOREIGN KEY (observation_id) REFERENCES observations(id),
    FOREIGN KEY (branch_id) REFERENCES branches(id)
)
```

**Context Management:**
```python
class BranchContext:
    """Thread-local branch context for isolated operations"""
    current_branch: str = "main"  # Default
    
    @contextmanager
    def use_branch(branch_id: str):
        old_branch = BranchContext.current_branch
        BranchContext.current_branch = branch_id
        try:
            yield
        finally:
            BranchContext.current_branch = old_branch
```

#### Core Operations

**1. Branch Creation**
```python
def create_branch(
    name: str,
    description: str,
    from_branch: str = "main"
) -> str:
    """
    Create new branch for hypothesis testing.
    
    Args:
        name: Branch name (e.g., "hypothesis_attention_masking")
        description: What hypothesis this branch tests
        from_branch: Parent branch to fork from
        
    Returns:
        branch_id: UUID of created branch
        
    Example:
        >>> branch_id = create_branch(
        ...     "test_layer5_specialization",
        ...     "Testing whether layer 5 specializes in context modeling",
        ...     from_branch="main"
        ... )
        'branch_8a3f...'
    """
```

**2. Branch Switching**
```python
def switch_branch(branch_name: str) -> Dict[str, Any]:
    """
    Switch working context to different branch.
    
    All subsequent observations recorded to this branch.
    
    Args:
        branch_name: Name of branch to switch to
        
    Returns:
        Dict with branch info and statistics
        
    Example:
        >>> switch_branch("test_layer5_specialization")
        {'branch': 'test_layer5_specialization', 
         'observations_in_branch': 12, ...}
    """
```

**3. Branch Listing**
```python
def list_branches(
    include_merged: bool = False,
    include_abandoned: bool = False
) -> List[Dict[str, Any]]:
    """
    List all branches with statistics.
    
    Args:
        include_merged: Include branches already merged
        include_abandoned: Include abandoned branches
        
    Returns:
        List of branch info dicts
        
    Example:
        >>> list_branches()
        [
            {
                'name': 'main',
                'observation_count': 1247,
                'created_at': 1704067200.0,
                'status': 'active'
            },
            {
                'name': 'test_layer5_specialization',
                'observation_count': 12,
                'created_at': 1704153600.0,
                'status': 'active',
                'parent': 'main'
            }
        ]
    """
```

**4. Branch Comparison**
```python
def compare_branches(
    branch1: str,
    branch2: str,
    show_details: bool = True
) -> Dict[str, Any]:
    """
    Compare two branches to see differences.
    
    Args:
        branch1: First branch name
        branch2: Second branch name
        show_details: Include detailed observation diffs
        
    Returns:
        Comparison report with:
        - unique_to_branch1: Observations only in branch1
        - unique_to_branch2: Observations only in branch2
        - common: Observations in both
        - conflicts: Same observation updated differently
        
    Example:
        >>> diff = compare_branches("main", "test_layer5_specialization")
        {
            'unique_to_test_layer5_specialization': 8,
            'unique_to_main': 3,
            'conflicts': 0,
            'recommendations': 'Branch has unique evidence supporting hypothesis'
        }
    """
```

**5. Branch Merging**
```python
def merge_branch(
    source_branch: str,
    target_branch: str = "main",
    strategy: str = "validate",
    conflict_resolution: str = "manual"
) -> Dict[str, Any]:
    """
    Merge source branch into target branch.
    
    Args:
        source_branch: Branch to merge from
        target_branch: Branch to merge into (default: main)
        strategy: Merge strategy:
            - "validate": Only merge if hypothesis confirmed
            - "cherry_pick": Select specific observations
            - "full": Merge all observations
        conflict_resolution: How to handle conflicts:
            - "manual": Stop and request human input
            - "keep_target": Keep target branch version
            - "keep_source": Keep source branch version
            - "create_both": Create versions for both
            
    Returns:
        Merge report with statistics and conflicts
        
    Example:
        >>> result = merge_branch(
        ...     "test_layer5_specialization",
        ...     target_branch="main",
        ...     strategy="validate"
        ... )
        {
            'merged_observations': 8,
            'conflicts': 0,
            'branch_status': 'merged',
            'confidence_increase': 0.12
        }
    """
```

**6. Branch Deletion**
```python
def delete_branch(
    branch_name: str,
    force: bool = False
) -> Dict[str, Any]:
    """
    Delete branch and associated observations.
    
    Args:
        branch_name: Branch to delete (cannot be 'main')
        force: Delete even if not merged (loses data)
        
    Returns:
        Deletion report
        
    Example:
        >>> delete_branch("failed_hypothesis_xyz", force=True)
        {'deleted_observations': 15, 'warnings': ['Unmerged data lost']}
    """
```

### Implementation Plan

**Step 1: Database Schema (Week 1)**
- Add `branches` table
- Add `observation_branches` junction table
- Add indices for performance
- Migration script for existing data (all to 'main' branch)

**Step 2: Branch Context Management (Week 1-2)**
- Implement `BranchContext` thread-local storage
- Update `record_observation()` to use current branch
- Update queries to filter by branch
- Test isolation (changes in branch don't affect main)

**Step 3: Basic Branch Operations (Week 2-3)**
- Implement `create_branch()`
- Implement `switch_branch()`
- Implement `list_branches()`
- Test branch creation and switching

**Step 4: Branch Comparison (Week 3-4)**
- Implement diff algorithm
- Identify unique observations per branch
- Detect conflicts (same observation updated differently)
- Generate comparison report

**Step 5: Merge Strategies (Week 4-6)**
- Implement validation merge (only if hypothesis supported)
- Implement cherry-pick merge (selective)
- Implement full merge (all observations)
- Conflict resolution handlers

**Step 6: Tool Interface (Week 6)**
- Add branch operations to `tool_interface.py`
- Write comprehensive documentation
- Add usage examples
- Test with model in Phase 1a

### Use Cases

**Use Case 1: Testing Competing Hypotheses**

```python
# Model observes high layer 5 activations
record_observation(
    obs_type="INTROSPECTION",
    description="Layer 5 shows consistently high activations on context tokens",
    data={"avg_activation": 0.87},
    tags=["layer5", "activations"]
)

# Form two competing hypotheses
branch1 = create_branch(
    "hypothesis_context_specialization",
    "Layer 5 specializes in context modeling"
)

branch2 = create_branch(
    "hypothesis_attention_artifact",
    "High activations are attention mechanism artifacts"
)

# Test hypothesis 1
with use_branch(branch1):
    # Record evidence for context specialization
    record_observation(...)
    record_observation(...)
    
# Test hypothesis 2  
with use_branch(branch2):
    # Record evidence for attention artifacts
    record_observation(...)
    record_observation(...)

# Compare evidence
comparison = compare_branches(branch1, branch2)

# Merge winning hypothesis
if comparison['confidence_branch1'] > comparison['confidence_branch2']:
    merge_branch(branch1, strategy="validate")
    delete_branch(branch2, force=True)
else:
    merge_branch(branch2, strategy="validate")
    delete_branch(branch1, force=True)
```

**Use Case 2: Experimental Analysis**

```python
# Create experiment branch
exp_branch = create_branch(
    "experiment_pruning_analysis",
    "Testing impact of weight pruning on different layers"
)

switch_branch(exp_branch)

# Run experiments in isolated branch
for layer in range(24):
    # Test pruning this layer
    result = test_layer_pruning(layer)
    record_observation(
        obs_type="MODIFICATION",
        description=f"Layer {layer} pruning impact",
        data=result,
        tags=["pruning", f"layer{layer}"]
    )

# Review results
switch_branch("main")
comparison = compare_branches("main", exp_branch)

# If findings valuable, merge
if comparison['significance'] > 0.8:
    merge_branch(exp_branch, strategy="cherry_pick")
else:
    delete_branch(exp_branch, force=True)
```

### Challenges

**Challenge 1: Cross-Layer References**

**Problem:**
- Patterns reference observations as evidence
- If observation in different branch, reference breaks
- Pattern in main can't reference branch observation

**Solutions:**

*Option A: Branch-Local Patterns*
```python
# Patterns also get branch assignment
# Can only reference observations in same branch
# Merge patterns along with observations
```

*Option B: Cross-Branch References with Validation*
```python
# Allow cross-branch references
# Mark as "tentative" if referencing other branch
# Validate during merge (confirm evidence still valid)
```

*Option C: Pattern Reconstruction*
```python
# Don't duplicate patterns to branches
# Reconstruct patterns during merge based on new observations
# Consolidation creates patterns from merged observations
```

**Recommendation:** Start with Option C (simplest), evolve to Option B if needed.

**Challenge 2: Merge Conflicts**

**Problem:**
```python
# Main branch
obs_123 = {
    "description": "Layer 5 activation average",
    "data": {"value": 0.75}
}

# Branch updates same observation
obs_123_branch = {
    "description": "Layer 5 activation average (refined)",
    "data": {"value": 0.82, "std_dev": 0.05}
}

# Merge conflict: Which version to keep?
```

**Resolution Strategies:**

1. **Keep Both (Version Tree):**
   ```python
   # Create both versions, mark as alternatives
   obs_123_v2_main = update_observation(obs_123, ...)
   obs_123_v2_branch = update_observation(obs_123, ...)
   # Tag both as "merge_conflict_resolved"
   ```

2. **Manual Review:**
   ```python
   # Stop merge, present conflict to user
   conflict = {
       "observation_id": "obs_123",
       "main_version": {...},
       "branch_version": {...},
       "recommendation": "Branch version more precise"
   }
   # Wait for manual resolution
   ```

3. **Automatic Resolution Rules:**
   ```python
   # Use heuristics
   if branch_version.importance > main_version.importance:
       keep = branch_version
   elif branch_version.data.get("std_dev"):  # More precise
       keep = branch_version
   else:
       keep = main_version
   ```

**Challenge 3: Resource Limits**

**Problem:**
- Can't store infinite branches
- Each branch consumes memory/storage
- Model might create too many branches

**Solutions:**

1. **Branch Limits:**
   ```python
   MAX_ACTIVE_BRANCHES = 5  # Excluding main
   MAX_BRANCH_AGE_HOURS = 48  # Auto-abandon old branches
   MAX_BRANCH_SIZE = 1000  # Max observations per branch
   ```

2. **Automatic Cleanup:**
   ```python
   # Periodically clean up
   - Abandon branches with no activity for 24 hours
   - Merge branches that reached high confidence
   - Delete branches marked "experiment" after 48 hours
   ```

3. **Branch Priorities:**
   ```python
   # Tag branches with priority
   create_branch("important_hypothesis", priority="high")
   create_branch("exploratory_test", priority="low")
   # Low priority branches cleaned up first
   ```

### Metrics & Success Criteria

**Functional:**
- ✅ Can create branches
- ✅ Observations isolated to branch
- ✅ Can switch between branches
- ✅ Can compare branches
- ✅ Can merge branches
- ✅ Conflicts detected and resolved

**Performance:**
- ✅ Branch operations <100ms
- ✅ Query filtering works correctly
- ✅ Merge operations <1s for typical branch
- ✅ Storage overhead <20% with 5 active branches

**User Experience:**
- ✅ Model creates branches for hypotheses
- ✅ Model successfully tests in isolation
- ✅ Model compares evidence across branches
- ✅ Model merges confirmed hypotheses
- ✅ Model abandons failed hypotheses

---

## Phase 3: Auto-Cleanup with Retention Policies

### Motivation

**Problem:**
- Obsolete observations accumulate over time
- Storage grows unbounded
- Most obsolete data never referenced again
- Performance degrades with large database

**Goal:**
- Automatic cleanup of old obsolete observations
- Configurable retention policies
- Safety: Never delete important historical data
- Transparency: Full audit trail of deletions

### Design Overview

#### Retention Policy Configuration

```python
@dataclass
class RetentionPolicy:
    """Configuration for automatic cleanup"""
    
    # Time-based retention
    obsolete_retention_days: int = 30
    superseded_retention_days: int = 90
    deprecated_retention_days: int = 60
    
    # Version-based retention
    min_version_retention: int = 2  # Keep at least N versions
    
    # Category-based exceptions
    preserve_corrections: bool = True  # Never delete corrections
    preserve_categories: List[str] = ["CRITICAL"]
    preserve_tags: List[str] = ["important", "milestone"]
    
    # Safety limits
    max_delete_per_run: int = 1000  # Don't delete too much at once
    dry_run: bool = False  # Preview before deleting
```

#### Cleanup Operations

**1. Scheduled Cleanup**
```python
def cleanup_old_observations(
    policy: RetentionPolicy,
    simulate: bool = False
) -> Dict[str, Any]:
    """
    Remove old obsolete observations per policy.
    
    Args:
        policy: Retention policy configuration
        simulate: If True, don't actually delete (preview)
        
    Returns:
        Cleanup report with stats
        
    Example:
        >>> policy = RetentionPolicy(
        ...     obsolete_retention_days=30,
        ...     preserve_corrections=True
        ... )
        >>> report = cleanup_old_observations(policy, simulate=True)
        {
            'candidates_for_deletion': 47,
            'would_preserve': 5,  # Corrections
            'space_would_free_mb': 12,
            'warnings': ['3 observations have dependent patterns']
        }
    """
```

**2. Selective Cleanup**
```python
def cleanup_observations_by_criteria(
    older_than_days: int = 30,
    status: str = "obsolete",
    exclude_tags: Optional[List[str]] = None,
    limit: int = 100
) -> Dict[str, Any]:
    """
    Clean up observations matching specific criteria.
    
    More targeted than policy-based cleanup.
    """
```

**3. Cascade Cleanup**
```python
def cleanup_orphaned_data() -> Dict[str, Any]:
    """
    Remove data orphaned by deletions.
    
    - Observation tags for deleted observations
    - Pattern evidence references to deleted observations
    - Theory pattern references to deleted patterns
    
    Returns:
        Report of orphaned data removed
    """
```

### Implementation Strategy

**Soft Delete → Hard Delete Pipeline:**

```python
# Phase 1: Soft delete (mark for deletion)
def mark_for_deletion(observation_id: str, reason: str):
    """Add 'marked_for_deletion' status"""
    cursor.execute("""
        UPDATE observations
        SET status = 'marked_for_deletion',
            obsolete_reason = ?
        WHERE id = ?
    """, (reason, observation_id))

# Phase 2: Archive (move to archive table)
def archive_observation(observation_id: str):
    """Move to archive table (cold storage)"""
    cursor.execute("""
        INSERT INTO observations_archive
        SELECT * FROM observations WHERE id = ?
    """, (observation_id,))
    # Keep in archive for 1 year

# Phase 3: Hard delete (permanent removal)
def permanently_delete(observation_id: str):
    """Remove from archive (gone forever)"""
    cursor.execute("""
        DELETE FROM observations_archive WHERE id = ?
    """, (observation_id,))
```

---

## Phase 4: Confidence Propagation & Lazy Revalidation

### Motivation

**Problem:**
- Observation obsoleted → patterns using it are now questionable
- But immediate cascade revalidation is expensive
- Need to propagate confidence changes lazily

**Goal:**
- Flag affected patterns when observation obsoleted
- Revalidate during consolidation (batched)
- Downgrade confidence when evidence weakens
- Mark patterns obsolete when evidence gone

### Algorithm

```python
def revalidate_pattern(pattern_id: str) -> Dict[str, Any]:
    """
    Check if pattern still valid given current evidence.
    
    Called during consolidation for flagged patterns.
    """
    pattern = patterns.get(pattern_id)
    evidence_obs = [observations.get(eid) for eid in pattern.evidence]
    
    # Count active vs obsolete evidence
    active = [o for o in evidence_obs if o and o.status == 'active']
    obsolete = [o for o in evidence_obs if o and o.status != 'active']
    
    active_ratio = len(active) / len(evidence_obs)
    
    if active_ratio == 0:
        # All evidence obsolete
        return obsolete_pattern(
            pattern_id,
            "All supporting evidence obsolete"
        )
    elif active_ratio < 0.5:
        # Majority obsolete - significant confidence hit
        new_confidence = pattern.confidence * active_ratio * 0.5
        return update_pattern(pattern_id, {
            "confidence": max(0.1, new_confidence),
            "tags": pattern.tags + ["evidence_degraded"],
            "metadata": {
                **pattern.metadata,
                "revalidated_at": time.time(),
                "evidence_loss_ratio": 1 - active_ratio
            }
        })
    elif active_ratio < 1.0:
        # Some evidence lost - minor confidence reduction
        new_confidence = pattern.confidence * (0.7 + 0.3 * active_ratio)
        return update_pattern(pattern_id, {
            "confidence": new_confidence,
            "metadata": {
                **pattern.metadata,
                "revalidated_at": time.time(),
                "evidence_loss_ratio": 1 - active_ratio
            }
        })
    else:
        # All evidence still active
        return {"status": "valid", "confidence": pattern.confidence}
```

---

## Phase 5: Epistemic Metrics & Learning Analytics

### Metrics to Track

```python
@dataclass
class EpistemicMetrics:
    """Metrics about knowledge evolution"""
    
    # Volume metrics
    total_observations: int
    total_patterns: int
    total_theories: int
    total_beliefs: int
    
    # Lifecycle metrics
    total_updates: int
    total_corrections: int
    total_obsoleted: int
    update_rate: float  # Updates per day
    correction_rate: float  # Percentage of observations corrected
    
    # Quality metrics
    avg_observation_lifespan_hours: float
    avg_version_count: float
    avg_confidence: float
    confidence_trend: str  # "increasing" | "decreasing" | "stable"
    
    # Error analysis
    corrections_by_category: Dict[str, int]
    time_to_correction_avg_minutes: float
    common_error_patterns: List[str]
    
    # Learning efficiency
    observations_per_consolidation: float
    patterns_per_observation: float
    theories_per_pattern: float
    knowledge_growth_rate: float  # New beliefs per day
```

### Visualization & Reporting

```python
def generate_epistemic_report(
    start_time: Optional[float] = None,
    end_time: Optional[float] = None
) -> Dict[str, Any]:
    """
    Generate comprehensive report on knowledge evolution.
    
    Includes:
    - Volume trends (observations over time)
    - Error rates (corrections over time)
    - Confidence evolution (avg confidence over time)
    - Category breakdown (which areas have most updates)
    - Efficiency metrics (consolidation effectiveness)
    """
```

---

## Phase 6: Cross-Layer Versioning & Consolidation v2

### Motivation

**Current limitation:**
- Only observations have versioning
- Patterns, theories, beliefs are still mutable
- Can't track how high-level beliefs evolved

**Goal:**
- Apply lifecycle management to all layers
- Track belief evolution over time
- Understand how theories get refined
- Full epistemic history across all layers

### Design

**Apply Phase 1 to each layer:**

```python
# Pattern versioning
update_pattern(pattern_id, updates, reason)
correct_pattern(pattern_id, correction, reason)
obsolete_pattern(pattern_id, reason, cascade)

# Theory versioning
update_theory(theory_id, updates, reason)
correct_theory(theory_id, correction, reason)
obsolete_theory(theory_id, reason, cascade)

# Belief versioning
update_belief(belief_id, updates, reason)
correct_belief(belief_id, correction, reason)
obsolete_belief(belief_id, reason, cascade)
```

**Consolidation v2:**
- Check observation versions when forming patterns
- Use most recent active version
- Revalidate patterns when evidence updated
- Propagate updates up through layers

---

## Timeline & Dependencies

```
Phase 1: Lifecycle Management [COMPLETE]
    ├─ Observation versioning ✅
    ├─ Update/Correct/Obsolete operations ✅
    └─ Tool interface ✅

Phase 2: Branching [4-6 weeks]
    ├─ Week 1-2: Database schema & context management
    ├─ Week 2-3: Basic branch operations
    ├─ Week 3-4: Branch comparison
    ├─ Week 4-6: Merge strategies & conflict resolution
    └─ Week 6: Tool interface & testing

Phase 3: Auto-Cleanup [2-3 weeks]
    ├─ Week 1: Policy engine & soft delete
    ├─ Week 2: Archive system
    └─ Week 3: Testing & safety verification

Phase 4: Confidence Propagation [2-3 weeks]
    ├─ Week 1: Flag & revalidation system
    ├─ Week 2: Algorithm implementation
    └─ Week 3: Integration with consolidation

Phase 5: Epistemic Metrics [1-2 weeks]
    ├─ Week 1: Metric collection
    └─ Week 2: Reporting & visualization

Phase 6: Cross-Layer Versioning [3-4 weeks]
    ├─ Week 1-2: Pattern/Theory/Belief versioning
    ├─ Week 2-3: Consolidation v2
    └─ Week 3-4: End-to-end testing
```

---

## Risk Analysis

### Technical Risks

**Risk 1: Complexity Explosion**
- Severity: HIGH
- Probability: MEDIUM
- Mitigation: Incremental rollout, extensive testing, simplify where possible

**Risk 2: Performance Degradation**
- Severity: MEDIUM
- Probability: MEDIUM
- Mitigation: Indexing strategy, query optimization, lazy evaluation

**Risk 3: Data Corruption**
- Severity: HIGH
- Probability: LOW
- Mitigation: Comprehensive backups, migration testing, rollback procedures

### Conceptual Risks

**Risk 1: Model Confusion**
- Problem: Too many features, model doesn't understand when to use each
- Mitigation: Clear documentation, examples, gradual feature introduction

**Risk 2: Branch Proliferation**
- Problem: Model creates too many branches, overwhelms system
- Mitigation: Branch limits, automatic cleanup, priority system

**Risk 3: Over-Correction**
- Problem: Model constantly corrects itself, loses confidence
- Mitigation: Metrics monitoring, confidence thresholds, correction rate limits

---

## Success Criteria (Overall)

### By End of Phase 2
- ✅ Model can create branches for hypotheses
- ✅ Model tests hypotheses in isolation
- ✅ Model compares evidence across branches
- ✅ Model merges confirmed hypotheses

### By End of Phase 3
- ✅ Automatic cleanup running smoothly
- ✅ Storage growth controlled (<10% per month)
- ✅ No accidental data loss
- ✅ Performance maintained

### By End of Phase 4
- ✅ Pattern confidence reflects evidence quality
- ✅ Obsolete observations trigger revalidation
- ✅ Consolidation revalidates flagged patterns
- ✅ Confidence propagates correctly

### By End of Phase 5
- ✅ Epistemic metrics tracked automatically
- ✅ Learning trends visible
- ✅ Error patterns identified
- ✅ Reports generated regularly

### By End of Phase 6
- ✅ All layers support versioning
- ✅ Belief evolution tracked
- ✅ Cross-layer propagation working
- ✅ Full epistemic history available

---

## Conclusion

This roadmap transforms the memory system from a simple observation store into a sophisticated epistemic reasoning engine. Each phase adds capabilities that enable the model to:

1. **Think scientifically** (branching for hypothesis testing)
2. **Manage knowledge efficiently** (auto-cleanup)
3. **Reason about uncertainty** (confidence propagation)
4. **Learn from mistakes** (epistemic metrics)
5. **Track belief evolution** (cross-layer versioning)

The ultimate goal: An AGI that maintains epistemic humility, learns from its mistakes, and continuously refines its understanding of itself and the world.

**Next immediate step:** Begin Phase 2 design and prototyping.
