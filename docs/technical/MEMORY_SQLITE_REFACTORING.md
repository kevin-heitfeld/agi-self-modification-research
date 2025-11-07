# Memory Layer SQLite Refactoring

**Date**: November 7, 2025  
**Status**: IN PROGRESS  
**Reason**: Consistent storage across all layers

---

## Rationale

User correctly noted that since we're already using SQLite for observations, we should use it for all layers for consistency. Benefits:

✅ **Consistency**: Same storage mechanism everywhere  
✅ **No added complexity**: SQLite already a dependency  
✅ **Better queries**: Can use SQL across all layers  
✅ **Relational**: Natural foreign keys between layers  
✅ **Future-proof**: Handles growth automatically  

---

## Changes Made

### ✅ Pattern Layer (`pattern_layer.py`)

**COMPLETE** - Converted from JSON to SQLite

- Changed from `patterns.json` to `patterns.db`
- Added SQLite initialization in `__init__`
- Updated all CRUD operations to use SQL
- Added indexes on type, confidence, support_count
- Maintains same API, just different backend

### ⏳ Theory Layer (`theory_layer.py`)

**TODO** - Needs conversion

Schema:
```sql
CREATE TABLE theories (
    id TEXT PRIMARY KEY,
    type TEXT NOT NULL,
    name TEXT NOT NULL,
    description TEXT NOT NULL,
    hypothesis TEXT NOT NULL,
    supporting_evidence TEXT NOT NULL,  -- JSON array
    confidence REAL NOT NULL,
    testable INTEGER NOT NULL,
    tested INTEGER NOT NULL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    tags TEXT NOT NULL,  -- JSON array
    metadata TEXT         -- JSON blob
);
```

### ⏳ Belief Layer (`belief_layer.py`)

**TODO** - Needs conversion

Schema:
```sql
CREATE TABLE beliefs (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    confidence REAL NOT NULL,
    evidence TEXT NOT NULL,  -- JSON array
    category TEXT,
    formed_at REAL NOT NULL,
    tags TEXT NOT NULL       -- JSON array
);
```

---

## Migration Script

Created `scripts/migrate_memory_to_sqlite.py` to convert existing data:

```bash
# Migrate with backup
python scripts/migrate_memory_to_sqlite.py

# Migrate without backup
python scripts/migrate_memory_to_sqlite.py --no-backup

# Specify custom memory directory
python scripts/migrate_memory_to_sqlite.py --memory-dir data/human_knowledge
```

---

## Next Steps

1. ✅ Pattern layer refactored
2. ⏳ Update theory_layer.py to use SQLite (similar to pattern_layer changes)
3. ⏳ Update belief_layer.py to use SQLite
4. ⏳ Update tests to expect SQLite instead of JSON
5. ⏳ Run migration script on any existing data
6. ⏳ Test everything works
7. ✅ Commit changes

---

## Files Changed

- `src/memory/pattern_layer.py` - Refactored ✅
- `src/memory/theory_layer.py` - TODO ⏳
- `src/memory/belief_layer.py` - TODO ⏳
- `scripts/migrate_memory_to_sqlite.py` - Created ✅
- Tests - TODO (update expectations) ⏳

---

## Testing Plan

After full refactoring:

```bash
# Run all memory tests
python -m pytest tests/test_*_layer.py -v

# Test migration script
python scripts/migrate_memory_to_sqlite.py --memory-dir data/memory_demo

# Verify data integrity
python -c "
from src.memory import MemorySystem
m = MemorySystem('data/memory_demo')
print(f'Patterns: {len(m.patterns.get_patterns())}')
print(f'Theories: {len(m.theories.get_theories())}')
print(f'Beliefs: {len(m.beliefs.get_beliefs())}')
"
```

---

## Status

**Current**: Pattern layer complete, migration script created  
**Remaining**: Theory and belief layers need same treatment  
**Time estimate**: 1-2 hours to complete remaining layers  
**Blocked**: No  
**Decision needed**: Continue refactoring now or defer to later?

---

**Recommendation**: Complete the refactoring now before Phase 1 experiments start, so all data is stored consistently from the beginning.
