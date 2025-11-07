"""
Migrate memory layers from JSON to SQLite

This script converts the pattern_layer, theory_layer, and belief_layer from
JSON-based storage to SQLite for consistency and better performance.

Usage:
    python scripts/migrate_memory_to_sqlite.py [--backup]

Author: AGI Self-Modification Research
Date: November 7, 2025
"""

import sqlite3
import json
import shutil
from pathlib import Path
import argparse


def migrate_pattern_layer_complete():
    """Pattern layer is already migrated through code changes."""
    print("✓ Pattern layer: Already migrated to SQLite in code")
    return True


def migrate_theory_layer(storage_dir: str, backup: bool = True):
    """
    Migrate theory_layer from theories.json to theories.db
    
    Args:
        storage_dir: Path to theories storage directory
        backup: Whether to backup JSON file before migration
    """
    storage_path = Path(storage_dir)
    json_file = storage_path / "theories.json"
    db_file = storage_path / "theories.db"
    
    if not json_file.exists():
        print(f"  No theories.json found at {json_file}, skipping")
        return True
    
    # Backup if requested
    if backup:
        backup_file = storage_path / f"theories.json.backup"
        shutil.copy(json_file, backup_file)
        print(f"  ✓ Backed up to {backup_file}")
    
    # Load JSON data
    with open(json_file, 'r') as f:
        theories_dict = json.load(f)
    
    # Create SQLite database
    conn = sqlite3.connect(str(db_file))
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS theories (
            id TEXT PRIMARY KEY,
            type TEXT NOT NULL,
            name TEXT NOT NULL,
            description TEXT NOT NULL,
            hypothesis TEXT NOT NULL,
            supporting_evidence TEXT NOT NULL,
            confidence REAL NOT NULL,
            testable INTEGER NOT NULL,
            tested INTEGER NOT NULL,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL,
            tags TEXT NOT NULL,
            metadata TEXT
        )
    """)
    
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_theories_confidence ON theories(confidence)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_theories_type ON theories(type)")
    
    # Migrate data
    for theory_id, theory_data in theories_dict.items():
        cursor.execute("""
            INSERT INTO theories
            (id, type, name, description, hypothesis, supporting_evidence, confidence,
             testable, tested, created_at, updated_at, tags, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            theory_data['id'],
            theory_data['type'],
            theory_data['name'],
            theory_data['description'],
            theory_data['hypothesis'],
            json.dumps(theory_data['supporting_evidence']),
            theory_data['confidence'],
            1 if theory_data.get('testable', True) else 0,
            1 if theory_data.get('tested', False) else 0,
            theory_data.get('created_at', 0.0),
            theory_data.get('updated_at', 0.0),
            json.dumps(theory_data.get('tags', [])),
            json.dumps(theory_data.get('metadata', {}))
        ))
    
    conn.commit()
    conn.close()
    
    print(f"  ✓ Migrated {len(theories_dict)} theories to SQLite")
    return True


def migrate_belief_layer(storage_dir: str, backup: bool = True):
    """
    Migrate belief_layer from beliefs.json to beliefs.db
    
    Args:
        storage_dir: Path to beliefs storage directory
        backup: Whether to backup JSON file before migration
    """
    storage_path = Path(storage_dir)
    json_file = storage_path / "beliefs.json"
    db_file = storage_path / "beliefs.db"
    
    if not json_file.exists():
        print(f"  No beliefs.json found at {json_file}, skipping")
        return True
    
    # Backup if requested
    if backup:
        backup_file = storage_path / f"beliefs.json.backup"
        shutil.copy(json_file, backup_file)
        print(f"  ✓ Backed up to {backup_file}")
    
    # Load JSON data
    with open(json_file, 'r') as f:
        beliefs_dict = json.load(f)
    
    # Create SQLite database
    conn = sqlite3.connect(str(db_file))
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS beliefs (
            id TEXT PRIMARY KEY,
            content TEXT NOT NULL,
            confidence REAL NOT NULL,
            evidence TEXT NOT NULL,
            category TEXT,
            formed_at REAL NOT NULL,
            tags TEXT NOT NULL
        )
    """)
    
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_beliefs_confidence ON beliefs(confidence)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_beliefs_category ON beliefs(category)")
    
    # Migrate data
    for belief_id, belief_data in beliefs_dict.items():
        cursor.execute("""
            INSERT INTO beliefs
            (id, content, confidence, evidence, category, formed_at, tags)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            belief_data['id'],
            belief_data['content'],
            belief_data['confidence'],
            json.dumps(belief_data['evidence']),
            belief_data.get('category'),
            belief_data.get('formed_at', 0.0),
            json.dumps(belief_data.get('tags', []))
        ))
    
    conn.commit()
    conn.close()
    
    print(f"  ✓ Migrated {len(beliefs_dict)} beliefs to SQLite")
    return True


def main():
    parser = argparse.ArgumentParser(description='Migrate memory layers to SQLite')
    parser.add_argument('--no-backup', action='store_true', help='Skip backup of JSON files')
    parser.add_argument('--memory-dir', default='data/memory_demo', help='Base memory directory')
    args = parser.parse_args()
    
    backup = not args.no_backup
    base_dir = Path(args.memory_dir)
    
    print("=" * 60)
    print("Memory Layer Migration: JSON → SQLite")
    print("=" * 60)
    print()
    
    # Check if base directory exists
    if not base_dir.exists():
        print(f"❌ Memory directory not found: {base_dir}")
        print(f"   Please specify correct path with --memory-dir")
        return 1
    
    print(f"Base directory: {base_dir}")
    print(f"Backup enabled: {backup}")
    print()
    
    # Migrate each layer
    print("[1/3] Pattern Layer")
    migrate_pattern_layer_complete()
    print()
    
    print("[2/3] Theory Layer")
    theory_dir = base_dir / "theories"
    if theory_dir.exists():
        migrate_theory_layer(str(theory_dir), backup=backup)
    else:
        print(f"  No theory directory found, skipping")
    print()
    
    print("[3/3] Belief Layer")
    belief_dir = base_dir / "beliefs"
    if belief_dir.exists():
        migrate_belief_layer(str(belief_dir), backup=backup)
    else:
        print(f"  No belief directory found, skipping")
    print()
    
    print("=" * 60)
    print("✅ Migration Complete!")
    print("=" * 60)
    print()
    print("Next steps:")
    print("1. Run tests to verify migration")
    print("2. If tests pass, you can delete .json.backup files")
    print("3. Update code to use SQLite for theory/belief layers")
    print()
    
    return 0


if __name__ == "__main__":
    exit(main())
