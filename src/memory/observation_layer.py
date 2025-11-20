"""
Observation Layer (Layer 1) - Direct Observations

This is the foundation of the memory system. It records raw events, measurements,
and outcomes from all operations.

Observations are:
- Timestamped events with full context
- Immutable (never modified after recording)
- Searchable and filterable
- Automatically consolidated over time

Types of observations:
- Modifications (weight changes, architecture changes)
- Performance metrics (benchmarks, inference times)
- Safety events (alerts, emergency stops)
- Introspection results (weight statistics, activation patterns)
- User interactions (commands, feedback)

Author: AGI Self-Modification Research Team
Date: November 7, 2025
"""

import json
import time
import uuid
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
from enum import Enum
import sqlite3
from datetime import datetime

from .base_layer import SQLiteLayerBase

logger = logging.getLogger(__name__)


class ObservationType(Enum):
    """Types of observations that can be recorded."""
    MODIFICATION = "modification"
    PERFORMANCE = "performance"
    SAFETY_EVENT = "safety_event"
    INTROSPECTION = "introspection"
    USER_INTERACTION = "user_interaction"
    CHECKPOINT = "checkpoint"
    SYSTEM_EVENT = "system_event"
    HYPOTHESIS = "hypothesis"
    BEHAVIOR = "behavior"
    DISCOVERY = "discovery"


@dataclass
class Observation:
    """A single observation record with versioning and lifecycle management."""
    id: str
    timestamp: float
    type: ObservationType
    category: str
    description: str
    data: Dict[str, Any]
    tags: List[str]
    importance: float  # 0.0-1.0 scale
    
    # Versioning and lifecycle (Phase 1: Correction & Versioning)
    status: str = "active"  # active | obsolete | deprecated | superseded
    version: int = 1  # Version number for tracking revisions
    replaced_by: Optional[str] = None  # ID of newer observation that replaces this
    corrects: Optional[str] = None  # ID of observation being corrected
    obsolete_reason: Optional[str] = None  # Why this was obsoleted/deprecated
    updated_at: Optional[float] = None  # When this version was created (for updates)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        d = asdict(self)
        d['type'] = self.type.value
        d['timestamp_human'] = datetime.fromtimestamp(self.timestamp).isoformat()
        return d

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'Observation':
        """Create from dictionary."""
        data['type'] = ObservationType(data['type'])
        return Observation(**data)


class ObservationLayer(SQLiteLayerBase):
    """
    Layer 1: Direct Observations

    Records all events and measurements as they happen. Provides efficient
    storage, retrieval, and querying of observation history.

    Features:
    - SQLite database for efficient queries
    - JSON export for portability
    - Automatic consolidation (summarization of old data)
    - Importance-based retention (keep important, forget trivial)
    - Tag-based organization

    Usage:
        >>> layer = ObservationLayer("data/memory/observations.db")
        >>>
        >>> # Record an observation
        >>> layer.record(
        ...     type=ObservationType.MODIFICATION,
        ...     category="weight_change",
        ...     description="Increased attention weights in layer 5",
        ...     data={'layer': 5, 'delta': 0.01, 'parameter_count': 1024},
        ...     tags=['attention', 'layer5'],
        ...     importance=0.8
        ... )
        >>>
        >>> # Query observations
        >>> recent = layer.get_recent(limit=10)
        >>> modifications = layer.query(type=ObservationType.MODIFICATION)
        >>> layer5_events = layer.query(tags=['layer5'])
    """
    
    # Column list for SELECT statements (including Phase 1 fields)
    SELECT_COLUMNS = "id, timestamp, type, category, description, data, tags, importance, status, version, replaced_by, corrects, obsolete_reason, updated_at"

    def __init__(self, db_path: str):
        """
        Initialize observation layer.

        Args:
            db_path: Path to the observations database file
        """
        # In-memory cache for recent observations (ID -> Observation)
        self.cache: Dict[str, Observation] = {}
        self.cache_size = 1000
        
        # Initialize base class (establishes DB connection)
        super().__init__(db_path)

    def _get_table_name(self) -> str:
        """Return the main table name for this layer."""
        return "observations"
    
    def _init_schema(self):
        """Initialize database schema."""
        cursor = self.conn.cursor()

        # Main observations table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS observations (
                id TEXT PRIMARY KEY,
                timestamp REAL NOT NULL,
                type TEXT NOT NULL,
                category TEXT NOT NULL,
                description TEXT NOT NULL,
                data TEXT NOT NULL,
                tags TEXT NOT NULL,
                importance REAL NOT NULL,
                
                -- Phase 1: Versioning and lifecycle management
                status TEXT DEFAULT 'active',
                version INTEGER DEFAULT 1,
                replaced_by TEXT,
                corrects TEXT,
                obsolete_reason TEXT,
                updated_at REAL
            )
        """)

        # Create indexes separately (SQLite syntax)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp ON observations(timestamp)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_type ON observations(type)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_category ON observations(category)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_importance ON observations(importance)
        """)

        # Tags table for efficient tag queries
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS observation_tags (
                observation_id TEXT NOT NULL,
                tag TEXT NOT NULL,
                FOREIGN KEY (observation_id) REFERENCES observations(id)
            )
        """)

        # Index for tag queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_tag ON observation_tags(tag)
        """)

        self.conn.commit()

    def record(
        self,
        obs_type: ObservationType,
        category: str,
        description: str,
        importance: float,
        data: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None
    ) -> str:
        """
        Record a new observation.

        Args:
            obs_type: Type of observation
            category: Category within type
            description: Human-readable description
            importance: Importance score (0.0-1.0)
            data: Structured data about the observation (optional, defaults to {})
            tags: Tags for organization (optional, defaults to [])

        Returns:
            Observation ID
        """
        data = data or {}
        tags = tags or []

        # Generate unique ID using UUID to avoid collisions
        obs_id = f"obs_{uuid.uuid4().hex[:12]}"

        # Create observation
        observation = Observation(
            id=obs_id,
            timestamp=time.time(),
            type=obs_type,
            category=category,
            description=description,
            data=data,
            tags=tags,
            importance=importance
        )

        # Store in database
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO observations
            (id, timestamp, type, category, description, data, tags, importance, 
             status, version, replaced_by, corrects, obsolete_reason, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            observation.id,
            observation.timestamp,
            observation.type.value,
            observation.category,
            observation.description,
            json.dumps(observation.data),
            json.dumps(observation.tags),
            observation.importance,
            observation.status,
            observation.version,
            observation.replaced_by,
            observation.corrects,
            observation.obsolete_reason,
            observation.updated_at
        ))

        # Store tags
        for tag in tags:
            cursor.execute("""
                INSERT INTO observation_tags (observation_id, tag)
                VALUES (?, ?)
            """, (observation.id, tag))

        self.conn.commit()

        # Add to cache
        self.cache[obs_id] = observation
        if len(self.cache) > self.cache_size:
            # Remove oldest entry (first key)
            oldest_id = next(iter(self.cache))
            del self.cache[oldest_id]

        return obs_id

    def get(self, observation_id: str, include_obsolete: bool = False) -> Optional[Observation]:
        """
        Get a specific observation by ID.

        Args:
            observation_id: The observation ID
            include_obsolete: If False (default), return None for obsolete observations

        Returns:
            Observation if found and not obsolete (unless include_obsolete=True), None otherwise
        """
        # Check cache first
        if observation_id in self.cache:
            obs = self.cache[observation_id]
            if not include_obsolete and obs.status != 'active':
                return None
            return obs

        # Query database
        cursor = self.conn.cursor()
        
        status_clause = "" if include_obsolete else "AND status = 'active'"
        
        cursor.execute(f"""
            SELECT {self.SELECT_COLUMNS}
            FROM observations
            WHERE id = ? {status_clause}
        """, (observation_id,))

        row = cursor.fetchone()
        if not row:
            return None

        return self._row_to_observation(row)

    def get_recent(self, limit: int = 100, include_obsolete: bool = False) -> List[Observation]:
        """
        Get most recent observations.

        Args:
            limit: Maximum number to return
            include_obsolete: If False (default), exclude obsolete observations

        Returns:
            List of observations, newest first
        """
        cursor = self.conn.cursor()
        
        status_clause = "" if include_obsolete else "WHERE status = 'active'"
        
        cursor.execute(f"""
            SELECT {self.SELECT_COLUMNS}
            FROM observations
            {status_clause}
            ORDER BY timestamp DESC
            LIMIT ?
        """, (limit,))

        return [self._row_to_observation(row) for row in cursor.fetchall()]

    def query(
        self,
        obs_type: Optional[ObservationType] = None,
        category: Optional[str] = None,
        tags: Optional[List[str]] = None,
        min_importance: Optional[float] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        limit: Optional[int] = None,
        include_obsolete: bool = False
    ) -> List[Observation]:
        """
        Query observations with filters.

        Args:
            type: Filter by observation type
            category: Filter by category
            tags: Filter by tags (must have all)
            min_importance: Minimum importance score
            start_time: Start timestamp
            end_time: End timestamp
            limit: Maximum results
            include_obsolete: If False (default), exclude obsolete observations

        Returns:
            List of matching observations
        """
        # Build query with Phase 1 columns
        query_parts = [f"SELECT DISTINCT {', '.join([f'o.{col}' for col in self.SELECT_COLUMNS.split(', ')])} FROM observations o"]
        where_parts = []
        params = []

        if tags:
            query_parts.append("JOIN observation_tags ot ON o.id = ot.observation_id")

        # Add status filter by default
        if not include_obsolete:
            where_parts.append("o.status = 'active'")

        if obs_type:
            where_parts.append("o.type = ?")
            params.append(obs_type.value)

        if category:
            where_parts.append("o.category = ?")
            params.append(category)

        if tags:
            placeholders = ','.join(['?'] * len(tags))
            where_parts.append(f"ot.tag IN ({placeholders})")
            params.extend(tags)

        if min_importance is not None:
            where_parts.append("o.importance >= ?")
            params.append(min_importance)

        if start_time is not None:
            where_parts.append("o.timestamp >= ?")
            params.append(start_time)

        if end_time is not None:
            where_parts.append("o.timestamp <= ?")
            params.append(end_time)

        if where_parts:
            query_parts.append("WHERE " + " AND ".join(where_parts))

        query_parts.append("ORDER BY o.timestamp DESC")

        if limit:
            query_parts.append(f"LIMIT {limit}")

        query = " ".join(query_parts)

        cursor = self.conn.cursor()
        cursor.execute(query, params)

        return [self._row_to_observation(row) for row in cursor.fetchall()]

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about observations.

        Returns:
            Dictionary with statistics
        """
        # Get base statistics from parent class
        stats = super().get_statistics()
        
        # Add observation-specific statistics
        cursor = self.conn.cursor()
        
        # Count by type
        stats['by_type'] = self._get_grouped_counts('type')
        
        # Top 10 categories
        cursor.execute("""
            SELECT category, COUNT(*)
            FROM observations
            GROUP BY category
            ORDER BY COUNT(*) DESC
            LIMIT 10
        """)
        stats['top_categories'] = {row[0]: row[1] for row in cursor.fetchall()}
        
        # Average importance
        stats['average_importance'] = self._get_average_value('importance')
        
        return stats

    def consolidate(self, older_than_days: int = 30, importance_threshold: float = 0.3):
        """
        Consolidate old, low-importance observations.

        Creates summaries of old observations and removes the detailed records
        to save space while preserving important information.

        Args:
            older_than_days: Consolidate observations older than this
            importance_threshold: Only consolidate if importance below this
        """
        cutoff_time = time.time() - (older_than_days * 24 * 3600)

        cursor = self.conn.cursor()

        # Get observations to consolidate
        cursor.execute("""
            SELECT COUNT(*), type, category
            FROM observations
            WHERE timestamp < ? AND importance < ?
            GROUP BY type, category
        """, (cutoff_time, importance_threshold))

        summaries = cursor.fetchall()

        # Create summary observations
        for count, obs_type, category in summaries:
            if count > 10:  # Only consolidate if there are many
                self.record(
                    obs_type=ObservationType.SYSTEM_EVENT,
                    category="consolidation",
                    description=f"Consolidated {count} {obs_type}/{category} observations",
                    data={
                        'consolidated_count': count,
                        'original_type': obs_type,
                        'original_category': category,
                        'cutoff_time': cutoff_time
                    },
                    tags=['consolidation'],
                    importance=0.4
                )

        # Delete consolidated observations
        cursor.execute("""
            DELETE FROM observations
            WHERE timestamp < ? AND importance < ?
        """, (cutoff_time, importance_threshold))

        # Clean up orphaned tags
        cursor.execute("""
            DELETE FROM observation_tags
            WHERE observation_id NOT IN (SELECT id FROM observations)
        """)

        self.conn.commit()

        # Clear cache
        self.cache = {}

    def export(self, filepath: str, limit: Optional[int] = None, export_format: str = 'json'):
        """
        Export observations to file.

        Args:
            filepath: Output file path
            limit: Maximum number of observations to export (default: None for all)
            export_format: Export format ('json' or 'csv')
        """
        observations = self.get_recent(limit=limit if limit else 10000)

        if export_format == 'json':
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump([obs.to_dict() for obs in observations], f, indent=2)
        elif export_format == 'csv':
            import csv
            with open(filepath, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'id', 'timestamp', 'type', 'category', 'description', 'importance',
                    'status', 'version', 'replaced_by', 'corrects', 'obsolete_reason', 'updated_at'
                ])
                for obs in observations:
                    writer.writerow([
                        obs.id,
                        obs.timestamp,
                        obs.type.value,
                        obs.category,
                        obs.description,
                        obs.importance,
                        obs.status,
                        obs.version,
                        obs.replaced_by,
                        obs.corrects,
                        obs.obsolete_reason,
                        obs.updated_at
                    ])

    # ===== Phase 1: Lifecycle Management Methods =====

    def update_observation(
        self,
        observation_id: str,
        updates: Dict[str, Any],
        reason: str
    ) -> str:
        """
        Create a new version of an observation with updates.
        
        The original observation is marked as 'superseded' and a new version
        is created with the updates applied. This preserves the full history.
        
        Args:
            observation_id: ID of observation to update
            updates: Dictionary of fields to update (e.g., {'importance': 0.9, 'description': '...'})
            reason: Human-readable reason for the update
            
        Returns:
            ID of the new observation version
            
        Example:
            >>> new_id = layer.update_observation(
            ...     "obs_abc123",
            ...     {"importance": 0.9, "description": "Corrected importance"},
            ...     "Realized this observation is more important than initially thought"
            ... )
        """
        # Get original observation
        original = self.get(observation_id)
        if not original:
            raise ValueError(f"Observation {observation_id} not found")
        
        if original.status != "active":
            raise ValueError(f"Cannot update observation {observation_id} with status '{original.status}'")
        
        # Create new version with updates
        new_data = original.data.copy()
        new_tags = original.tags.copy()
        new_description = original.description
        new_importance = original.importance
        new_category = original.category
        
        # Apply updates
        if 'data' in updates:
            new_data.update(updates['data'])
        if 'tags' in updates:
            new_tags = updates['tags']
        if 'description' in updates:
            new_description = updates['description']
        if 'importance' in updates:
            new_importance = updates['importance']
        if 'category' in updates:
            new_category = updates['category']
        
        # Record new version
        new_id = self.record(
            obs_type=original.type,
            category=new_category,
            description=new_description,
            data=new_data,
            tags=new_tags,
            importance=new_importance
        )
        
        # Update new observation's metadata
        cursor = self.conn.cursor()
        cursor.execute("""
            UPDATE observations
            SET version = ?, corrects = ?, updated_at = ?
            WHERE id = ?
        """, (original.version + 1, observation_id, time.time(), new_id))
        
        # Mark original as superseded
        cursor.execute("""
            UPDATE observations
            SET status = 'superseded', replaced_by = ?, obsolete_reason = ?
            WHERE id = ?
        """, (new_id, reason, observation_id))
        
        self.conn.commit()
        
        # Update cache
        if observation_id in self.cache:
            del self.cache[observation_id]
        
        logger.info(f"Updated observation {observation_id} → {new_id} (v{original.version + 1}): {reason}")
        return new_id

    def correct_observation(
        self,
        observation_id: str,
        correction_description: str,
        corrected_data: Optional[Dict[str, Any]] = None,
        reason: str = ""
    ) -> str:
        """
        Mark an observation as incorrect and create a corrected version.
        
        This is semantically stronger than update - it means "this was wrong".
        The original is marked 'obsolete' (not just 'superseded').
        
        Args:
            observation_id: ID of incorrect observation
            correction_description: Corrected description
            corrected_data: Corrected data (optional)
            reason: Why the original was incorrect
            
        Returns:
            ID of the corrected observation
            
        Example:
            >>> corrected_id = layer.correct_observation(
            ...     "obs_abc123",
            ...     "Layer 5 actually has LOW importance (not high)",
            ...     {"layer": 5, "importance_level": "low"},
            ...     "Initial analysis was based on flawed assumption"
            ... )
        """
        # Get original observation
        original = self.get(observation_id)
        if not original:
            raise ValueError(f"Observation {observation_id} not found")
        
        # Create corrected observation
        corrected_id = self.record(
            obs_type=original.type,
            category=original.category,
            description=correction_description,
            data=corrected_data if corrected_data is not None else original.data,
            tags=original.tags + ["correction"],
            importance=original.importance
        )
        
        # Update metadata
        cursor = self.conn.cursor()
        cursor.execute("""
            UPDATE observations
            SET corrects = ?, version = ?, updated_at = ?
            WHERE id = ?
        """, (observation_id, original.version + 1, time.time(), corrected_id))
        
        # Mark original as obsolete
        cursor.execute("""
            UPDATE observations
            SET status = 'obsolete', replaced_by = ?, obsolete_reason = ?
            WHERE id = ?
        """, (corrected_id, reason or "Corrected by newer observation", observation_id))
        
        self.conn.commit()
        
        # Update cache
        if observation_id in self.cache:
            del self.cache[observation_id]
        
        logger.info(f"Corrected observation {observation_id} → {corrected_id}: {reason}")
        return corrected_id

    def obsolete_observation(
        self,
        observation_id: str,
        reason: str,
        cascade: bool = False
    ) -> Dict[str, Any]:
        """
        Mark an observation as no longer valid without creating a replacement.
        
        Use this when you determine an observation was simply wrong or
        no longer applicable (vs. update/correct which create new versions).
        
        Args:
            observation_id: ID of observation to obsolete
            reason: Why this observation is being obsoleted
            cascade: If True, mark dependent patterns for revalidation
            
        Returns:
            Dictionary with statistics on affected patterns/theories/beliefs
            
        Example:
            >>> stats = layer.obsolete_observation(
            ...     "obs_abc123",
            ...     "This was a duplicate observation",
            ...     cascade=True
            ... )
            >>> print(f"Affected {stats['patterns_flagged']} patterns")
        """
        # Get observation
        obs = self.get(observation_id)
        if not obs:
            raise ValueError(f"Observation {observation_id} not found")
        
        # Mark as obsolete
        cursor = self.conn.cursor()
        cursor.execute("""
            UPDATE observations
            SET status = 'obsolete', obsolete_reason = ?, updated_at = ?
            WHERE id = ?
        """, (reason, time.time(), observation_id))
        
        self.conn.commit()
        
        # Update cache
        if observation_id in self.cache:
            del self.cache[observation_id]
        
        stats = {
            "observation_id": observation_id,
            "status": "obsolete",
            "reason": reason,
            "patterns_flagged": 0,
            "cascade_enabled": cascade
        }
        
        # If cascade is enabled, we'd mark dependent patterns for revalidation
        # This would require pattern layer integration - leaving as placeholder
        if cascade:
            logger.info(f"Cascade revalidation requested for obs {observation_id} (not yet implemented)")
            stats["cascade_note"] = "Cascade flagging not yet implemented - will be done during consolidation"
        
        logger.info(f"Obsoleted observation {observation_id}: {reason}")
        return stats

    def _row_to_observation(self, row: tuple) -> Observation:
        """Convert database row to Observation object."""
        return Observation(
            id=row[0],
            timestamp=row[1],
            type=ObservationType(row[2]),
            category=row[3],
            description=row[4],
            data=json.loads(row[5]),
            tags=json.loads(row[6]),
            importance=row[7],
            # Phase 1 fields (with defaults for backward compatibility)
            status=row[8] if len(row) > 8 else "active",
            version=row[9] if len(row) > 9 else 1,
            replaced_by=row[10] if len(row) > 10 else None,
            corrects=row[11] if len(row) > 11 else None,
            obsolete_reason=row[12] if len(row) > 12 else None,
            updated_at=row[13] if len(row) > 13 else None
        )
