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
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
from enum import Enum
import sqlite3
from datetime import datetime


class ObservationType(Enum):
    """Types of observations that can be recorded."""
    MODIFICATION = "modification"
    PERFORMANCE = "performance"
    SAFETY_EVENT = "safety_event"
    INTROSPECTION = "introspection"
    USER_INTERACTION = "user_interaction"
    CHECKPOINT = "checkpoint"
    SYSTEM_EVENT = "system_event"


@dataclass
class Observation:
    """A single observation record."""
    id: str
    timestamp: float
    type: ObservationType
    category: str
    description: str
    data: Dict[str, Any]
    tags: List[str]
    importance: float  # 0.0-1.0 scale

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


class ObservationLayer:
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
        >>> layer = ObservationLayer("data/memory/observations")
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

    def __init__(self, storage_dir: str):
        """
        Initialize observation layer.

        Args:
            storage_dir: Directory for storing observations
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        # SQLite database for efficient queries
        self.db_path = self.storage_dir / "observations.db"
        self.conn = sqlite3.connect(str(self.db_path))
        self._init_database()

        # In-memory cache for recent observations (ID -> Observation)
        self.cache: Dict[str, Observation] = {}
        self.cache_size = 1000

    def _init_database(self):
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
                importance REAL NOT NULL
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
        data: Dict[str, Any],
        tags: List[str],
        importance: float = 0.5
    ) -> str:
        """
        Record a new observation.

        Args:
            obs_type: Type of observation
            category: Category within type
            description: Human-readable description
            data: Structured data about the observation
            tags: Optional tags for organization
            importance: Importance score (0.0-1.0)

        Returns:
            Observation ID
        """
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
            (id, timestamp, type, category, description, data, tags, importance)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            observation.id,
            observation.timestamp,
            observation.type.value,
            observation.category,
            observation.description,
            json.dumps(observation.data),
            json.dumps(observation.tags),
            observation.importance
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

    def get(self, observation_id: str) -> Optional[Observation]:
        """
        Get a specific observation by ID.

        Args:
            observation_id: The observation ID

        Returns:
            Observation if found, None otherwise
        """
        # Check cache first
        if observation_id in self.cache:
            return self.cache[observation_id]

        # Query database
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT id, timestamp, type, category, description, data, tags, importance
            FROM observations
            WHERE id = ?
        """, (observation_id,))

        row = cursor.fetchone()
        if not row:
            return None

        return self._row_to_observation(row)

    def get_recent(self, limit: int = 100) -> List[Observation]:
        """
        Get most recent observations.

        Args:
            limit: Maximum number to return

        Returns:
            List of observations, newest first
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT id, timestamp, type, category, description, data, tags, importance
            FROM observations
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
        limit: Optional[int] = None
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

        Returns:
            List of matching observations
        """
        # Build query
        query_parts = ["SELECT DISTINCT o.id, o.timestamp, o.type, o.category, o.description, o.data, o.tags, o.importance FROM observations o"]
        where_parts = []
        params = []

        if tags:
            query_parts.append("JOIN observation_tags ot ON o.id = ot.observation_id")

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
        cursor = self.conn.cursor()

        # Total count
        cursor.execute("SELECT COUNT(*) FROM observations")
        total = cursor.fetchone()[0]

        # Count by type
        cursor.execute("""
            SELECT type, COUNT(*)
            FROM observations
            GROUP BY type
        """)
        by_type = {row[0]: row[1] for row in cursor.fetchall()}

        # Count by category
        cursor.execute("""
            SELECT category, COUNT(*)
            FROM observations
            GROUP BY category
            ORDER BY COUNT(*) DESC
            LIMIT 10
        """)
        top_categories = {row[0]: row[1] for row in cursor.fetchall()}

        # Time range
        cursor.execute("""
            SELECT MIN(timestamp), MAX(timestamp)
            FROM observations
        """)
        time_range = cursor.fetchone()

        # Average importance
        cursor.execute("SELECT AVG(importance) FROM observations")
        avg_importance = cursor.fetchone()[0] or 0.0

        return {
            'total': total,
            'by_type': by_type,
            'top_categories': top_categories,
            'time_range': {
                'start': time_range[0],
                'end': time_range[1],
                'span_hours': (time_range[1] - time_range[0]) / 3600 if time_range[0] else 0
            },
            'average_importance': avg_importance
        }

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
                    type=ObservationType.SYSTEM_EVENT,
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

    def export(self, filepath: str, export_format: str = 'json'):
        """
        Export observations to file.

        Args:
            filepath: Output file path
            format: Export format ('json' or 'csv')
        """
        observations = self.get_recent(limit=10000)  # Get all recent

        if export_format == 'json':
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump([obs.to_dict() for obs in observations], f, indent=2)
        elif export_format == 'csv':
            import csv
            with open(filepath, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['id', 'timestamp', 'type', 'category', 'description', 'importance'])
                for obs in observations:
                    writer.writerow([
                        obs.id,
                        obs.timestamp,
                        obs.type.value,
                        obs.category,
                        obs.description,
                        obs.importance
                    ])

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
            importance=row[7]
        )

    def __del__(self):
        """Cleanup database connection."""
        if hasattr(self, 'conn'):
            self.conn.close()
