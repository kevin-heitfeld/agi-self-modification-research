"""
Base Layer for SQLite-backed Memory Layers

Provides common functionality for observation, pattern, theory, and belief layers
to reduce code duplication and ensure consistency.

Author: AGI Self-Modification Research Team
Date: November 20, 2025
"""

import sqlite3
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from abc import ABC, abstractmethod


class SQLiteLayerBase(ABC):
    """
    Base class for SQLite-backed memory layers.
    
    Provides common database operations:
    - Connection management
    - Statistics gathering
    - Database initialization
    - Cleanup
    
    Subclasses must implement:
    - _get_table_name(): Return the main table name
    - _init_schema(): Create table schema
    """
    
    def __init__(self, db_path: str):
        """
        Initialize layer with database connection.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        
        # Initialize schema (subclass responsibility)
        self._init_schema()
    
    @abstractmethod
    def _get_table_name(self) -> str:
        """Return the main table name for this layer."""
        pass
    
    @abstractmethod
    def _init_schema(self):
        """Initialize database schema (tables and indexes)."""
        pass
    
    def close(self):
        """Close database connection."""
        if hasattr(self, 'conn') and self.conn:
            self.conn.close()
    
    def __del__(self):
        """Cleanup database connection."""
        self.close()
    
    # ===== Common Statistics Methods =====
    
    def _get_total_count(self, table_name: Optional[str] = None) -> int:
        """
        Get total count of records in table.
        
        Args:
            table_name: Table to count (defaults to main table)
            
        Returns:
            Total number of records
        """
        if table_name is None:
            table_name = self._get_table_name()
        
        cursor = self.conn.cursor()
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        return cursor.fetchone()[0]
    
    def _get_grouped_counts(
        self,
        field_name: str,
        table_name: Optional[str] = None,
        limit: Optional[int] = None,
        order_by: str = "count"
    ) -> Dict[str, int]:
        """
        Get counts grouped by a field.
        
        Args:
            field_name: Field to group by
            table_name: Table to query (defaults to main table)
            limit: Maximum number of groups to return
            order_by: Sort by 'count' (default) or 'field'
            
        Returns:
            Dictionary mapping field values to counts
        """
        if table_name is None:
            table_name = self._get_table_name()
        
        cursor = self.conn.cursor()
        
        # Build query
        query = f"""
            SELECT {field_name}, COUNT(*) as count
            FROM {table_name}
            GROUP BY {field_name}
        """
        
        # Add ordering
        if order_by == "count":
            query += " ORDER BY count DESC"
        else:
            query += f" ORDER BY {field_name}"
        
        # Add limit
        if limit:
            query += f" LIMIT {limit}"
        
        cursor.execute(query)
        return {row[0]: row[1] for row in cursor.fetchall()}
    
    def _get_time_range(
        self,
        timestamp_field: str = "timestamp",
        table_name: Optional[str] = None
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        Get min and max timestamps.
        
        Args:
            timestamp_field: Name of timestamp field
            table_name: Table to query (defaults to main table)
            
        Returns:
            Tuple of (min_timestamp, max_timestamp)
        """
        if table_name is None:
            table_name = self._get_table_name()
        
        cursor = self.conn.cursor()
        cursor.execute(f"""
            SELECT MIN({timestamp_field}), MAX({timestamp_field})
            FROM {table_name}
        """)
        row = cursor.fetchone()
        return (row[0], row[1])
    
    def _get_average_value(
        self,
        field_name: str,
        table_name: Optional[str] = None,
        where_clause: Optional[str] = None
    ) -> Optional[float]:
        """
        Get average value of a numeric field.
        
        Args:
            field_name: Field to average
            table_name: Table to query (defaults to main table)
            where_clause: Optional WHERE clause (without the WHERE keyword)
            
        Returns:
            Average value or None if no records
        """
        if table_name is None:
            table_name = self._get_table_name()
        
        cursor = self.conn.cursor()
        query = f"SELECT AVG({field_name}) FROM {table_name}"
        
        if where_clause:
            query += f" WHERE {where_clause}"
        
        cursor.execute(query)
        result = cursor.fetchone()[0]
        return float(result) if result is not None else None
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about this layer.
        
        Default implementation provides common statistics.
        Subclasses can override to add layer-specific stats.
        
        Returns:
            Dictionary with statistics
        """
        table_name = self._get_table_name()
        
        stats = {
            'total': self._get_total_count(),
            'time_range': {
                'min': None,
                'max': None,
                'span_hours': None
            }
        }
        
        # Get time range if table has timestamp field
        cursor = self.conn.cursor()
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = [row[1] for row in cursor.fetchall()]
        
        if 'timestamp' in columns:
            min_time, max_time = self._get_time_range()
            stats['time_range']['min'] = min_time
            stats['time_range']['max'] = max_time
            if min_time and max_time:
                stats['time_range']['span_hours'] = (max_time - min_time) / 3600
        
        return stats
    
    # ===== Common Export Methods =====
    
    def export(self, filepath: str, limit: Optional[int] = None):
        """
        Export records to JSON file.
        
        Args:
            filepath: Output file path
            limit: Maximum number of records to export (None = all)
        """
        # This is a generic implementation
        # Subclasses should override with proper serialization
        cursor = self.conn.cursor()
        
        query = f"SELECT * FROM {self._get_table_name()}"
        if limit:
            query += f" LIMIT {limit}"
        
        cursor.execute(query)
        rows = cursor.fetchall()
        
        # Convert to list of dicts
        data = [dict(row) for row in rows]
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    # ===== Common Maintenance Methods =====
    
    def vacuum(self):
        """Optimize database file size."""
        self.conn.execute("VACUUM")
    
    def analyze(self):
        """Update query planner statistics."""
        self.conn.execute("ANALYZE")
    
    def get_db_size_mb(self) -> float:
        """Get database file size in MB."""
        return self.db_path.stat().st_size / (1024 * 1024) if self.db_path.exists() else 0.0
