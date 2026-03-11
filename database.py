import sqlite3
import json
import os
from pathlib import Path
from datetime import datetime

class CoffeeDatabase:
    def __init__(self, db_path="data/coffee_data.db"):
        self.db_path = db_path
        # Ensure the data directory exists
        Path(os.path.dirname(self.db_path)).mkdir(exist_ok=True)
        self._init_db()

    def _get_connection(self):
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        """Initialize the database schema."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    filename TEXT,
                    mode TEXT,
                    final_label TEXT,
                    final_confidence REAL,
                    fruit_count INTEGER DEFAULT 0,
                    unripe_count INTEGER DEFAULT 0,
                    ripe_count INTEGER DEFAULT 0,
                    overripe_count INTEGER DEFAULT 0,
                    unripe_ratio REAL DEFAULT 0.0,
                    ripe_ratio REAL DEFAULT 0.0,
                    overripe_ratio REAL DEFAULT 0.0,
                    message TEXT,
                    is_rejected BOOLEAN DEFAULT 0
                )
            """)
            conn.commit()

    def save_predictions(self, data_list):
        """Save a list of prediction records to the database."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            for data in data_list:
                # Handle nested fruit_counts if they exist (common in YOLO/Hybrid)
                counts = data.get('fruit_counts', {})
                ratios = data.get('distribution', {})
                
                # Fallback property names for legacy data support
                fc = data.get('fruit_count', counts.get('total', data.get('fruits_detected', 0)))
                uc = data.get('unripe_count', counts.get('unripe', 0))
                rc = data.get('ripe_count', counts.get('ripe', 0))
                oc = data.get('overripe_count', counts.get('overripe', 0))
                
                ur = data.get('unripe_ratio', ratios.get('unripe', 0.0))
                rr = data.get('ripe_ratio', ratios.get('ripe', 0.0))
                or_ratio = data.get('overripe_ratio', ratios.get('overripe', 0.0))
                
                # Confidence fallback
                conf = data.get('final_confidence', data.get('confidence', 0.0))

                cursor.execute("""
                    INSERT INTO predictions (
                        timestamp, filename, mode, final_label, final_confidence,
                        fruit_count, unripe_count, ripe_count, overripe_count,
                        unripe_ratio, ripe_ratio, overripe_ratio, message, is_rejected
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    data.get('timestamp', datetime.now().isoformat()),
                    data.get('filename'),
                    data.get('mode'),
                    data.get('final_label'),
                    conf,
                    fc, uc, rc, oc,
                    ur, rr, or_ratio,
                    data.get('message'),
                    1 if data.get('final_label') == 'REJECTED' else 0
                ))
            conn.commit()

    def get_history(self, limit=1000):
        """Retrieve the combined history of valid and rejected predictions."""
        with self._get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM predictions 
                ORDER BY timestamp DESC, id DESC 
                LIMIT ?
            """, (limit,))
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def get_total_count(self):
        """Get the total number of attempts stored."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM predictions")
            return cursor.fetchone()[0]

    def migrate_from_json(self, history_json_path, rejected_json_path):
        """Migrate data from legacy JSON files to SQLite."""
        total_migrated = 0
        
        for base_path in [history_json_path, rejected_json_path]:
            # Try both original and backup
            for suffix in ["", ".bak"]:
                json_path = str(base_path) + suffix
                if not os.path.exists(json_path):
                    continue
                    
                try:
                    with open(json_path, 'r') as f:
                        data_list = json.load(f)
                    
                    if not data_list:
                        continue
                        
                    self.save_predictions(data_list)
                    total_migrated += len(data_list)
                    
                    # Rename to mark as processed if it wasn't already a .bak
                    if not json_path.endswith(".bak"):
                        backup_path = json_path + ".bak"
                        if not os.path.exists(backup_path):
                            os.rename(json_path, backup_path)
                            print(f"[MIGRATION] Migrated {len(data_list)} records from {os.path.basename(json_path)}")
                except Exception as e:
                    print(f"[ERROR] Migration failed for {json_path}: {e}")
                
        return total_migrated

if __name__ == "__main__":
    # Test script and one-time migration
    db = CoffeeDatabase()
    print("Database initialized.")
    
    # Run migration if JSON exists
    migrated = db.migrate_from_json("data/predictions_history.json", "data/rejected_history.json")
    if migrated > 0:
        print(f"Successfully migrated {migrated} records to SQLite.")
    else:
        print("No legacy data found or already migrated.")
