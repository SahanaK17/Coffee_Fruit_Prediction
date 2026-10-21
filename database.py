import sqlite3
import json
import os
import pandas as pd
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
                    
                    -- Vision Metrics
                    fruit_count INTEGER DEFAULT 0,
                    ripe_count INTEGER DEFAULT 0,
                    unripe_count INTEGER DEFAULT 0,
                    overripe_count INTEGER DEFAULT 0,
                    ripeness_ratio REAL DEFAULT 0.0,
                    
                    -- Maturity Model Details
                    yolo_confidence REAL DEFAULT 0.0,
                    efficientnet_confidence REAL DEFAULT 0.0,
                    hybrid_confidence REAL DEFAULT 0.0,
                    
                    -- Yield/Environment Metrics
                    temperature REAL DEFAULT 0.0,
                    humidity REAL DEFAULT 0.0,
                    rainfall_today REAL DEFAULT 0.0,
                    rainfall_30d REAL DEFAULT 0.0,
                    rainfall_90d REAL DEFAULT 0.0,
                    avg_temp_30d REAL DEFAULT 0.0,
                    predicted_yield REAL DEFAULT 0.0,
                    ripening_days INTEGER DEFAULT 0,
                    expected_harvest_date TEXT,
                    
                    -- Soil Metrics
                    soil_type TEXT DEFAULT 'loamy',
                    water_retention TEXT DEFAULT 'medium',
                    drainage TEXT DEFAULT 'moderate',
                    soil_score REAL DEFAULT 0.7,
                    
                    message TEXT,
                    is_rejected BOOLEAN DEFAULT 0
                )
            """)
            
            # Auto-migrate/Update columns if they don't exist
            cols = {
                "rainfall_today": "REAL DEFAULT 0.0",
                "rainfall_30d": "REAL DEFAULT 0.0",
                "rainfall_90d": "REAL DEFAULT 0.0",
                "avg_temp_30d": "REAL DEFAULT 0.0",
                "ripening_days": "INTEGER DEFAULT 0",
                "expected_harvest_date": "TEXT",
                "yolo_confidence": "REAL DEFAULT 0.0",
                "efficientnet_confidence": "REAL DEFAULT 0.0",
                "hybrid_confidence": "REAL DEFAULT 0.0",
                "soil_type": "TEXT DEFAULT 'loamy'",
                "water_retention": "TEXT DEFAULT 'medium'",
                "drainage": "TEXT DEFAULT 'moderate'",
                "soil_score": "REAL DEFAULT 0.7"
            }
            
            for col, spec in cols.items():
                try:
                    cursor.execute(f"ALTER TABLE predictions ADD COLUMN {col} {spec}")
                except sqlite3.OperationalError:
                    pass
            
            conn.commit()

    def save_predictions(self, data_list):
        """Save a list of prediction records (dicts) to the database."""
        if not isinstance(data_list, list):
            data_list = [data_list]
            
        with self._get_connection() as conn:
            cursor = conn.cursor()
            for data in data_list:
                # Harmonize keys between Maturity and Yield modes
                filename = data.get('filename', data.get('image_name', ''))
                mode = data.get('mode', 'N/A')
                
                # Confidence handling
                conf = data.get('final_confidence', data.get('confidence', 0.0))
                y_conf = data.get('yolo_confidence', 0.0)
                e_conf = data.get('efficientnet_confidence', 0.0)
                h_conf = data.get('hybrid_confidence', conf if 'hybrid' in mode.lower() else 0.0)

                cursor.execute("""
                    INSERT INTO predictions (
                        timestamp, filename, mode, final_label, final_confidence,
                        fruit_count, ripe_count, unripe_count, overripe_count, ripeness_ratio,
                        yolo_confidence, efficientnet_confidence, hybrid_confidence,
                        temperature, humidity, rainfall_today, rainfall_30d, rainfall_90d,
                        avg_temp_30d, predicted_yield, ripening_days, expected_harvest_date,
                        soil_type, water_retention, drainage, soil_score,
                        message, is_rejected
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    data.get('timestamp', datetime.now().isoformat()),
                    filename,
                    mode,
                    data.get('final_label', data.get('predicted_maturity', 'N/A')),
                    conf,
                    data.get('fruit_count', data.get('total_fruits_detected', 0)),
                    data.get('ripe_count', 0),
                    data.get('unripe_count', 0),
                    data.get('overripe_count', 0),
                    data.get('ripeness_ratio', 0.0),
                    y_conf, e_conf, h_conf,
                    data.get('temperature', 0.0),
                    data.get('humidity', 0.0),
                    data.get('rainfall_today', data.get('rainfall', 0.0)),
                    data.get('rainfall_30d', 0.0),
                    data.get('rainfall_90d', 0.0),
                    data.get('avg_temp_30d', 0.0),
                    data.get('predicted_yield', 0.0),
                    data.get('ripening_days', 0),
                    data.get('expected_harvest_date', ''),
                    data.get('soil_type', 'loamy'),
                    data.get('water_retention', 'medium'),
                    data.get('drainage', 'moderate'),
                    data.get('soil_score', 0.7),
                    data.get('message', ''),
                    1 if data.get('final_label') == 'REJECTED' or data.get('predicted_maturity') == 'REJECTED' else 0
                ))
            conn.commit()

    def get_dashboard_stats(self):
        """Calculate aggregated stats for the dashboard directly from SQL."""
        stats = {
            "maturity": {
                "total_predictions": 0,
                "hybrid_avg": 0.0,
                "yolo_avg": 0.0,
                "efficientnet_avg": 0.0
            },
            "yield": {
                "total_yield_predictions": 0,
                "avg_yield": 0.0,
                "avg_ripeness": 0.0,
                "avg_fruit_count": 0.0,
                "avg_ripening_days": 0.0
            }
        }
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Maturity Stats: Average ONLY the non-zero measurements
            cursor.execute("""
                SELECT 
                    COUNT(*), 
                    AVG(NULLIF(hybrid_confidence, 0.0)), 
                    AVG(NULLIF(yolo_confidence, 0.0)), 
                    AVG(NULLIF(efficientnet_confidence, 0.0)) 
                FROM predictions 
                WHERE is_rejected = 0 AND mode != 'Yield'
            """)
            row = cursor.fetchone()
            if row and row[0]:
                stats["maturity"]["total_predictions"] = row[0]
                stats["maturity"]["hybrid_avg"] = round(row[1] or 0.0, 4)
                stats["maturity"]["yolo_avg"] = round(row[2] or 0.0, 4)
                stats["maturity"]["efficientnet_avg"] = round(row[3] or 0.0, 4)
                
            # Yield Stats
            cursor.execute("""
                SELECT 
                    COUNT(*), 
                    AVG(NULLIF(predicted_yield, 0.0)), 
                    AVG(NULLIF(ripeness_ratio, 0.0)),
                    AVG(NULLIF(fruit_count, 0)),
                    AVG(NULLIF(ripening_days, 0)),
                    AVG(NULLIF(soil_score, 0.0))
                FROM predictions 
                WHERE is_rejected = 0 AND mode = 'Yield'
            """)
            row = cursor.fetchone()
            if row and row[0]:
                stats["yield"]["total_yield_predictions"] = row[0]
                stats["yield"]["avg_yield"] = round(row[1] or 0.0, 2)
                stats["yield"]["avg_ripeness"] = round(row[2] or 0.0, 4)
                stats["yield"]["avg_fruit_count"] = round(row[3] or 0.0, 1)
                stats["yield"]["avg_ripening_days"] = round(row[4] or 0.0, 1)
                stats["yield"]["avg_soil_score"] = round(row[5] or 0.7, 4)
                
        return stats

    def get_total_count(self):
        """Get the total number of attempts stored."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM predictions")
            return cursor.fetchone()[0]

    def get_history(self, limit=1000):
        """Retrieve prediction history from the database."""
        with self._get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM predictions ORDER BY id DESC LIMIT ?", (limit,))
            return [dict(row) for row in cursor.fetchall()]

    def migrate_from_csv(self, maturity_csv, yield_csv):
        """Migrate existing CSV data to SQLite."""
        total = 0
        if os.path.exists(maturity_csv):
            try:
                df = pd.read_csv(maturity_csv)
                data = df.to_dict(orient='records')
                for d in data: d['mode'] = d.get('mode', 'Hybrid')
                self.save_predictions(data)
                total += len(data)
                print(f"[MIGRATION] Migrated {len(data)} maturity records from CSV")
            except Exception as e:
                print(f"[ERROR] Maturity CSV migration failed: {e}")

        if os.path.exists(yield_csv):
            try:
                df = pd.read_csv(yield_csv)
                data = df.to_dict(orient='records')
                for d in data: d['mode'] = 'Yield'
                self.save_predictions(data)
                total += len(data)
                print(f"[MIGRATION] Migrated {len(data)} yield records from CSV")
            except Exception as e:
                print(f"[ERROR] Yield CSV migration failed: {e}")
        
        return total

if __name__ == "__main__":
    from src.csv_storage import MATURITY_CSV, YIELD_CSV
    db = CoffeeDatabase()
    print("Database initialized.")
    migrated = db.migrate_from_csv(MATURITY_CSV, YIELD_CSV)
    print(f"Total records migrated: {migrated}")
