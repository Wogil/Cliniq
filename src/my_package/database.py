import sqlite3
import json
import datetime
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from pathlib import Path

class CliniqDatabase:
    """Database management for Cliniq laboratory orders and analysis."""
    
    def __init__(self, db_path: str = "cliniq_data.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database with required tables."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Laboratory Orders Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS laboratory_orders (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    patient_age INTEGER,
                    patient_gender TEXT,
                    mts_category TEXT,
                    diagnosis TEXT,
                    comorbidities TEXT,  -- JSON string
                    vital_signs TEXT,    -- JSON string
                    symptom_duration TEXT,
                    pain_scale INTEGER,
                    additional_notes TEXT,
                    laboratory_values TEXT,  -- JSON string
                    reasoning TEXT,
                    estimated_duration TEXT,
                    urgency_level TEXT,
                    cost_efficiency INTEGER,
                    quality_check TEXT,
                    session_id TEXT
                )
            """)
            
            # Laboratory Results Table (für historische Blutwerte)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS laboratory_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    order_id INTEGER,
                    parameter_name TEXT,
                    value REAL,
                    unit TEXT,
                    reference_min REAL,
                    reference_max REAL,
                    status TEXT,  -- normal, high, low, critical
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (order_id) REFERENCES laboratory_orders (id)
                )
            """)
            
            # Analysis Results Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS analysis_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    analysis_type TEXT,  -- cost_analysis, trend_analysis, etc.
                    analysis_data TEXT,  -- JSON string
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Vector Embeddings Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS embeddings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    content_id INTEGER,
                    content_type TEXT,  -- 'order', 'result', 'case_study'
                    embedding_vector TEXT,  -- JSON string of vector
                    metadata TEXT,      -- JSON string
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.commit()
    
    def save_laboratory_order(self, patient_data: Dict[str, Any], lab_result: Dict[str, Any], session_id: str) -> int:
        """Save a laboratory order to the database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO laboratory_orders (
                    patient_age, patient_gender, mts_category, diagnosis,
                    comorbidities, vital_signs, symptom_duration, pain_scale,
                    additional_notes, laboratory_values, reasoning,
                    estimated_duration, urgency_level, cost_efficiency,
                    quality_check, session_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                patient_data['patient_data']['age'],
                patient_data['patient_data']['gender'],
                patient_data['patient_data']['mts_category'],
                patient_data['clinical_data']['suspected_diagnosis'],
                json.dumps(patient_data['clinical_data']['comorbidities']),
                json.dumps(patient_data['vital_signs']),
                patient_data['clinical_data']['symptom_duration'],
                patient_data['clinical_data']['pain_scale'],
                patient_data['clinical_data']['additional_notes'],
                json.dumps(lab_result.get('laboratory_values', [])),
                lab_result.get('reasoning', ''),
                lab_result.get('estimated_duration', ''),
                lab_result.get('urgency_level', ''),
                lab_result.get('cost_efficiency', 0),
                lab_result.get('quality_check', ''),
                session_id
            ))
            
            order_id = cursor.lastrowid
            conn.commit()
            return order_id
    
    def save_laboratory_results(self, order_id: int, results: List[Dict[str, Any]]):
        """Save laboratory test results."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            for result in results:
                cursor.execute("""
                    INSERT INTO laboratory_results (
                        order_id, parameter_name, value, unit,
                        reference_min, reference_max, status
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    order_id,
                    result['parameter_name'],
                    result['value'],
                    result['unit'],
                    result.get('reference_min'),
                    result.get('reference_max'),
                    result['status']
                ))
            
            conn.commit()
    
    def get_orders_for_analysis(self, limit: int = 100) -> pd.DataFrame:
        """Get laboratory orders for analysis."""
        with sqlite3.connect(self.db_path) as conn:
            query = """
                SELECT * FROM laboratory_orders 
                ORDER BY timestamp DESC 
                LIMIT ?
            """
            return pd.read_sql_query(query, conn, params=(limit,))
    
    def get_cost_analysis(self) -> Dict[str, Any]:
        """Analyze cost efficiency trends."""
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query("""
                SELECT 
                    DATE(timestamp) as date,
                    AVG(cost_efficiency) as avg_efficiency,
                    COUNT(*) as total_orders,
                    SUM(CASE WHEN cost_efficiency >= 4 THEN 1 ELSE 0 END) as efficient_orders,
                    diagnosis,
                    urgency_level
                FROM laboratory_orders 
                WHERE timestamp >= date('now', '-30 days')
                GROUP BY DATE(timestamp)
                ORDER BY date DESC
            """, conn)
            
            if df.empty:
                return {
                    "total_orders": 0,
                    "avg_efficiency": 0,
                    "efficiency_rate": 0,
                    "cost_savings": 0,
                    "trend_data": []
                }
            
            total_orders = df['total_orders'].sum()
            avg_efficiency = df['avg_efficiency'].mean()
            efficient_orders = df['efficient_orders'].sum()
            efficiency_rate = (efficient_orders / total_orders * 100) if total_orders > 0 else 0
            
            # Geschätzte Kostenersparnis (basierend auf Effizienz)
            cost_per_test = 25  # Euro pro Test (Schätzung)
            potential_savings = efficient_orders * cost_per_test * 0.3  # 30% Ersparnis bei effizienten Tests
            
            return {
                "total_orders": int(total_orders),
                "avg_efficiency": float(avg_efficiency),
                "efficiency_rate": float(efficiency_rate),
                "cost_savings": float(potential_savings),
                "trend_data": df.to_dict('records')
            }
    
    def get_similar_cases(self, current_case: Dict[str, Any], limit: int = 5) -> List[Dict[str, Any]]:
        """Find similar cases based on diagnosis and vital signs."""
        with sqlite3.connect(self.db_path) as conn:
            # Vereinfachte Ähnlichkeitssuche basierend auf Diagnose und Alter
            query = """
                SELECT *,
                    ABS(patient_age - ?) as age_diff,
                    CASE WHEN diagnosis LIKE ? THEN 1 ELSE 0 END as diagnosis_match
                FROM laboratory_orders 
                WHERE diagnosis_match = 1 OR age_diff <= 10
                ORDER BY diagnosis_match DESC, age_diff ASC
                LIMIT ?
            """
            
            df = pd.read_sql_query(query, conn, params=(
                current_case['patient_data']['age'],
                f"%{current_case['clinical_data']['suspected_diagnosis']}%",
                limit
            ))
            
            return df.to_dict('records')
    
    def create_sample_data(self):
        """Create sample laboratory results for testing."""
        sample_results = [
            # Beispiel 1: Appendizitis-Fall
            {
                "order_data": {
                    "patient_data": {"age": 28, "gender": "Männlich (m)", "mts_category": "orange"},
                    "clinical_data": {
                        "suspected_diagnosis": "Akute Appendizitis",
                        "comorbidities": [],
                        "symptom_duration": "6-24 Stunden",
                        "pain_scale": 8,
                        "additional_notes": "Rechtsseitige Unterbauchschmerzen"
                    },
                    "vital_signs": {
                        "systolic_bp": 130, "diastolic_bp": 85, "heart_rate": 95,
                        "temperature": 38.2, "respiratory_rate": 18, "oxygen_saturation": 98
                    }
                },
                "lab_results": [
                    {"parameter_name": "CRP", "value": 45.2, "unit": "mg/L", "reference_min": 0, "reference_max": 5, "status": "high"},
                    {"parameter_name": "Leukozyten", "value": 14.5, "unit": "G/L", "reference_min": 4.0, "reference_max": 10.0, "status": "high"},
                    {"parameter_name": "Neutrophile", "value": 82, "unit": "%", "reference_min": 50, "reference_max": 70, "status": "high"},
                ]
            },
            # Beispiel 2: Pneumonie-Fall
            {
                "order_data": {
                    "patient_data": {"age": 65, "gender": "Weiblich (w)", "mts_category": "yellow"},
                    "clinical_data": {
                        "suspected_diagnosis": "Pneumonie",
                        "comorbidities": ["COPD/Asthma"],
                        "symptom_duration": "1-3 Tage",
                        "pain_scale": 4,
                        "additional_notes": "Husten mit Auswurf, Dyspnoe"
                    },
                    "vital_signs": {
                        "systolic_bp": 145, "diastolic_bp": 90, "heart_rate": 105,
                        "temperature": 39.1, "respiratory_rate": 24, "oxygen_saturation": 92
                    }
                },
                "lab_results": [
                    {"parameter_name": "CRP", "value": 89.3, "unit": "mg/L", "reference_min": 0, "reference_max": 5, "status": "high"},
                    {"parameter_name": "PCT", "value": 2.1, "unit": "ng/mL", "reference_min": 0, "reference_max": 0.25, "status": "high"},
                    {"parameter_name": "Leukozyten", "value": 16.8, "unit": "G/L", "reference_min": 4.0, "reference_max": 10.0, "status": "high"},
                ]
            }
        ]
        
        for sample in sample_results:
            # Erstelle Lab Order Response
            lab_response = {
                "laboratory_values": [result["parameter_name"] for result in sample["lab_results"]],
                "reasoning": "Basierend auf klinischen Symptomen und Vitalparametern",
                "estimated_duration": "2-4",
                "urgency_level": "hoch",
                "cost_efficiency": 5,
                "quality_check": "Empfehlung entspricht Leitlinien"
            }
            
            # Speichere Order
            order_id = self.save_laboratory_order(
                sample["order_data"],
                lab_response,
                f"sample_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            
            # Speichere Results
            self.save_laboratory_results(order_id, sample["lab_results"])
        
        print("Sample data created successfully!")