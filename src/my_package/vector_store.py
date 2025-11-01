import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple
import json
import sqlite3

# Optional: Try to import sentence_transformers and sklearn
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

class CliniqVectorStore:
    """Vector embeddings and similarity search for medical cases."""
    
    def __init__(self, db_path: str = "cliniq_data.db"):
        self.db_path = db_path
        # Verwende ein medizinisches Sentence Transformer Model
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.model = SentenceTransformer('all-MiniLM-L6-v2')
            except:
                print("Warning: Could not load sentence transformer model.")
                self.model = None
        else:
            print("Info: sentence-transformers not available. Using simple embeddings.")
            self.model = None
    
    def create_case_embedding(self, patient_data: Dict[str, Any], lab_result: Dict[str, Any]) -> np.ndarray:
        """Create embedding vector for a medical case."""
        if not self.model:
            # Fallback: Simple feature vector
            return self._create_simple_embedding(patient_data, lab_result)
        
        # Erstelle Textbeschreibung des Falls
        case_text = self._create_case_description(patient_data, lab_result)
        
        # Generiere Embedding
        embedding = self.model.encode([case_text])[0]
        return embedding
    
    def _create_case_description(self, patient_data: Dict[str, Any], lab_result: Dict[str, Any]) -> str:
        """Create a text description of the medical case for embedding."""
        description_parts = []
        
        # Patient info
        description_parts.append(f"Patient: {patient_data['patient_data']['age']} Jahre, {patient_data['patient_data']['gender']}")
        description_parts.append(f"MTS-Kategorie: {patient_data['patient_data']['mts_category']}")
        
        # Diagnosis
        description_parts.append(f"Verdachtsdiagnose: {patient_data['clinical_data']['suspected_diagnosis']}")
        
        # Symptoms
        description_parts.append(f"Symptomdauer: {patient_data['clinical_data']['symptom_duration']}")
        description_parts.append(f"Schmerzen: {patient_data['clinical_data']['pain_scale']}/10")
        
        # Comorbidities
        if patient_data['clinical_data']['comorbidities']:
            description_parts.append(f"Vorerkrankungen: {', '.join(patient_data['clinical_data']['comorbidities'])}")
        
        # Vital signs
        vitals = patient_data['vital_signs']
        description_parts.append(f"Blutdruck: {vitals['blood_pressure']} mmHg")
        description_parts.append(f"Puls: {vitals['heart_rate']} bpm")
        description_parts.append(f"Temperatur: {vitals['temperature']}°C")
        description_parts.append(f"SpO2: {vitals['oxygen_saturation']}%")
        
        # Lab values
        if lab_result.get('laboratory_values'):
            description_parts.append(f"Laborwerte: {', '.join(lab_result['laboratory_values'])}")
        
        # Reasoning
        if lab_result.get('reasoning'):
            description_parts.append(f"Begründung: {lab_result['reasoning']}")
        
        return " | ".join(description_parts)
    
    def _create_simple_embedding(self, patient_data: Dict[str, Any], lab_result: Dict[str, Any]) -> np.ndarray:
        """Create a simple feature vector when sentence transformers are not available."""
        features = []
        
        # Age (normalized)
        features.append(patient_data['patient_data']['age'] / 100.0)
        
        # Gender (one-hot)
        gender = patient_data['patient_data']['gender']
        features.extend([1.0 if gender == 'Männlich (m)' else 0.0])
        features.extend([1.0 if gender == 'Weiblich (w)' else 0.0])
        
        # MTS category (encoded)
        mts_map = {'red': 5, 'orange': 4, 'yellow': 3, 'green': 2, 'blue': 1}
        features.append(mts_map.get(patient_data['patient_data']['mts_category'], 0) / 5.0)
        
        # Vital signs (normalized)
        vitals = patient_data['vital_signs']
        features.extend([
            vitals['systolic_bp'] / 200.0,
            vitals['diastolic_bp'] / 120.0,
            vitals['heart_rate'] / 200.0,
            vitals['temperature'] / 42.0,
            vitals['respiratory_rate'] / 60.0,
            vitals['oxygen_saturation'] / 100.0
        ])
        
        # Pain scale
        features.append(patient_data['clinical_data']['pain_scale'] / 10.0)
        
        # Lab values count
        lab_count = len(lab_result.get('laboratory_values', []))
        features.append(lab_count / 20.0)  # Assuming max 20 lab values
        
        # Cost efficiency
        features.append(lab_result.get('cost_efficiency', 0) / 5.0)
        
        # Pad to fixed length
        while len(features) < 50:
            features.append(0.0)
        
        return np.array(features[:50])
    
    def _simple_cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Simple cosine similarity implementation."""
        dot_product = np.dot(vec1, vec2)
        norm_a = np.linalg.norm(vec1)
        norm_b = np.linalg.norm(vec2)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return dot_product / (norm_a * norm_b)
    
    def save_embedding(self, order_id: int, patient_data: Dict[str, Any], lab_result: Dict[str, Any]):
        """Save embedding to database."""
        embedding = self.create_case_embedding(patient_data, lab_result)
        
        metadata = {
            "diagnosis": patient_data['clinical_data']['suspected_diagnosis'],
            "age": patient_data['patient_data']['age'],
            "gender": patient_data['patient_data']['gender'],
            "mts_category": patient_data['patient_data']['mts_category'],
            "urgency": lab_result.get('urgency_level', ''),
            "cost_efficiency": lab_result.get('cost_efficiency', 0)
        }
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO embeddings (content_id, content_type, embedding_vector, metadata)
                VALUES (?, ?, ?, ?)
            """, (
                order_id,
                'order',
                json.dumps(embedding.tolist()),
                json.dumps(metadata)
            ))
            conn.commit()
    
    def find_similar_cases(self, patient_data: Dict[str, Any], lab_result: Dict[str, Any], limit: int = 5) -> List[Dict[str, Any]]:
        """Find similar cases using vector similarity."""
        query_embedding = self.create_case_embedding(patient_data, lab_result)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT e.content_id, e.embedding_vector, e.metadata,
                       lo.diagnosis, lo.laboratory_values, lo.reasoning, lo.cost_efficiency
                FROM embeddings e
                JOIN laboratory_orders lo ON e.content_id = lo.id
                WHERE e.content_type = 'order'
            """)
            
            results = []
            for row in cursor.fetchall():
                content_id, embedding_str, metadata_str, diagnosis, lab_values, reasoning, cost_eff = row
                
                # Lade Embedding
                stored_embedding = np.array(json.loads(embedding_str))
                metadata = json.loads(metadata_str)
                
                # Berechne Ähnlichkeit
                if SKLEARN_AVAILABLE:
                    similarity = cosine_similarity([query_embedding], [stored_embedding])[0][0]
                else:
                    # Simple cosine similarity implementation
                    similarity = self._simple_cosine_similarity(query_embedding, stored_embedding)
                
                results.append({
                    'order_id': content_id,
                    'similarity': float(similarity),
                    'metadata': metadata,
                    'diagnosis': diagnosis,
                    'laboratory_values': json.loads(lab_values) if lab_values else [],
                    'reasoning': reasoning,
                    'cost_efficiency': cost_eff
                })
            
            # Sortiere nach Ähnlichkeit
            results.sort(key=lambda x: x['similarity'], reverse=True)
            return results[:limit]
    
    def get_recommendations(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get AI recommendations based on similar cases."""
        # Erstelle temporäres lab_result für Suche
        temp_lab_result = {"laboratory_values": [], "cost_efficiency": 3}
        
        similar_cases = self.find_similar_cases(patient_data, temp_lab_result, limit=10)
        
        if not similar_cases:
            return {
                "recommended_tests": [],
                "confidence": 0.0,
                "reasoning": "Keine ähnlichen Fälle gefunden",
                "similar_cases_count": 0
            }
        
        # Analysiere ähnliche Fälle
        all_tests = []
        high_similarity_cases = [case for case in similar_cases if case['similarity'] > 0.7]
        
        for case in high_similarity_cases:
            all_tests.extend(case['laboratory_values'])
        
        if not all_tests:
            # Fallback: Nutze alle Fälle mit geringerer Schwelle
            high_similarity_cases = [case for case in similar_cases if case['similarity'] > 0.5]
            for case in high_similarity_cases:
                all_tests.extend(case['laboratory_values'])
        
        # Zähle Test-Häufigkeiten
        test_counts = {}
        for test in all_tests:
            test_counts[test] = test_counts.get(test, 0) + 1
        
        # Empfehle Tests basierend auf Häufigkeit
        total_cases = len(high_similarity_cases) if high_similarity_cases else len(similar_cases)
        recommended_tests = []
        
        for test, count in sorted(test_counts.items(), key=lambda x: x[1], reverse=True):
            confidence = count / total_cases if total_cases > 0 else 0
            if confidence >= 0.3:  # Mindestens 30% der ähnlichen Fälle
                recommended_tests.append({
                    "test": test,
                    "confidence": confidence,
                    "frequency": count,
                    "total_cases": total_cases
                })
        
        # Erstelle Begründung
        avg_similarity = sum(case['similarity'] for case in similar_cases[:5]) / min(5, len(similar_cases))
        
        reasoning = f"Basierend auf {len(similar_cases)} ähnlichen Fällen (Ø Ähnlichkeit: {avg_similarity:.2f}). "
        reasoning += f"Top-Empfehlungen aus {total_cases} hochähnlichen Fällen."
        
        return {
            "recommended_tests": recommended_tests[:10],  # Top 10
            "confidence": float(avg_similarity),
            "reasoning": reasoning,
            "similar_cases_count": len(similar_cases),
            "similar_cases": similar_cases[:3]  # Top 3 für Anzeige
        }

class CliniqAnalytics:
    """Analytics and insights for laboratory orders."""
    
    def __init__(self, db_path: str = "cliniq_data.db"):
        self.db_path = db_path
    
    def get_efficiency_trends(self) -> Dict[str, Any]:
        """Get cost efficiency trends over time."""
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query("""
                SELECT 
                    DATE(timestamp) as date,
                    AVG(cost_efficiency) as avg_efficiency,
                    COUNT(*) as order_count,
                    diagnosis,
                    urgency_level
                FROM laboratory_orders 
                WHERE timestamp >= date('now', '-30 days')
                GROUP BY DATE(timestamp), diagnosis
                ORDER BY date DESC
            """, conn)
            
            if df.empty:
                return {"trend_data": [], "summary": {}}
            
            # Gesamtstatistiken
            summary = {
                "total_orders": int(df['order_count'].sum()),
                "avg_efficiency": float(df['avg_efficiency'].mean()),
                "best_day": df.loc[df['avg_efficiency'].idxmax(), 'date'] if not df.empty else None,
                "efficiency_improvement": 0.0  # Berechne Trend
            }
            
            return {
                "trend_data": df.to_dict('records'),
                "summary": summary
            }
    
    def get_diagnosis_insights(self) -> Dict[str, Any]:
        """Get insights by diagnosis."""
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query("""
                SELECT 
                    diagnosis,
                    COUNT(*) as frequency,
                    AVG(cost_efficiency) as avg_efficiency,
                    AVG(patient_age) as avg_age,
                    urgency_level,
                    laboratory_values
                FROM laboratory_orders 
                GROUP BY diagnosis
                ORDER BY frequency DESC
            """, conn)
            
            insights = []
            for _, row in df.iterrows():
                lab_values = json.loads(row['laboratory_values']) if row['laboratory_values'] else []
                insights.append({
                    "diagnosis": row['diagnosis'],
                    "frequency": int(row['frequency']),
                    "avg_efficiency": float(row['avg_efficiency']),
                    "avg_age": float(row['avg_age']),
                    "common_tests": lab_values[:5],  # Top 5 tests
                    "urgency": row['urgency_level']
                })
            
            return {"insights": insights}
    
    def get_real_time_dashboard_data(self) -> Dict[str, Any]:
        """Get real-time dashboard data."""
        with sqlite3.connect(self.db_path) as conn:
            # Heute's Statistiken
            today_stats = pd.read_sql_query("""
                SELECT 
                    COUNT(*) as today_orders,
                    AVG(cost_efficiency) as today_efficiency,
                    SUM(CASE WHEN cost_efficiency >= 4 THEN 1 ELSE 0 END) as efficient_orders
                FROM laboratory_orders 
                WHERE DATE(timestamp) = DATE('now')
            """, conn).iloc[0]
            
            # Letzte 7 Tage Trend
            week_trend = pd.read_sql_query("""
                SELECT 
                    DATE(timestamp) as date,
                    COUNT(*) as orders,
                    AVG(cost_efficiency) as efficiency
                FROM laboratory_orders 
                WHERE timestamp >= date('now', '-7 days')
                GROUP BY DATE(timestamp)
                ORDER BY date
            """, conn)
            
            # Top Diagnosen heute
            top_diagnoses = pd.read_sql_query("""
                SELECT diagnosis, COUNT(*) as count
                FROM laboratory_orders 
                WHERE DATE(timestamp) = DATE('now')
                GROUP BY diagnosis
                ORDER BY count DESC
                LIMIT 5
            """, conn)
            
            # Kostenersparnis Schätzung
            efficiency_rate = (today_stats['efficient_orders'] / today_stats['today_orders'] * 100) if today_stats['today_orders'] > 0 else 0
            estimated_savings = today_stats['efficient_orders'] * 25 * 0.3  # 25€ pro Test, 30% Ersparnis
            
            return {
                "today_orders": int(today_stats['today_orders']),
                "today_efficiency": float(today_stats['today_efficiency']),
                "efficiency_rate": float(efficiency_rate),
                "estimated_savings": float(estimated_savings),
                "week_trend": week_trend.to_dict('records'),
                "top_diagnoses": top_diagnoses.to_dict('records')
            }