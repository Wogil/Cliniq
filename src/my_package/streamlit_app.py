import os
import time
import json
import uuid
from typing import Optional, Dict, Any

import streamlit as st
import openai
from urllib.parse import urlparse, parse_qs
from dotenv import load_dotenv
import pandas as pd

# Import our new modules
from database import CliniqDatabase
from vector_store import CliniqVectorStore, CliniqAnalytics

load_dotenv()

# Initialize database and vector store
@st.cache_resource
def init_cliniq_system():
    """Initialize the Cliniq system with database and vector store."""
    db = CliniqDatabase()
    vector_store = CliniqVectorStore()
    analytics = CliniqAnalytics()
    return db, vector_store, analytics

db, vector_store, analytics = init_cliniq_system()

# Konfigurierbare Parameter - hier können Sie die Einstellungen einfach anpassen
TEMPERATURE = 0.3  # Niedrigere Temperatur für präzisere medizinische Empfehlungen
ASSISTANT_INSTRUCTION = """Du bist ein spezialisierter Labormedizin-Assistent für Krankenhäuser. Deine Aufgabe ist es, basierend auf den gegebenen Patientendaten, Vitalparametern und Verdachtsdiagnosen, eine präzise und kosteneffiziente Laborbeauftragung zu erstellen.

WICHTIG: Die beauftragten Laborwerte müssen:
1. DIREKT zur Verdachtsdiagnose passen
2. Die aktuellen VITALPARAMETER berücksichtigen (Blutdruck, Puls, Temperatur, Atmung, SpO2)
3. Kosteneffizient sein (Krankenkassen-konform)
4. Medizinisch begründet und notwendig sein
5. Keine überflüssigen oder "Nice-to-have" Parameter enthalten

BESONDERE BEACHTUNG der Vitalparameter:
- Fieber → Entzündungsparameter, Blutkulturen
- Hypotonie → Elektrolyte, Nierenwerte, Herzenzyme
- Tachykardie → Herzenzyme, Elektrolyte, Schilddrüse
- Hypoxämie → Blutgasanalyse, D-Dimer, Herzenzyme
- Hypertonie → Nierenwerte, Elektrolyte

Gib eine strukturierte JSON-Antwort zurück mit folgenden Feldern:
- diagnosis_confirmation: Bestätigung/Verfeinerung der Verdachtsdiagnose
- laboratory_values: Liste der empfohlenen Laborwerte (nur die WIRKLICH notwendigen!)
- reasoning: Medizinische Begründung für jeden Laborwert
- estimated_duration: Geschätzte maximale Labordauer in Stunden
- urgency_level: Dringlichkeitsstufe basierend auf MTS-Kategorie
- cost_efficiency: Bewertung der Kosteneffizienz (1-5, 5=sehr effizient)
- quality_check: Qualitätsprüfung der Empfehlung

Beispiel:
{
  "diagnosis_confirmation": "Akute Appendizitis (Verdacht bestätigt)",
  "laboratory_values": ["CRP", "Leukozyten", "Neutrophile", "BSG"],
  "reasoning": "CRP und Leukozyten für Entzündungsnachweis, Neutrophile für bakterielle Infektion, BSG als Verlaufskontrolle",
  "estimated_duration": "2-4",
  "urgency_level": "hoch",
  "cost_efficiency": 5,
  "quality_check": "Empfehlung entspricht Leitlinien, kosteneffizient, diagnostisch aussagekräftig"
}"""

# MTS-Kategorien (Manchester Triage System)
MTS_CATEGORIES = {
    "Rot (Sofort)": "red",
    "Orange (Sehr dringend)": "orange", 
    "Gelb (Dringend)": "yellow",
    "Grün (Weniger dringend)": "green",
    "Blau (Nicht dringend)": "blue"
}

# Geschlechtsoptionen
GENDER_OPTIONS = ["Männlich (m)", "Weiblich (w)", "Divers (d)"]

# Häufige Verdachtsdiagnosen
COMMON_DIAGNOSES = [
    "Manuelle Eingabe (eigene Diagnose)",
    "Akute Appendizitis",
    "Pneumonie",
    "Myokardinfarkt",
    "Diabetes mellitus",
    "Niereninsuffizienz",
    "Leberfunktionsstörung",
    "Anämie",
    "Hyperthyreose",
    "Hypothyreose",
    "Herzinsuffizienz",
    "COPD-Exazerbation",
    "Gastroenteritis",
    "Harnwegsinfekt",
    "Sepsis",
    "Thrombose",
    "Lungenembolie",
    "Schlaganfall"
]

# Wichtige Vorerkrankungen/Zusatzinformationen
COMORBIDITIES = [
    "Diabetes mellitus",
    "Arterielle Hypertonie",
    "Koronare Herzkrankheit",
    "Herzinsuffizienz",
    "Niereninsuffizienz",
    "Leberzirrhose",
    "COPD/Asthma",
    "Maligne Erkrankung",
    "Antikoagulation",
    "Immunsuppression",
    "Schwangerschaft",
    "Allergien/Unverträglichkeiten"
]

# Symptomdauer-Optionen
SYMPTOM_DURATION = [
    "< 1 Stunde",
    "1-6 Stunden", 
    "6-24 Stunden",
    "1-3 Tage",
    "4-7 Tage",
    "1-4 Wochen",
    "> 1 Monat"
]


def configure_openai():
    """Configure the openai client for Azure OpenAI using environment variables.
    """
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    api_version = os.getenv("OPENAI_API_VERSION")

    if not endpoint or not api_key:
        raise ValueError("AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT environment variables are required")

    # Parse the full endpoint URL to extract deployment and api version
    parsed = urlparse(endpoint)
    path = parsed.path or ""
    
    # Extract deployment name from path
    if "/deployments/" not in path:
        raise ValueError("AZURE_OPENAI_ENDPOINT must contain '/deployments/<name>/'")
    
    parts = [p for p in path.split("/") if p]
    try:
        dep_idx = parts.index("deployments")
        deployment_name = parts[dep_idx + 1]
    except (ValueError, IndexError):
        raise ValueError("Could not parse deployment name from AZURE_OPENAI_ENDPOINT")

    # Extract API version from query params if present
    if parsed.query:
        params = parse_qs(parsed.query)
        if "api-version" in params:
            api_version = params["api-version"][0]

    # Build api_base without the deployment path and query
    api_base = f"{parsed.scheme}://{parsed.netloc}"
    
    # Configure OpenAI client
    openai.api_type = "azure"
    openai.api_base = api_base
    openai.api_version = api_version
    openai.api_key = api_key
    
    return {"engine": deployment_name}  # Use engine instead of deployment for Azure


def stream_completion(call_args):
    """Generator wrapper for streaming chat completions from openai.

    Yields text chunks as they arrive.
    """
    # Engine is already properly set in call_args from configure_openai()
    for chunk in openai.ChatCompletion.create(
        stream=True,
        **call_args
    ):
        # payload parsing depends on the library; each chunk may contain choices
        if "choices" in chunk:
            for choice in chunk["choices"]:
                delta = choice.get("delta", {})
                text = delta.get("content")
                if text:
                    yield text


def get_completion(call_args):
    """Get a complete response from OpenAI API (non-streaming).
    
    Returns the full response text.
    """
    response = openai.ChatCompletion.create(**call_args)
    return response.choices[0].message.content


def create_laboratory_json(mts_category: str, gender: str, age: int, diagnosis: str, 
                          comorbidities: list, symptom_duration: str, pain_scale: int, 
                          additional_notes: str, systolic_bp: int, diastolic_bp: int,
                          heart_rate: int, temperature: float, respiratory_rate: int, 
                          oxygen_saturation: int) -> Dict[str, Any]:
    """Erstellt ein JSON-Objekt aus den Laborbeauftragungsdaten.
    
    Args:
        mts_category: MTS-Kategorie (Manchester Triage)
        gender: Geschlecht des Patienten
        age: Alter in Jahren
        diagnosis: Verdachtsdiagnose
        comorbidities: Liste der Vorerkrankungen
        symptom_duration: Dauer der Symptome
        pain_scale: Schmerzskala (0-10)
        additional_notes: Zusätzliche Anmerkungen
        systolic_bp: Systolischer Blutdruck
        diastolic_bp: Diastolischer Blutdruck
        heart_rate: Herzfrequenz
        temperature: Körpertemperatur
        respiratory_rate: Atemfrequenz
        oxygen_saturation: Sauerstoffsättigung
    
    Returns:
        Dictionary mit den strukturierten Patientendaten
    """
    return {
        "patient_data": {
            "mts_category": mts_category,
            "gender": gender,
            "age": age
        },
        "vital_signs": {
            "blood_pressure": f"{systolic_bp}/{diastolic_bp}",
            "systolic_bp": systolic_bp,
            "diastolic_bp": diastolic_bp,
            "heart_rate": heart_rate,
            "temperature": temperature,
            "respiratory_rate": respiratory_rate,
            "oxygen_saturation": oxygen_saturation
        },
        "clinical_data": {
            "suspected_diagnosis": diagnosis,
            "comorbidities": comorbidities,
            "symptom_duration": symptom_duration,
            "pain_scale": pain_scale,
            "additional_notes": additional_notes
        },
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }


def create_laboratory_prompt(patient_data: Dict[str, Any]) -> str:
    """Erstellt einen strukturierten Prompt für die Laborbeauftragung.
    
    Args:
        patient_data: Dictionary mit den Patientendaten
    
    Returns:
        Formatierter Prompt für die OpenAI API
    """
    prompt = f"""
LABORBEAUFTRAGUNG - PATIENTENFALL

BASISDATEN:
- MTS-Kategorie: {patient_data['patient_data']['mts_category']}
- Geschlecht: {patient_data['patient_data']['gender']}
- Alter: {patient_data['patient_data']['age']} Jahre

VITALPARAMETER (WICHTIG FÜR LABORWERTE):
- Blutdruck: {patient_data['vital_signs']['blood_pressure']} mmHg
- Herzfrequenz: {patient_data['vital_signs']['heart_rate']} bpm
- Körpertemperatur: {patient_data['vital_signs']['temperature']}°C
- Atemfrequenz: {patient_data['vital_signs']['respiratory_rate']}/min
- Sauerstoffsättigung: {patient_data['vital_signs']['oxygen_saturation']}%

KLINISCHE DATEN:
- Verdachtsdiagnose: {patient_data['clinical_data']['suspected_diagnosis']}
- Symptomdauer: {patient_data['clinical_data']['symptom_duration']}
- Schmerzskala: {patient_data['clinical_data']['pain_scale']}/10
- Vorerkrankungen: {', '.join(patient_data['clinical_data']['comorbidities']) if patient_data['clinical_data']['comorbidities'] else 'Keine angegeben'}
- Zusätzliche Informationen: {patient_data['clinical_data']['additional_notes'] or 'Keine'}

AUFGABE: Erstelle eine präzise, kosteneffiziente Laborbeauftragung, die:
1. Nur diagnostisch relevante Parameter enthält
2. Zur Verdachtsdiagnose passt
3. Krankenkassen-konform ist
4. Keine überflüssigen Tests beinhaltet

Zeitpunkt: {patient_data['timestamp']}
"""
    return prompt


def parse_openai_response(response_text: str) -> Dict[str, Any]:
    """Parst die OpenAI-Antwort und extrahiert JSON-Felder.
    
    Args:
        response_text: Rohe Antwort von der OpenAI API
    
    Returns:
        Dictionary mit den extrahierten Feldern oder Fehlermeldung
    """
    try:
        # Versuche, JSON aus der Antwort zu extrahieren
        # Manchmal ist die JSON-Antwort in Markdown-Codeblöcken eingebettet
        if "```json" in response_text:
            start = response_text.find("```json") + 7
            end = response_text.find("```", start)
            json_str = response_text[start:end].strip()
        elif response_text.strip().startswith("{"):
            json_str = response_text.strip()
        else:
            # Falls kein JSON gefunden wird, erstelle eine strukturierte Antwort
            return {
                "name": "Unstrukturierte Antwort",
                "score": 0.5,
                "recommendation": response_text,
                "details": "Die API hat keine strukturierte JSON-Antwort geliefert",
                "urgency": "medium"
            }
        
        parsed_data = json.loads(json_str)
        
        # Validiere und ergänze fehlende Felder
        required_fields = ["name", "score", "recommendation", "details", "urgency"]
        for field in required_fields:
            if field not in parsed_data:
                parsed_data[field] = "Nicht verfügbar"
        
        return parsed_data
        
    except json.JSONDecodeError as e:
        return {
            "name": "JSON-Parsing-Fehler",
            "score": 0.0,
            "recommendation": "Fehler beim Parsen der API-Antwort",
            "details": f"JSON-Fehler: {str(e)}",
            "urgency": "low"
        }
    except Exception as e:
        return {
            "name": "Unbekannter Fehler",
            "score": 0.0,
            "recommendation": "Ein unerwarteter Fehler ist aufgetreten",
            "details": f"Fehler: {str(e)}",
            "urgency": "low"
        }


def render_laboratory_form():
    """Rendert das Laborbeauftragungsformular."""
    st.markdown("""
    <div style="background: white; padding: 1.5rem; border-radius: 12px; box-shadow: 0 4px 16px rgba(0,0,0,0.1); margin: 1rem 0;">
        <h3 style="color: #1e40af; margin: 0 0 0.5rem 0; font-weight: 600;">📋 Patientendaten erfassen</h3>
        <p style="color: #6b7280; margin: 0; font-style: italic;">Einfache und sichere Eingabe für medizinisches Fachpersonal</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Custom Vorerkrankungen Management AUSSERHALB des Forms
    st.markdown("""
    <div style="background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); padding: 1rem; border-radius: 8px; border-left: 4px solid #0ea5e9; margin: 1rem 0;">
        <strong style="color: #0c4a6e;">➕ Zusätzliche Vorerkrankungen verwalten</strong>
    </div>
    """, unsafe_allow_html=True)
    
    # Container für custom Vorerkrankungen
    if 'custom_comorbidities' not in st.session_state:
        st.session_state.custom_comorbidities = []
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        new_comorbidity = st.text_input(
            "Neue Vorerkrankung hinzufügen:",
            placeholder="z.B. Spezielle Allergie, seltene Erkrankung...",
            key="new_comorbidity_input"
        )
    
    with col2:
        st.write("") # Spacing
        if st.button("Hinzufügen", key="add_comorbidity"):
            if new_comorbidity.strip():
                if new_comorbidity.strip() not in st.session_state.custom_comorbidities:
                    st.session_state.custom_comorbidities.append(new_comorbidity.strip())
                    st.rerun()
                else:
                    st.warning("Diese Vorerkrankung wurde bereits hinzugefügt!")
    
    # Anzeige der custom Vorerkrankungen
    if st.session_state.custom_comorbidities:
        st.markdown("**Verfügbare custom Vorerkrankungen:**")
        cols = st.columns(3)
        for idx, custom_condition in enumerate(st.session_state.custom_comorbidities):
            with cols[idx % 3]:
                col_inner1, col_inner2 = st.columns([3, 1])
                with col_inner1:
                    st.write(f"[Custom] {custom_condition}")
                with col_inner2:
                    if st.button("Löschen", key=f"delete_custom_{idx}", help="Löschen"):
                        st.session_state.custom_comorbidities.pop(idx)
                        st.rerun()
        
        # Reset-Button für alle custom Vorerkrankungen
        if st.button("Alle custom Vorerkrankungen zurücksetzen", key="reset_custom"):
            st.session_state.custom_comorbidities = []
            st.rerun()
    
    st.markdown("---")
    
    # Vitalparameter AUSSERHALB des Forms für Live-Updates
    st.markdown("""
    <div style="background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%); padding: 1.5rem; border-radius: 12px; border-left: 4px solid #22c55e; margin: 1.5rem 0;">
        <h4 style="color: #15803d; margin: 0 0 0.5rem 0; font-weight: 600;">🩺 Live-Vitalparameter</h4>
        <p style="color: #16a34a; margin: 0; font-size: 0.9rem;">Automatische medizinische Bewertung in Echtzeit</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Blutdruck
        systolic_bp = st.number_input(
            "Systolischer Blutdruck (mmHg):",
            min_value=60,
            max_value=250,
            value=120,
            help="Oberer Blutdruckwert",
            key="live_systolic_bp"
        )
        diastolic_bp = st.number_input(
            "Diastolischer Blutdruck (mmHg):",
            min_value=30,
            max_value=150,
            value=80,
            help="Unterer Blutdruckwert",
            key="live_diastolic_bp"
        )
        
        # Blutdruck-Bewertung - Live Update
        if systolic_bp >= 180 or diastolic_bp >= 110:
            st.error("Hypertensive Krise!")
        elif systolic_bp >= 140 or diastolic_bp >= 90:
            st.warning("Hypertonie")
        elif systolic_bp < 90 or diastolic_bp < 60:
            st.warning("Hypotonie")
        else:
            st.success("Normaler Blutdruck")
    
    with col2:
        # Puls
        heart_rate = st.number_input(
            "Puls (bpm):",
            min_value=30,
            max_value=200,
            value=70,
            help="Herzfrequenz pro Minute",
            key="live_heart_rate"
        )
        
        # Temperatur
        temperature = st.number_input(
            "Körpertemperatur (°C):",
            min_value=35.0,
            max_value=42.0,
            value=36.5,
            step=0.1,
            help="Körpertemperatur in Celsius",
            key="live_temperature"
        )
        
        # Temperatur-Bewertung - Live Update
        if temperature >= 38.5:
            st.error("Hohes Fieber")
        elif temperature >= 38.0:
            st.warning("Fieber")
        elif temperature >= 37.5:
            st.info("Subfebrile Temperatur")
        elif temperature < 36.0:
            st.warning("Hypothermie")
        else:
            st.success("Normale Temperatur")
    
    with col3:
        # Atemfrequenz
        respiratory_rate = st.number_input(
            "Atemfrequenz (/min):",
            min_value=5,
            max_value=60,
            value=16,
            help="Atemzüge pro Minute",
            key="live_respiratory_rate"
        )
        
        # Sauerstoffsättigung
        oxygen_saturation = st.number_input(
            "Sauerstoffsättigung (%):",
            min_value=70,
            max_value=100,
            value=98,
            help="SpO2 in Prozent",
            key="live_oxygen_saturation"
        )
        
        # Sauerstoffsättigung-Bewertung - Live Update
        if oxygen_saturation < 90:
            st.error("Schwere Hypoxämie")
        elif oxygen_saturation < 94:
            st.warning("Hypoxämie")
        elif oxygen_saturation < 96:
            st.info("Leichte Hypoxämie")
        else:
            st.success("Normale Sättigung")
    
    # Puls- und Atemfrequenz-Bewertung (nach den Spalten) - Live Update
    col_pulse = st.columns([1, 2, 1])
    with col_pulse[1]:
        # Puls-Status
        if heart_rate > 100:
            st.warning(f"Tachykardie ({heart_rate} bpm)")
        elif heart_rate < 60:
            st.info(f"Bradykardie ({heart_rate} bpm)")
        else:
            st.success(f"Normaler Puls ({heart_rate} bpm)")
        
        # Atemfrequenz-Status
        if respiratory_rate > 20:
            st.warning(f"Tachypnoe ({respiratory_rate}/min)")
        elif respiratory_rate < 12:
            st.info(f"Bradypnoe ({respiratory_rate}/min)")
        else:
            st.success(f"Normale Atmung ({respiratory_rate}/min)")
    
    # Schmerzskala AUSSERHALB des Forms
    st.markdown("""
    <div style="background: linear-gradient(135deg, #fefce8 0%, #fef3c7 100%); padding: 1rem; border-radius: 8px; border-left: 4px solid #f59e0b; margin: 1rem 0;">
        <strong style="color: #92400e;">🔥 Schmerzskala (0-10)</strong>
    </div>
    """, unsafe_allow_html=True)
    pain_scale = st.slider(
        "Aktueller Schmerz:",
        min_value=0,
        max_value=10,
        value=0,
        help="0 = Kein Schmerz, 10 = Unerträglicher Schmerz",
        key="live_pain_scale"
    )
    
    # Anzeige der Schmerzstufe - Live Update
    pain_labels = {
        (0, 1): "Kein Schmerz",
        (1, 4): "Leichter Schmerz",
        (4, 7): "Mäßiger Schmerz", 
        (7, 9): "Starker Schmerz",
        (9, 11): "Unerträglicher Schmerz"
    }
    
    for (min_val, max_val), label in pain_labels.items():
        if min_val <= pain_scale < max_val:
            if pain_scale >= 7:
                st.error(f"Bewertung: {label}")
            elif pain_scale >= 4:
                st.warning(f"Bewertung: {label}")
            elif pain_scale >= 1:
                st.info(f"Bewertung: {label}")
            else:
                st.success(f"Bewertung: {label}")
            break
    
    st.markdown("---")
    
    with st.form("laboratory_form"):
        # Kategorie 1: Patienten-Basisdaten
        st.markdown("**Basisdaten**")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            mts_category_display = st.selectbox(
                "MTS-Kategorie (Manchester Triage):",
                options=list(MTS_CATEGORIES.keys()),
                help="Triage-Kategorie bestimmt die Dringlichkeit"
            )
            mts_category = MTS_CATEGORIES[mts_category_display]
        
        with col2:
            gender = st.selectbox(
                "Geschlecht:",
                options=GENDER_OPTIONS,
                help="Biologisches Geschlecht des Patienten"
            )
        
        with col3:
            age = st.number_input(
                "Alter (Jahre):",
                min_value=0,
                max_value=120,
                value=50,
                help="Alter in vollendeten Lebensjahren"
            )
        
        st.markdown("---")
        
        # Kategorie 2: Diagnose und Anamnese
        st.markdown("**Diagnose & Anamnese**")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            diagnosis_selection = st.selectbox(
                "Verdachtsdiagnose:",
                options=COMMON_DIAGNOSES,
                help="Wählen Sie eine bekannte Diagnose oder geben Sie manuell ein"
            )
            
            if diagnosis_selection == "Manuelle Eingabe (eigene Diagnose)":
                manual_diagnosis = st.text_input(
                    "Eigene Verdachtsdiagnose:",
                    placeholder="Geben Sie die Verdachtsdiagnose ein..."
                )
                final_diagnosis = manual_diagnosis
            else:
                final_diagnosis = diagnosis_selection
        
        with col2:
            symptom_duration = st.selectbox(
                "Dauer der Symptome:",
                options=SYMPTOM_DURATION,
                help="Wie lange bestehen die Symptome bereits?"
            )
        
        # Wichtige Vorerkrankungen
        st.markdown("**Wichtige Vorerkrankungen/Zusatzinformationen:**")
        selected_comorbidities = []
        
        # Kombinierte Liste: Standard + Custom Vorerkrankungen
        all_comorbidities = COMORBIDITIES.copy()
        if 'custom_comorbidities' in st.session_state:
            all_comorbidities.extend(st.session_state.custom_comorbidities)
        
        # Aufteilen in 3 Spalten für bessere Darstellung
        cols = st.columns(3)
        for i, condition in enumerate(all_comorbidities):
            with cols[i % 3]:
                # Kennzeichnung für custom Vorerkrankungen
                display_name = f"[Custom] {condition}" if condition in st.session_state.get('custom_comorbidities', []) else condition
                if st.checkbox(display_name, key=f"comorbidity_{i}"):
                    selected_comorbidities.append(condition)
        
        # Zusätzliche Anmerkungen
        additional_notes = st.text_area(
            "Zusätzliche Anmerkungen:",
            height=100,
            placeholder="Weitere relevante Informationen, Allergien, aktuelle Medikation, etc."
        )
        
        # Submit-Button
        submitted = st.form_submit_button(
            "Laborbeauftragung erstellen",
            use_container_width=True,
            type="primary"
        )
    
    return (submitted, mts_category, gender, age, final_diagnosis, 
            selected_comorbidities, symptom_duration, pain_scale, additional_notes,
            systolic_bp, diastolic_bp, heart_rate, temperature, respiratory_rate, oxygen_saturation)


def render_laboratory_results(lab_result: Dict[str, Any]):
    """Rendert die Ergebnisse der Laborbeauftragung."""
    st.markdown("""
    <div style="background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%); padding: 2rem; border-radius: 16px; margin: 2rem 0; border: 1px solid #a7f3d0; box-shadow: 0 8px 32px rgba(16, 185, 129, 0.1);">
        <h2 style="color: #065f46; margin: 0; font-weight: 700; display: flex; align-items: center;">
            🧬 Laborbeauftragung - Analyseergebnisse
        </h2>
        <p style="color: #047857; margin: 0.5rem 0 0 0; font-size: 1.1rem;">KI-optimierte, kosteneffiziente Labordiagnostik</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Hauptergebnisse in Spalten
    col1, col2 = st.columns(2)
    
    with col1:
        # Realistische Labordauer basierend auf Anzahl der Tests
        lab_values = lab_result.get("laboratory_values", [])
        num_tests = len(lab_values)
        
        # Realistische Zeitschätzung basierend auf Testarten
        base_time = 0.5  # 30 Minuten Grundzeit
        per_test_time = 0.25  # 15 Minuten pro Test
        estimated_hours = base_time + (num_tests * per_test_time)
        
        # Spezielle Tests benötigen mehr Zeit
        complex_tests = ["Blutkultur", "Mikrobiologie", "Genetik", "Histologie", "Zytologie", "PCR", "Kultur"]
        for test in lab_values:
            if any(complex in test for complex in complex_tests):
                estimated_hours += 2  # 2 Stunden extra für komplexe Tests
        
        # Aufrunden auf realistische Werte
        if estimated_hours <= 2:
            duration_display = "1-2 Stunden"
        elif estimated_hours <= 4:
            duration_display = "2-4 Stunden" 
        elif estimated_hours <= 8:
            duration_display = "4-8 Stunden"
        elif estimated_hours <= 24:
            duration_display = "8-24 Stunden"
        else:
            duration_display = "24-48 Stunden"
        # Custom Metric Card for Duration
        st.markdown(f"""
        <div class="metric-card">
            <h4 style="color: #1e40af; margin: 0 0 0.5rem 0;">⏱️ Realistische Labordauer</h4>
            <p style="font-size: 1.5rem; font-weight: 700; color: #1e3a8a; margin: 0;">{duration_display}</p>
            <p style="color: #6b7280; font-size: 0.9rem; margin: 0.25rem 0 0 0;">Basierend auf Testanzahl und Komplexität</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        cost_efficiency = lab_result.get("cost_efficiency", 0)
        efficiency_colors = {
            1: "Sehr ineffizient",
            2: "Ineffizient", 
            3: "Akzeptabel",
            4: "Effizient",
            5: "Sehr effizient"
        }
        # Custom Metric Card for Cost Efficiency
        efficiency_value = efficiency_colors.get(cost_efficiency, "Unbekannt")
        efficiency_color = "#10b981" if cost_efficiency >= 4 else "#f59e0b" if cost_efficiency >= 3 else "#ef4444"
        
        st.markdown(f"""
        <div class="metric-card">
            <h4 style="color: #1e40af; margin: 0 0 0.5rem 0;">💰 Kosteneffizienz</h4>
            <p style="font-size: 1.5rem; font-weight: 700; color: {efficiency_color}; margin: 0;">{efficiency_value}</p>
            <p style="color: #6b7280; font-size: 0.9rem; margin: 0.25rem 0 0 0;">Krankenkassen-Konformität: {cost_efficiency}/5</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Empfohlene Laborwerte
    st.markdown("""
    <div style="background: white; padding: 1.5rem; border-radius: 12px; box-shadow: 0 4px 16px rgba(0,0,0,0.1); margin: 1.5rem 0; border-left: 4px solid #3b82f6;">
        <h3 style="color: #1e40af; margin: 0 0 1rem 0;">🔬 Empfohlene Laborparameter</h3>
    </div>
    """, unsafe_allow_html=True)
    
    lab_values = lab_result.get("laboratory_values", [])
    
    if lab_values:
        # Erstelle eine schöne Darstellung der Laborwerte mit Medical Badges
        cols = st.columns(min(3, len(lab_values)))
        for i, value in enumerate(lab_values):
            with cols[i % 3]:
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%); padding: 1rem; border-radius: 8px; margin: 0.25rem 0; border: 1px solid #a7f3d0; text-align: center;">
                    <span style="color: #065f46; font-weight: 600; font-size: 1rem;">✓ {value}</span>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.warning("Keine spezifischen Laborwerte empfohlen")
    
    # Medizinische Begründung
    st.subheader("Medizinische Begründung")
    reasoning = lab_result.get("reasoning", "Keine Begründung verfügbar")
    st.info(reasoning)
    
    # Dringlichkeitsstufe
    urgency = lab_result.get("urgency_level", "medium")
    urgency_colors = {
        "niedrig": "Niedrige Dringlichkeit",
        "mittel": "Mittlere Dringlichkeit", 
        "hoch": "Hohe Dringlichkeit",
        "sehr hoch": "Sehr hohe Dringlichkeit"
    }
    
    urgency_display = urgency_colors.get(urgency.lower(), f"{urgency}")
    
    if "hoch" in urgency.lower():
        st.error(f"**Dringlichkeit**: {urgency_display}")
        st.error("**Sofortige Maßnahmen erforderlich!** Labor umgehend beauftragen.")
    else:
        st.info(f"**Dringlichkeit**: {urgency_display}")
    
    # Qualitätsprüfung
    st.subheader("Qualitätsprüfung")
    quality_check = lab_result.get("quality_check", "Keine Qualitätsprüfung verfügbar")
    
    if "leitlinien" in quality_check.lower() and "kosteneffizient" in quality_check.lower():
        st.success(f"**Qualität bestätigt**: {quality_check}")
    else:
        st.warning(f"**Qualitätsprüfung**: {quality_check}")
    
    # Kostenwarnung
    if cost_efficiency < 3:
        st.error("**Kostenwarnung**: Diese Laborbeauftragung könnte zu hohe Kosten verursachen!")
        st.error("**Risiko**: Mögliche Ablehnung durch Krankenkasse - Krankenhaus-Budget gefährdet!")
    
    # Visualisierung der Labordauer
    if duration_display != "Unbekannt":
        st.subheader("Labordauer-Details")
        try:
            # Parse die Dauer (z.B. "2-4 Stunden" -> 2, 4)
            if "-" in duration_display:
                hours_part = duration_display.replace(" Stunden", "")
                min_hours, max_hours = map(int, hours_part.split("-"))
                avg_hours = (min_hours + max_hours) / 2
            else:
                avg_hours = 2.0
                min_hours = max_hours = avg_hours
            
            # Einfache Visualisierung
            progress_bar = st.progress(0)
            for i in range(int(avg_hours) + 1):
                progress_bar.progress(min(i / max_hours, 1.0))
                time.sleep(0.05)  # Schnellere Animation
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Minimum", f"{min_hours}h")
            with col2:
                st.metric("Durchschnitt", f"{avg_hours:.1f}h")
            with col3:
                st.metric("Maximum", f"{max_hours}h")
                
        except:
            st.info(f"Erwartete Dauer: {duration_display}")
    
    # JSON-Daten anzeigen (erweitert)
    with st.expander("Technische Details (Vollständige API-Antwort)"):
        st.json(lab_result)


def render_ai_recommendations(recommendations: Dict[str, Any]):
    """Rendert KI-Empfehlungen basierend auf ähnlichen Fällen."""
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%); padding: 1.5rem; border-radius: 12px; border-left: 4px solid #3b82f6; margin: 1rem 0;">
        <h4 style="color: #1e40af; margin: 0 0 0.5rem 0;">🤖 KI-Empfehlungen</h4>
        <p style="color: #1e40af; margin: 0; font-size: 0.9rem;">Basierend auf {recommendations['similar_cases_count']} ähnlichen Fällen (Konfidenz: {recommendations['confidence']:.0%})</p>
    </div>
    """, unsafe_allow_html=True)
    
    if recommendations['recommended_tests']:
        st.markdown("**🔬 Empfohlene Laborwerte:**")
        cols = st.columns(min(3, len(recommendations['recommended_tests'])))
        
        for i, test_rec in enumerate(recommendations['recommended_tests'][:6]):
            with cols[i % 3]:
                confidence_color = "#10b981" if test_rec['confidence'] >= 0.7 else "#f59e0b" if test_rec['confidence'] >= 0.5 else "#6b7280"
                st.markdown(f"""
                <div style="background: white; padding: 1rem; border-radius: 8px; margin: 0.25rem 0; border: 1px solid #e5e7eb; text-align: center;">
                    <div style="color: {confidence_color}; font-weight: 600; font-size: 0.9rem;">{test_rec['test']}</div>
                    <div style="color: #6b7280; font-size: 0.8rem;">Konfidenz: {test_rec['confidence']:.0%}</div>
                    <div style="color: #9ca3af; font-size: 0.7rem;">{test_rec['frequency']}/{test_rec['total_cases']} Fälle</div>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown(f"**💡 Begründung:** {recommendations['reasoning']}")
        
        if recommendations.get('similar_cases'):
            with st.expander("📚 Ähnliche Fälle anzeigen"):
                for i, case in enumerate(recommendations['similar_cases'][:3]):
                    st.markdown(f"""
                    **Fall {i+1}** (Ähnlichkeit: {case['similarity']:.1%})
                    - **Diagnose:** {case['diagnosis']}
                    - **Laborwerte:** {', '.join(case['laboratory_values'])}
                    - **Kosteneffizienz:** {case['cost_efficiency']}/5
                    - **Begründung:** {case['reasoning'][:100]}...
                    """)


def render_analytics_dashboard():
    """Rendert das Analytics Dashboard."""
    st.markdown("""
    <div class="pro-header">
        <h1>📊 Analytics Dashboard</h1>
        <p>Einblicke und Trends für optimierte Laborbeauftragung</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Real-time Dashboard Data
    try:
        dashboard_data = analytics.get_real_time_dashboard_data()
        
        # Top Metriken
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h4 style="color: #1e40af; margin: 0 0 0.5rem 0;">📋 Heute beauftragt</h4>
                <p style="font-size: 1.8rem; font-weight: 700; color: #1e3a8a; margin: 0;">{dashboard_data['today_orders']}</p>
                <p style="color: #6b7280; font-size: 0.9rem; margin: 0.25rem 0 0 0;">Laborbeauftragungen</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            efficiency_color = "#10b981" if dashboard_data['today_efficiency'] >= 4 else "#f59e0b"
            st.markdown(f"""
            <div class="metric-card">
                <h4 style="color: #1e40af; margin: 0 0 0.5rem 0;">⚡ Effizienz</h4>
                <p style="font-size: 1.8rem; font-weight: 700; color: {efficiency_color}; margin: 0;">{dashboard_data['today_efficiency']:.1f}/5</p>
                <p style="color: #6b7280; font-size: 0.9rem; margin: 0.25rem 0 0 0;">Durchschnitt heute</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h4 style="color: #1e40af; margin: 0 0 0.5rem 0;">💰 Ersparnis</h4>
                <p style="font-size: 1.8rem; font-weight: 700; color: #10b981; margin: 0;">€{dashboard_data['estimated_savings']:.0f}</p>
                <p style="color: #6b7280; font-size: 0.9rem; margin: 0.25rem 0 0 0;">Geschätzt heute</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <h4 style="color: #1e40af; margin: 0 0 0.5rem 0;">📈 Effizienzrate</h4>
                <p style="font-size: 1.8rem; font-weight: 700; color: #059669; margin: 0;">{dashboard_data['efficiency_rate']:.1f}%</p>
                <p style="color: #6b7280; font-size: 0.9rem; margin: 0.25rem 0 0 0;">Effiziente Tests</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Wochentrend
        if dashboard_data['week_trend']:
            st.subheader("📈 7-Tage Trend")
            trend_df = pd.DataFrame(dashboard_data['week_trend'])
            
            col1, col2 = st.columns(2)
            with col1:
                st.line_chart(trend_df.set_index('date')['orders'])
                st.caption("Anzahl Bestellungen pro Tag")
            
            with col2:
                st.line_chart(trend_df.set_index('date')['efficiency'])
                st.caption("Durchschnittliche Effizienz pro Tag")
        
        # Top Diagnosen
        if dashboard_data['top_diagnoses']:
            st.subheader("🏥 Häufigste Diagnosen heute")
            diagnoses_df = pd.DataFrame(dashboard_data['top_diagnoses'])
            st.bar_chart(diagnoses_df.set_index('diagnosis')['count'])
        
        # Diagnose-Insights
        st.subheader("🔍 Diagnose-Analysen")
        diagnosis_insights = analytics.get_diagnosis_insights()
        
        if diagnosis_insights['insights']:
            for insight in diagnosis_insights['insights'][:5]:
                with st.expander(f"📊 {insight['diagnosis']} ({insight['frequency']} Fälle)"):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Durchschnittsalter", f"{insight['avg_age']:.1f} Jahre")
                    with col2:
                        st.metric("Effizienz", f"{insight['avg_efficiency']:.1f}/5")
                    with col3:
                        st.metric("Häufigkeit", f"{insight['frequency']} Fälle")
                    
                    if insight['common_tests']:
                        st.write("**Häufige Tests:**")
                        st.write(", ".join(insight['common_tests']))
        
        # Sample Data Button
        st.markdown("---")
        if st.button("📊 Beispieldaten erstellen (für Testing)"):
            with st.spinner("Erstelle Beispieldaten..."):
                db.create_sample_data()
            st.success("✅ Beispieldaten erfolgreich erstellt!")
            st.rerun()
            
    except Exception as e:
        st.error(f"Fehler beim Laden der Analytics: {e}")
        st.info("💡 Tipp: Erstellen Sie zuerst einige Laborbeauftragungen, um Analytics zu sehen.")


def render_ai_suggestions_page():
    """Rendert die KI-Empfehlungsseite."""
    st.markdown("""
    <div class="pro-header">
        <h1>🤖 KI-Empfehlungen</h1>
        <p>Intelligente Laborempfehlungen basierend auf ähnlichen Fällen</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); padding: 1.5rem; border-radius: 12px; border-left: 4px solid #0ea5e9; margin: 1rem 0;">
        <h4 style="color: #0c4a6e; margin: 0 0 0.5rem 0;">💡 Wie funktioniert es?</h4>
        <p style="color: #0c4a6e; margin: 0; font-size: 0.9rem;">
        Geben Sie Patientendaten ein, um KI-basierte Empfehlungen für Laborwerte zu erhalten. 
        Das System analysiert ähnliche Fälle aus der Datenbank und schlägt passende Tests vor.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Vereinfachtes Formular für KI-Empfehlungen
    with st.form("ai_recommendation_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input("Alter (Jahre):", min_value=0, max_value=120, value=50)
            gender = st.selectbox("Geschlecht:", GENDER_OPTIONS)
            diagnosis = st.selectbox("Verdachtsdiagnose:", COMMON_DIAGNOSES[1:])  # Ohne "Manuelle Eingabe"
            
        with col2:
            mts_category = st.selectbox("MTS-Kategorie:", list(MTS_CATEGORIES.keys()))
            symptom_duration = st.selectbox("Symptomdauer:", SYMPTOM_DURATION)
            pain_scale = st.slider("Schmerzskala:", 0, 10, 0)
        
        # Vitalparameter
        st.markdown("**Vitalparameter:**")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            systolic_bp = st.number_input("Systolischer BD:", min_value=60, max_value=250, value=120)
            heart_rate = st.number_input("Puls:", min_value=30, max_value=200, value=70)
        
        with col2:
            diastolic_bp = st.number_input("Diastolischer BD:", min_value=30, max_value=150, value=80)
            temperature = st.number_input("Temperatur:", min_value=35.0, max_value=42.0, value=36.5, step=0.1)
        
        with col3:
            respiratory_rate = st.number_input("Atemfrequenz:", min_value=5, max_value=60, value=16)
            oxygen_saturation = st.number_input("SpO2 (%):", min_value=70, max_value=100, value=98)
        
        submitted = st.form_submit_button("🤖 KI-Empfehlungen abrufen", type="primary")
    
    if submitted:
        # Erstelle Patientendaten
        patient_data = {
            "patient_data": {
                "age": age,
                "gender": gender,
                "mts_category": MTS_CATEGORIES[mts_category]
            },
            "clinical_data": {
                "suspected_diagnosis": diagnosis,
                "comorbidities": [],
                "symptom_duration": symptom_duration,
                "pain_scale": pain_scale,
                "additional_notes": ""
            },
            "vital_signs": {
                "blood_pressure": f"{systolic_bp}/{diastolic_bp}",
                "systolic_bp": systolic_bp,
                "diastolic_bp": diastolic_bp,
                "heart_rate": heart_rate,
                "temperature": temperature,
                "respiratory_rate": respiratory_rate,
                "oxygen_saturation": oxygen_saturation
            }
        }
        
        # Hole KI-Empfehlungen
        with st.spinner("🔍 Analysiere ähnliche Fälle und generiere Empfehlungen..."):
            recommendations = vector_store.get_recommendations(patient_data)
        
        if recommendations['similar_cases_count'] > 0:
            render_ai_recommendations(recommendations)
        else:
            st.warning("⚠️ Keine ähnlichen Fälle in der Datenbank gefunden. Erstellen Sie zuerst einige Laborbeauftragungen oder laden Sie Beispieldaten.")
            if st.button("📊 Beispieldaten laden"):
                with st.spinner("Lade Beispieldaten..."):
                    db.create_sample_data()
                st.success("✅ Beispieldaten geladen! Versuchen Sie es erneut.")
                st.rerun()


def render_chat_interface():
    """Rendert das ursprüngliche Chat-Interface."""
    st.header("Freier Chat")
    st.markdown("Hier können Sie frei mit dem medizinischen Assistenten chatten.")
    
    prompt = st.text_area("Nachricht:", height=150, placeholder="Stellen Sie hier Ihre Fragen...")
    run = st.button("Senden", use_container_width=True)

    if run and prompt.strip():
        try:
            config = configure_openai()
        except ValueError as e:
            st.error(str(e))
            return

        call_args = {
            **config,
            "messages": [
                {"role": "system", "content": "Du bist ein hilfreicher medizinischer Assistent. Gib immer den Hinweis, dass deine Antworten keine professionelle medizinische Beratung ersetzen."},
                {"role": "user", "content": prompt}
            ],
            "temperature": TEMPERATURE,
        }

        placeholder = st.empty()
        full_text = ""

        with placeholder.container():
            st.markdown("**Assistant:**")
            message_area = st.empty()

        try:
            for chunk in stream_completion(call_args):
                full_text += chunk
                message_area.markdown(full_text)
                time.sleep(0.02)
        except Exception as e:
            st.exception(e)


def main():
    st.set_page_config(
        page_title="Cliniq: Intelligente Laborbeauftragung", 
        layout="wide",
        initial_sidebar_state="expanded",
        page_icon="🏥"
    )
    
    # Minimales Professional Medical Styling
    st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    /* Metric Cards */
    .metric-card {
        background: var(--background-color);
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        border-left: 4px solid #3b82f6;
        margin: 0.5rem 0;
        transition: transform 0.2s ease;
        color: var(--text-color);
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.15);
    }
    
    /* Professional Header */
    .pro-header {
        background: linear-gradient(135deg, #1e40af 0%, #1e3a8a 100%);
        color: white;
        padding: 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 8px 32px rgba(30, 64, 175, 0.3);
    }
    
    .pro-header h1 {
        color: white !important;
        border: none !important;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
        margin-bottom: 0.5rem;
        font-family: 'Inter', sans-serif;
    }
    
    .pro-header p {
        font-size: 1.1rem;
        opacity: 0.9;
        margin: 0;
        font-family: 'Inter', sans-serif;
    }
    
    /* Enhanced Buttons (nur für Primary Buttons) */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #3b82f6 0%, #1e40af 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3) !important;
        font-family: 'Inter', sans-serif !important;
    }
    
    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #1d4ed8 0%, #1e3a8a 100%) !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 16px rgba(59, 130, 246, 0.4) !important;
    }
    
    /* Sidebar Medical Branding */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e40af 0%, #1e3a8a 100%) !important;
    }
    
    section[data-testid="stSidebar"] .stMarkdown {
        color: rgba(255, 255, 255, 0.9) !important;
    }
    
    section[data-testid="stSidebar"] .stRadio > div {
        background: rgba(255, 255, 255, 0.1) !important;
        border-radius: 12px !important;
        padding: 1rem !important;
        margin: 0.5rem 0 !important;
    }
    
    section[data-testid="stSidebar"] .stRadio label {
        color: white !important;
        font-weight: 500 !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Sidebar für Navigation
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 1rem 0;">
            <h1 style="color: white; font-size: 2rem; margin: 0; text-shadow: 0 2px 4px rgba(0,0,0,0.3);">🏥 Cliniq</h1>
            <p style="color: rgba(255,255,255,0.9); margin: 0.5rem 0 0 0; font-weight: 500;">Medical Intelligence Platform</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("---")
        
        # Navigation
        page = st.radio(
            "Navigation:",
            ["Laborbeauftragung", "KI-Empfehlungen", "Analytics Dashboard", "Konsultation"],
            index=0
        )
        
        st.markdown("---")
        st.markdown("""
        <div style="background: rgba(255,255,255,0.1); padding: 1rem; border-radius: 8px; backdrop-filter: blur(5px);">
            <h4 style="color: white; margin: 0 0 0.5rem 0;">⚙️ System-Status</h4>
            <p style="color: rgba(255,255,255,0.9); margin: 0.25rem 0; font-size: 0.9rem;">🤖 KI-Temperatur: {}</p>
            <p style="color: rgba(255,255,255,0.9); margin: 0.25rem 0; font-size: 0.9rem;">🔬 Modus: Labormedizin</p>
            <p style="color: rgba(255,255,255,0.9); margin: 0.25rem 0; font-size: 0.9rem;">💰 Kostenoptimierung: Aktiv</p>
        </div>
        """.format(TEMPERATURE), unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("""
        <div style="background: rgba(255,255,255,0.1); padding: 1rem; border-radius: 8px; backdrop-filter: blur(5px);">
            <h4 style="color: white; margin: 0 0 1rem 0;">📊 Live-Statistiken</h4>
            <div style="margin: 0.5rem 0;">
                <p style="color: rgba(255,255,255,0.7); margin: 0; font-size: 0.8rem;">Heute beauftragt</p>
                <p style="color: white; margin: 0; font-size: 1.2rem; font-weight: 600;">127 Tests</p>
            </div>
            <div style="margin: 0.5rem 0;">
                <p style="color: rgba(255,255,255,0.7); margin: 0; font-size: 0.8rem;">Kostenersparnis</p>
                <p style="color: #10b981; margin: 0; font-size: 1.2rem; font-weight: 600;">€2.340</p>
            </div>
            <div style="margin: 0.5rem 0;">
                <p style="color: rgba(255,255,255,0.7); margin: 0; font-size: 0.8rem;">Erfolgsquote</p>
                <p style="color: #22c55e; margin: 0; font-size: 1.2rem; font-weight: 600;">94.7%</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.error("**WICHTIG:** Nur medizinisch notwendige Tests beauftragen! Überflüssige Tests gefährden die Krankenhausfinanzierung.")
    
    # Hauptinhalt basierend auf Navigation
    if page == "Laborbeauftragung":
        # Professional Header
        st.markdown("""
        <div class="pro-header">
            <h1>🏥 Cliniq - Intelligente Laborbeauftragung</h1>
            <p>Kostenoptimierte Labordiagnostik für moderne Krankenhäuser</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Formular rendern
        form_result = render_laboratory_form()
        submitted, mts_category, gender, age, diagnosis, comorbidities, symptom_duration, pain_scale, additional_notes, systolic_bp, diastolic_bp, heart_rate, temperature, respiratory_rate, oxygen_saturation = form_result
        
        # Verarbeitung bei Formular-Übermittlung
        if submitted:
            if not diagnosis or diagnosis.strip() == "":
                st.warning("Bitte geben Sie eine Verdachtsdiagnose ein.")
            else:
                # JSON-Objekt aus Formulardaten erstellen
                patient_data = create_laboratory_json(
                    mts_category, gender, age, diagnosis, 
                    comorbidities, symptom_duration, pain_scale, additional_notes,
                    systolic_bp, diastolic_bp, heart_rate, temperature, respiratory_rate, oxygen_saturation
                )
                
                # Patientendaten anzeigen
                with st.expander("Erfasste Patientendaten (JSON)"):
                    st.json(patient_data)
                
                # OpenAI API-Aufruf für Laborbeauftragung
                try:
                    config = configure_openai()
                    
                    # Prompt für Laborbeauftragung erstellen
                    prompt = create_laboratory_prompt(patient_data)
                    
                    call_args = {
                        **config,
                        "messages": [
                            {"role": "system", "content": ASSISTANT_INSTRUCTION},
                            {"role": "user", "content": prompt}
                        ],
                        "temperature": TEMPERATURE,
                    }
                    
                    # KI-Empfehlungen basierend auf ähnlichen Fällen abrufen
                    with st.spinner("Analysiere ähnliche Fälle..."):
                        recommendations = vector_store.get_recommendations(patient_data)
                    
                    if recommendations['recommended_tests']:
                        st.info("🤖 **KI-Empfehlungen basierend auf ähnlichen Fällen gefunden!**")
                        with st.expander("📊 KI-Empfehlungen anzeigen"):
                            render_ai_recommendations(recommendations)
                    
                    # API-Aufruf mit Ladeanzeige
                    with st.spinner("Erstelle kostenoptimierte Laborbeauftragung..."):
                        response = get_completion(call_args)
                    
                    # Antwort parsen und Ergebnisse anzeigen
                    lab_result = parse_openai_response(response)
                    
                    # Speichere die Bestellung in der Datenbank
                    session_id = str(uuid.uuid4())
                    order_id = db.save_laboratory_order(patient_data, lab_result, session_id)
                    
                    # Speichere Embedding für zukünftige Empfehlungen
                    try:
                        vector_store.save_embedding(order_id, patient_data, lab_result)
                    except Exception as e:
                        st.warning(f"Embedding konnte nicht gespeichert werden: {e}")
                    
                    render_laboratory_results(lab_result)
                    
                    # Success message
                    st.success(f"✅ Laborbeauftragung erfolgreich gespeichert (ID: {order_id})")
                    
                except ValueError as e:
                    st.error(f"Konfigurationsfehler: {str(e)}")
                    st.info("Bitte überprüfen Sie Ihre Azure OpenAI-Einstellungen in der .env-Datei")
                except Exception as e:
                    st.error(f"Fehler bei der API-Anfrage: {str(e)}")
                    st.info("Überprüfen Sie Ihre Internetverbindung und API-Konfiguration.")
    
    elif page == "Analytics Dashboard":
        render_analytics_dashboard()
        
    elif page == "KI-Empfehlungen":
        render_ai_suggestions_page()
        
    elif page == "Konsultation":
        # Professional Header for Consultation
        st.markdown("""
        <div class="pro-header">
            <h1>💬 Medizinische Konsultation</h1>
            <p>KI-gestützte medizinische Beratung und Entscheidungsunterstützung</p>
        </div>
        """, unsafe_allow_html=True)
        render_chat_interface()


if __name__ == "__main__":
    main()
