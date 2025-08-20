import streamlit as st
import pandas as pd
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
import os
from dotenv import load_dotenv
from openai import OpenAI  # Use the OpenAI library
import json

# --- Page Configuration ---
st.set_page_config(page_title="AI Data Cleaning Assistant", layout="wide")
st.title("🤖 AI-Powered Data Cleaning Assistant")
st.write("Upload your messy medical dataset to begin the analysis.")

# --- Load API Keys ---
load_dotenv()
# We only need the OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- Agent Function ---
def analyze_medical_terminology(df):
    """
    Uses OpenAI API to analyze text columns for medical terminology issues.
    """
    # Combine text from relevant columns into one sample string
    text_columns = ['provisionaldiagnosis', 'chief_remark', 'clinical_note', 'finaldiagnosis']
    combined_text = ""
    for col in text_columns:
        if col in df.columns:
            sample = " ".join(df[col].dropna().head(50).astype(str).tolist())
            combined_text += sample + " "

    if not combined_text.strip():
        return {"error": "No text data found in the relevant columns."}

    # Craft the prompt for OpenAI
    prompt = f"""
    You are an expert medical terminologist. Analyze the following sample of clinical text.
    Identify three types of potential data quality issues:
    1.  **Spelling Errors**: Common misspellings of medical terms.
    2.  **Abbreviations**: Common medical abbreviations that should be expanded (e.g., "MI", "SOB").
    3.  **Synonyms**: Different terms used for the same condition (e.g., "HTN", "high blood pressure").

    TEXT SAMPLE: "{combined_text[:4000]}"  # Limit text size for API

    Return your findings ONLY as a single, valid JSON object with the keys "spelling_errors", "abbreviations", and "synonyms". 
    For each key, provide a list of identified terms. For example: 
    {{
      "spelling_errors": [["diabetis", "diabetes"]],
      "abbreviations": [["MI", "Myocardial Infarction"]],
      "synonyms": [["HTN", "high blood pressure", "hypertension"]]
    }}
    """

    try:
        # Call the OpenAI API
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}, # Use JSON mode for reliability
            temperature=0,
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        return {"error": f"Failed to call OpenAI API: {e}"}


# --- File Uploader ---
uploaded_file = st.file_uploader("Choose a file", type=['csv', 'xlsx'])

# --- Main Logic ---
if uploaded_file is not None:
    try:
        if 'df' not in st.session_state or st.session_state.get('file_name') != uploaded_file.name:
            st.info(f"Reading file '{uploaded_file.name}'...")
            if uploaded_file.name.endswith('.xlsx'):
                df = pd.read_excel(uploaded_file)
            else:
                df = pd.read_csv(uploaded_file, encoding='latin1')
            
            st.session_state['df'] = df
            st.session_state['file_name'] = uploaded_file.name
            st.success("File read successfully!")
            if 'profile_report' in st.session_state: del st.session_state['profile_report']
            if 'terminology_issues' in st.session_state: del st.session_state['terminology_issues']

        st.write("### Data Preview:")
        st.dataframe(st.session_state['df'].head())

        # --- Agent 1 Trigger ---
        if st.button("Analyze Data Quality (Agent 1)"):
            with st.spinner("Data Profiler Agent is analyzing... This may take a moment."):
                profile = ProfileReport(st.session_state['df'], title="Data Quality Profile", explorative=True)
                st.session_state['profile_report'] = profile
            st.success("Data quality analysis complete!")

    except Exception as e:
        st.error(f"An error occurred: {e}")

# --- Display Reports ---
if 'profile_report' in st.session_state:
    st.write("### Comprehensive Data Quality Report:")
    st_profile_report(st.session_state['profile_report'])
    st.write("---")

    # --- Agent 2 Trigger ---
    if st.button("Analyze Medical Terminology (Agent 2)"):
        with st.spinner("Medical Knowledge Agent is analyzing terminology with OpenAI..."):
            issues = analyze_medical_terminology(st.session_state['df'])
            st.session_state['terminology_issues'] = issues
        st.success("Terminology analysis complete!")

if 'terminology_issues' in st.session_state:
    st.write("### Medical Terminology Issues:")
    st.json(st.session_state['terminology_issues'])