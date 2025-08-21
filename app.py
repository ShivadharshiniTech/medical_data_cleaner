import streamlit as st
import pandas as pd
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
import os
from dotenv import load_dotenv
from openai import OpenAI
import json

# --- Page Configuration ---
st.set_page_config(page_title="AI Data Cleaning Assistant", layout="wide")
st.title("🤖 AI-Powered Data Cleaning Assistant")
st.write("Upload your messy medical dataset to begin the analysis.")

# --- Load API Keys ---
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- Agent Function ---
@st.cache_data # Cache the results to avoid re-running the API call
def analyze_medical_terminology(df_sample):
    text_columns = ['provisionaldiagnosis', 'chief_remark', 'clinical_note', 'finaldiagnosis']
    combined_text = ""
    for col in text_columns:
        if col in df_sample.columns:
            sample = " ".join(df_sample[col].dropna().head(50).astype(str).tolist())
            combined_text += sample + " "

    if not combined_text.strip():
        return {"error": "No text data found in the relevant columns."}

    prompt = f"""
    You are an expert medical terminologist. Your first and most important instruction is to return your findings ONLY as a single, valid JSON object.

        Analyze the following sample of clinical text. Identify three types of potential data quality issues:
        1.  **Spelling Errors**: Common misspellings of medical terms.
        2.  **Abbreviations**: Common medical abbreviations that should be expanded (e.g., "MI", "SOB").
        3.  **Synonyms**: Different terms used for the same condition (e.g., "HTN", "high blood pressure").

        TEXT SAMPLE: "{combined_text[:4000]}"  # Limit text size for API

        For each key ("spelling_errors", "abbreviations", "synonyms"), provide a list of identified terms. For example: 
        {{
          "spelling_errors": [["diabetis", "diabetes"]],
          "abbreviations": [["MI", "Myocardial Infarction"]],
          "synonyms": [["HTN", "high blood pressure", "hypertension"]]
        }}
    """ 

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
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
            # Clear old results
            st.session_state.pop('profile_report', None)
            st.session_state.pop('terminology_issues', None)
            st.session_state.pop('user_selections', None)

        st.write("### Data Preview:")
        st.dataframe(st.session_state['df'].head())

        if st.button("Analyze Data Quality (Agent 1)"):
            with st.spinner("Data Profiler Agent is analyzing..."):
                profile = ProfileReport(st.session_state['df'], title="Data Quality Profile", explorative=True)
                st.session_state['profile_report'] = profile
            st.success("Data quality analysis complete!")
    except Exception as e:
        st.error(f"An error occurred: {e}")

# --- Display Reports and User Selection ---
if 'profile_report' in st.session_state:
    st.header("Phase 1 & 2: Data Profile and Terminology Issues")
    st.write("### Comprehensive Data Quality Report:")
    st_profile_report(st.session_state['profile_report'])
    st.write("---")

    if st.button("Analyze Medical Terminology (Agent 2)"):
        with st.spinner("Medical Knowledge Agent is analyzing terminology..."):
            issues = analyze_medical_terminology(st.session_state['df'])
            st.session_state['terminology_issues'] = issues
        st.success("Terminology analysis complete!")

if 'terminology_issues' in st.session_state:
    st.write("### Medical Terminology Issues:")
    issues = st.session_state['terminology_issues']
    if "error" in issues:
        st.error(issues["error"])
    else:
        st.json(issues)

        # --- Phase 3: User Review ---
        st.header("Phase 3: Review and Select Cleaning Actions")
        selections = {}
        with st.form("cleaning_form"):
            st.write("Select the cleaning operations you want to perform:")

            # Dynamically create checkboxes
            for issue_type, items in issues.items():
                if items: # Only show if there are issues of this type
                    st.subheader(f"Fix {issue_type.replace('_', ' ').title()}")
                    for item in items:
                        label = f"{item[0]} → {item[1]}"
                        unique_key = f"{issue_type}-{item[0]}"
                        selections[label] = st.checkbox(label, value=True, key=unique_key)

            submitted = st.form_submit_button("Generate Cleaning Plan")
            if submitted:
                st.session_state['user_selections'] = {k: v for k, v in selections.items() if v}
                st.success("Your selections have been saved!")

if 'user_selections' in st.session_state:
    st.write("### Your Selections:")
    st.json(st.session_state['user_selections'])