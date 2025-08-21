import streamlit as st
import pandas as pd
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
import os
from dotenv import load_dotenv
from openai import OpenAI
import json
import io

# --- Page Configuration ---
st.set_page_config(page_title="AI Data Cleaning Assistant", layout="wide")
st.title("🤖 AI-Powered Data Cleaning Assistant")
st.write("Upload your messy medical dataset to begin the analysis.")

# --- Load API Keys ---
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- Agent Functions ---
# (Your existing analyze_medical_terminology and generate_cleaning_code functions go here - no changes needed)
@st.cache_data
def analyze_medical_terminology(df_sample):
    text_columns = ['provisionaldiagnosis', 'chief_remark', 'clinical_note', 'finaldiagnosis']
    combined_text = ""
    for col in text_columns:
        if col in df_sample.columns:
            sample = " ".join(df_sample[col].dropna().head(50).astype(str).tolist())
            combined_text += sample + " "
    if not combined_text.strip(): return {"error": "No text data found."}
    prompt = f"""You are an expert medical terminologist. Your first and most important instruction is to return your findings ONLY as a single, valid JSON object. Analyze the following sample of clinical text. Identify three types of potential data quality issues: 1. Spelling Errors: Common misspellings of medical terms. 2. Abbreviations: Common medical abbreviations that should be expanded (e.g., "MI", "SOB"). 3. Synonyms: Different terms used for the same condition (e.g., "HTN", "high blood pressure"). TEXT SAMPLE: "{combined_text[:4000]}" For each key ("spelling_errors", "abbreviations", "synonyms"), provide a list of identified terms. For example: {{"spelling_errors": [["diabetis", "diabetes"]], "abbreviations": [["MI", "Myocardial Infarction"]], "synonyms": [["HTN", "high blood pressure", "hypertension"]]}}"""
    try:
        response = client.chat.completions.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": prompt}], response_format={"type": "json_object"}, temperature=0)
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        return {"error": f"Failed to call OpenAI API: {e}"}

def generate_cleaning_code(selections):
    tasks_string = "\n".join([f"- {task}" for task in selections])
    prompt = f"""You are an expert Python data scientist specializing in the pandas library. Your task is to generate a Python function that cleans a pandas DataFrame based on a list of user-approved tasks. The function signature MUST be: def clean_data(df):. Here are the cleaning tasks you must implement:\n{tasks_string}\nIMPORTANT RULES: 1. The function must take a pandas DataFrame `df` as input and return the modified DataFrame. 2. Use `df.replace()` or `df[column].str.replace()` for substitutions. Be precise. For abbreviations like 'MI', use regex to ensure you only replace the whole word, for example: `r'\\bMI\\b'`. 3. The text columns to clean are: 'provisionaldiagnosis', 'chief_remark', 'clinical_note', 'finaldiagnosis'. You must apply the cleaning operations to all of these columns. 4. The entire response must be a single, raw Python code block. Do not use markdown like ```python. Do not add any explanatory text."""
    try:
        response = client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": prompt}], temperature=0)
        return response.choices[0].message.content
    except Exception as e:
        return f"# An error occurred during code generation: {e}"

# --- FILE UPLOADER AND WORKFLOW ---
uploaded_file = st.file_uploader("Choose a file", type=['csv', 'xlsx'])
if uploaded_file is not None:
    try:
        if 'df' not in st.session_state or st.session_state.get('file_name') != uploaded_file.name:
            st.info(f"Reading file '{uploaded_file.name}'...")
            if uploaded_file.name.endswith('.xlsx'): df = pd.read_excel(uploaded_file)
            else: df = pd.read_csv(uploaded_file, encoding='latin1')
            st.session_state['df'] = df
            st.session_state['file_name'] = uploaded_file.name
            st.success("File read successfully!")
            # Clear session state on new upload
            keys_to_clear = ['profile_report', 'terminology_issues', 'user_selections', 'generated_code', 'cleaned_df', 'quality_report']
            for key in keys_to_clear:
                st.session_state.pop(key, None)

        st.write("### Data Preview:"); st.dataframe(st.session_state['df'].head())
        if st.button("Analyze Data Quality (Agent 1)"):
            with st.spinner("Data Profiler Agent is analyzing..."):
                profile = ProfileReport(st.session_state['df'], title="Data Quality Profile", explorative=True)
                st.session_state['profile_report'] = profile
            st.success("Data quality analysis complete!")
    except Exception as e: st.error(f"An error occurred: {e}")

# --- DISPLAY REPORTS, USER SELECTION, CODE GEN ---
if 'profile_report' in st.session_state:
    st.header("Phase 1 & 2: Data Profile and Terminology Issues")
    st.write("### Comprehensive Data Quality Report:"); st_profile_report(st.session_state['profile_report'])
    st.write("---")
    if st.button("Analyze Medical Terminology (Agent 2)"):
        with st.spinner("Medical Knowledge Agent is analyzing terminology..."):
            issues = analyze_medical_terminology(st.session_state['df'])
            st.session_state['terminology_issues'] = issues
        st.success("Terminology analysis complete!")

if 'terminology_issues' in st.session_state:
    issues = st.session_state['terminology_issues']
    if "error" in issues: st.error(issues["error"])
    else:
        st.write("### Medical Terminology Issues:"); st.json(issues)
        st.header("Phase 3: Review and Select Cleaning Actions")
        selections = {}
        with st.form("cleaning_form"):
            st.write("Select the cleaning operations you want to perform:")
            for issue_type, items in issues.items():
                if items:
                    st.subheader(f"Fix {issue_type.replace('_', ' ').title()}")
                    for idx, item in enumerate(items):
                        label = f"{item[0]} → {item[1]}"
                        unique_key = f"{issue_type}-{item[0]}-{idx}"
                        selections[label] = st.checkbox(label, value=True, key=unique_key)
            submitted = st.form_submit_button("Generate Cleaning Plan")
            if submitted:
                st.session_state['user_selections'] = [k for k, v in selections.items() if v]
                st.success("Your selections have been saved!")

if 'user_selections' in st.session_state:
    st.header("Phase 4: Processing Strategy & Code Generation")
    st.write("### Your Selections:"); st.json(st.session_state['user_selections'])
    if st.button("Generate Cleaning Code (Agent 4)"):
        with st.spinner("Code Generation Agent is writing a Python script..."):
            code = generate_cleaning_code(st.session_state['user_selections'])
            st.session_state['generated_code'] = code
        st.success("Cleaning code generated!")

# --- PHASE 5: EXECUTION AND DELIVERY ---
if 'generated_code' in st.session_state:
    st.write("### Generated Python Code:"); st.code(st.session_state['generated_code'], language='python')
    if st.button("Execute Cleaning Code and Validate (Agent 5)"):
        with st.spinner("Executing cleaning script and performing QA..."):
            original_df = st.session_state['df']
            code_to_exec = st.session_state['generated_code']

            # Create a copy to clean
            df_to_clean = original_df.copy()

            # Create a dictionary to hold the dynamically defined function
            local_scope = {}

            # Execute the generated code string. This defines the `clean_data` function inside local_scope
            exec(code_to_exec, {}, local_scope)

            # Get the function from the scope
            clean_data_func = local_scope['clean_data']

            # Run the function to get the cleaned dataframe
            cleaned_df = clean_data_func(df_to_clean)
            st.session_state['cleaned_df'] = cleaned_df

            # QA Agent: Compare the dataframes to generate a report
            comparison = original_df.compare(cleaned_df)
            issues_resolved = len(comparison)
            quality_report = {
                "Total Rows Processed": len(original_df),
                "Total Data Points (Cells) Changed": int(comparison.size / 2),
                "Number of Rows with Changes": issues_resolved,
            }
            st.session_state['quality_report'] = quality_report
        st.success("Execution and validation complete!")

if 'quality_report' in st.session_state:
    st.header("Phase 5: Final Output")
    st.write("### Quality Assurance Report:")
    st.json(st.session_state['quality_report'])

    st.write("### Cleaned Data Preview:")
    st.dataframe(st.session_state['cleaned_df'].head())

    # Convert cleaned dataframe to CSV for download
    output_csv = st.session_state['cleaned_df'].to_csv(index=False).encode('utf-8')

    st.download_button(
       label="📥 Download Cleaned CSV",
       data=output_csv,
       file_name="cleaned_medical_data.csv",
       mime="text/csv",
    )