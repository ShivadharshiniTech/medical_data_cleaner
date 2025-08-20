import streamlit as st
import pandas as pd
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report # This line is changed

# --- Page Configuration ---
st.set_page_config(
    page_title="AI Data Cleaning Assistant",
    layout="wide"
)

# --- Header ---
st.title("🤖 AI-Powered Data Cleaning Assistant")
st.write("Upload your messy medical dataset (Excel or CSV) to begin the analysis.")

# --- File Uploader ---
uploaded_file = st.file_uploader(
    "Choose a file",
    type=['csv', 'xlsx']
)

# --- Main Logic ---
if uploaded_file is not None:
    try:
        if 'df' not in st.session_state or st.session_state.get('file_name') != uploaded_file.name:
            st.info(f"Reading file '{uploaded_file.name}'...")
            if uploaded_file.name.endswith('.xlsx'):
                df = pd.read_excel(uploaded_file)
            else:
                df = pd.read_csv(uploaded_file)
            
            st.session_state['df'] = df
            st.session_state['file_name'] = uploaded_file.name
            st.success("File read successfully!")

        st.write("### Data Preview:")
        st.dataframe(st.session_state['df'].head())

        # --- Agent Trigger ---
        if st.button("Analyze Data Quality (Agent 1)"):
            with st.spinner("Data Profiler Agent is analyzing the data... This may take a moment."):
                profile = ProfileReport(
                    st.session_state['df'],
                    title="Data Quality Profile",
                    explorative=True
                )
                st.session_state['profile_report'] = profile
            
            st.success("Analysis complete!")

    except Exception as e:
        st.error(f"An error occurred: {e}")

# Display the report if it exists in the session state
if 'profile_report' in st.session_state:
    st.write("### Comprehensive Data Quality Report:")
    st_profile_report(st.session_state['profile_report'])