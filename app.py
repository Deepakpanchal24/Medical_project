import streamlit as st
import datetime
from model_utils import load_and_merge_data, embedding_model
from diagnosis_engine import diagnose_patient
from database import init_db, view_chat_history
from calendar_utils import calculate_age

# Initialize session state for input values
if 'patient_name' not in st.session_state:
    st.session_state.patient_name = ""
if 'birth_date' not in st.session_state:
    st.session_state.birth_date = None
if 'weight' not in st.session_state:
    st.session_state.weight = ""
if 'gender' not in st.session_state:
    st.session_state.gender = "Male"
if 'symptoms' not in st.session_state:
    st.session_state.symptoms = ""
if 'medical_history' not in st.session_state:
    st.session_state.medical_history = ""
if 'test_results' not in st.session_state:
    st.session_state.test_results = ""
if 'physician_query' not in st.session_state:
    st.session_state.physician_query = ""

# Initialize DB (runs once, minimal overhead)
init_db()

# Cache the FAISS index and data loading
@st.cache_resource
def load_data():
    csv_path = "MediMind_cleaned_utf8.csv"
    pdf_path = "Diagnostic.pdf"
    try:
        faiss_index, texts, csv_data = load_and_merge_data(csv_path, pdf_path)
        if faiss_index is None:
            st.error("Failed to load data. Check CSV and PDF files.")
            return None, [], None
        return faiss_index, texts, csv_data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, [], None

# Load cached data
faiss_index, texts, csv_data = load_data()

# Cache the diagnosis function
@st.cache_data(show_spinner=False)
def cached_diagnose_patient(_patient_name, _age, _weight, _gender, _symptoms, _medical_history, _test_results, _physician_query):
    if faiss_index is None or not texts:
        return "Error: Data not loaded. Please check CSV and PDF files."
    return diagnose_patient(
        _patient_name, _age, _weight, _gender,
        _symptoms, _medical_history, _test_results, _physician_query,
        embedding_model=embedding_model,
        faiss_index=faiss_index,
        texts=texts
    )

# Wrapper function for diagnosis
def diagnose_wrapper(patient_name, birth_date, weight, gender, symptoms, medical_history, test_results, physician_query):
    age = calculate_age(birth_date) if birth_date else 0
    return cached_diagnose_patient(
        patient_name, age, weight, gender,
        symptoms, medical_history, test_results, physician_query
    )

# Streamlit UI
st.title("🧠 MediMind Diagnostic Assistant")

with st.container():
    st.subheader("Diagnosis")
    st.write("Enter patient details and query below:")

    # Use session state to control widget values
    patient_name = st.text_input("Patient Name", value=st.session_state.patient_name, key="input_patient_name")
    birth_date = st.date_input(
        "Date of Birth",
        value=st.session_state.birth_date,
        min_value=datetime.date(1900, 1, 1),
        max_value=datetime.date.today(),
        help="Select the patient's date of birth",
        key="input_birth_date"
    )
    age = calculate_age(birth_date) if birth_date else 0
    st.write(f"Age: {age} years")
    weight = st.text_input("Weight (kg)", value=st.session_state.weight, key="input_weight")
    gender = st.selectbox("Gender", options=["Male", "Female", "Other"], index=["Male", "Female", "Other"].index(st.session_state.gender), key="input_gender")
    symptoms = st.text_area("Symptoms", value=st.session_state.symptoms, key="input_symptoms")
    medical_history = st.text_area("Medical History", value=st.session_state.medical_history, key="input_medical_history")
    test_results = st.text_area("Test Results", value=st.session_state.test_results, key="input_test_results")
    physician_query = st.text_area("Physician Query", value=st.session_state.physician_query, key="input_physician_query")

    # Update session state with current input values
    st.session_state.patient_name = patient_name
    st.session_state.birth_date = birth_date
    st.session_state.weight = weight
    st.session_state.gender = gender
    st.session_state.symptoms = symptoms
    st.session_state.medical_history = medical_history
    st.session_state.test_results = test_results
    st.session_state.physician_query = physician_query

    col1, col2 = st.columns(2)
    with col1:
        submit_btn = st.button("Submit")
    with col2:
        clear_btn = st.button("Clear")

    if submit_btn:
        if not patient_name or not physician_query or not birth_date:
            st.error("Please provide patient name, date of birth, and physician query.")
        else:
            with st.spinner("Generating diagnosis..."):
                response = diagnose_wrapper(
                    patient_name, birth_date, weight, gender,
                    symptoms, medical_history, test_results, physician_query
                )
                st.text_area("MediMind Diagnostic Response", response, height=400, key="diagnosis_output")

    if clear_btn:
        # Clear session state values
        st.session_state.patient_name = ""
        st.session_state.birth_date = None
        st.session_state.weight = ""
        st.session_state.gender = "Male"
        st.session_state.symptoms = ""
        st.session_state.medical_history = ""
        st.session_state.test_results = ""
        st.session_state.physician_query = ""
        st.rerun()

with st.container():
    st.subheader("Chat History")
    if st.button("Refresh History"):
        with st.spinner("Loading chat history..."):
            history = view_chat_history()
            st.text_area("Chat History", history, height=300, disabled=True, key="chat_history_output")