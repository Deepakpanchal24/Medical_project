import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
import numpy as np
import streamlit as st

# Cache the model globally
@st.cache_resource
def get_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2', device='cpu')

embedding_model = get_embedding_model()

def load_csv_data(csv_path):
    try:
        df = pd.read_csv(csv_path)
        if not all(col in df.columns for col in ['disease', 'symptoms', 'treatment_plan']):
            raise ValueError("CSV missing required columns: disease, symptoms, treatment_plan")
        texts = [f"Disease: {row['disease']}\nSymptoms: {row['symptoms']}\nTreatment: {row['treatment_plan']}"
                 for _, row in df.iterrows()]
        with st.spinner("Generating CSV embeddings..."):
            embeddings = embedding_model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)
        return index, texts, df
    except FileNotFoundError:
        st.error(f"CSV file {csv_path} not found.")
        return None, [], pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
        return None, [], pd.DataFrame()

def load_pdf_data(pdf_path):
    try:
        reader = PdfReader(pdf_path)
        pages = [page.extract_text() for page in reader.pages if page.extract_text()]
        if not pages:
            st.warning("No text extracted from PDF.")
            return None, []
        texts = pages
        with st.spinner("Generating PDF embeddings..."):
            embeddings = embedding_model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)
        return index, texts
    except FileNotFoundError:
        st.error(f"PDF file {pdf_path} not found.")
        return None, []
    except Exception as e:
        st.error(f"Error loading PDF: {e}")
        return None, []

def merge_faiss_indexes(index1, index2):
    if index1 is None and index2 is None:
        return None
    if index1 is None:
        return index2
    if index2 is None:
        return index1
    merged_index = faiss.IndexFlatL2(index1.d)
    merged_index.add(index1.reconstruct_n(0, index1.ntotal))
    merged_index.add(index2.reconstruct_n(0, index2.ntotal))
    return merged_index

def load_and_merge_data(csv_path, pdf_path):
    index_csv, texts_csv, df_csv = load_csv_data(csv_path)
    index_pdf, texts_pdf = load_pdf_data(pdf_path)
    merged_index = merge_faiss_indexes(index_csv, index_pdf)
    merged_texts = texts_csv + texts_pdf
    return merged_index, merged_texts, df_csv