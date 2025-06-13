import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from database import save_chat_history

from dotenv import load_dotenv
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
llm = ChatGroq(model="llama-3.1-8b-instant", groq_api_key=groq_api_key, temperature=0.0)

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are MediMind, an AI assistant for physicians. Analyze patient data and provide:

- For pediatric patients (under 18 years):
    * Adjust drug dosages based on weight if available (e.g., ibuprofen 10 mg/kg).
    * Consider age-specific normal ranges (e.g., vitals, developmental milestones).
    * Highlight pediatric red flags (e.g., poor feeding, lethargy, inconsolable crying).
    * Recommend non-invasive or child-friendly interventions when possible.
    * Provide caregiver communication tips for home care and monitoring.
    
- Symptom Analysis: List up to 3 differential diagnoses with probabilities and reasoning.

- Clinical Decision Support: Suggest immediate actions, treatment plan, follow-ups, and documentation.

Context: {context}
Question: {question}

Format response as:

[Symptom Analysis]

- [Diagnosis 1]: X% – Reason
- [Diagnosis 2]: X% – Reason
- [Diagnosis 3]: X% – Reason

[Clinical Decision Support]

- Immediate Actions: [Actions]

- Treatment Plan:
    * Medications with weight-based dosing (if pediatric)
    * Non-pharmacological strategies (if applicable)
    * Safety precautions and comfort measures

- Follow-Ups:
    * What to monitor at home
    * When to return or escalate (pediatric-specific warnings)

- Documentation:
    * Summary of findings
    * Age-specific assessments
    * Parental/caregiver instructions
    """
)

def search_similar_cases(query, faiss_index, texts, embedding_model, top_k=3):
    query_embedding = embedding_model.encode([query], convert_to_numpy=True)
    distances, indices = faiss_index.search(query_embedding, k=top_k)
    results = [texts[i] for i in indices[0] if i < len(texts)]
    return results

def diagnose_patient(patient_name, age, weight, gender, symptoms, medical_history, test_results, physician_query,
                     faiss_index, texts, embedding_model):
    if not patient_name or not physician_query:
        return "Please provide patient name and query."
    
    query = (
        f"Patient: {patient_name}\n"
        f"Age: {age}\n"
        f"Weight: {weight} kg\n"
        f"Gender: {gender}\n"
        f"Symptoms: {symptoms}\n"
        f"Medical History: {medical_history}\n"
        f"Test Results: {test_results}\n"
        f"Query: {physician_query}"
    )
    
    context_list = search_similar_cases(query, faiss_index, texts, embedding_model)
    context = "\n".join(context_list) if context_list else ""
    
    if not context:
        return "No matching records found. Consider general evaluation."
    
    response = llm.invoke(prompt.format(context=context, question=query)).content
    
    # Save chat history
    save_chat_history(patient_name, query, response)
    
    return response