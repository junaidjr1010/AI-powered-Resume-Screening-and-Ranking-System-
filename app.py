import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- Streamlit App Title ---
st.title("AI-Powered Resume Screening & Ranking System 📄✨")

# --- File Upload Section ---
uploaded_files = st.file_uploader("Upload Resumes (PDF)", type="pdf", accept_multiple_files=True)

# --- Job Description Input ---
job_description = st.text_area("Enter Job Description Here", height=150)

if uploaded_files and job_description:
    resumes_texts = []
    file_names = []

    # Extract text from each resume
    for uploaded_file in uploaded_files:
        pdf_reader = PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + " "
        resumes_texts.append(text)
        file_names.append(uploaded_file.name)

    # --- Convert Resumes & Job Description to TF-IDF Vectors ---
    vectorizer = TfidfVectorizer()
    all_texts = [job_description] + resumes_texts
    tfidf_matrix = vectorizer.fit_transform(all_texts)

    # --- Compute Similarity Scores ---
    job_vector = tfidf_matrix[0]  # First item is the job description
    resume_vectors = tfidf_matrix[1:]  # Remaining are resumes
    similarity_scores = cosine_similarity(job_vector, resume_vectors).flatten()

    # --- Rank Resumes Based on Similarity ---
    ranked_resumes = sorted(zip(file_names, similarity_scores), key=lambda x: x[1], reverse=True)

    # --- Display Ranked Resumes ---
    st.subheader("📌 Resume Ranking:")
    for rank, (file, score) in enumerate(ranked_resumes, start=1):
        st.write(f"**{rank}. {file}** - Match Score: `{score:.2f}`")

    st.success("✅ Resume screening completed successfully!")
