
import streamlit as st
import pickle
import re
import docx2txt
import spacy
import fitz  # PyMuPDF for PDF
from sklearn.feature_extraction.text import TfidfVectorizer

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
tfidf = pickle.load(open("tfidf.pkl", "rb"))
nlp = spacy.load("en_core_web_sm")

st.set_page_config(page_title="Resume Classifier", layout="wide")

st.title("🧠 Smart Resume Analyzer")
st.write("Upload a resume to classify the job role and extract details.")

# -------------------- Helper Functions -------------------- #

def extract_text(file):
    if file.name.endswith(".pdf"):
        doc = fitz.open(stream=file.read(), filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    elif file.name.endswith(".docx"):
        return docx2txt.process(file)
    elif file.name.endswith(".txt"):
        return file.read().decode("utf-8")
    else:
        return ""

def extract_name(text):
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            return ent.text
    return "Not found"

def extract_qualification(text):
    degrees = ["B.Tech", "M.Tech", "MBA", "B.E", "MCA", "PhD", "B.Sc", "M.Sc"]
    found = [deg for deg in degrees if deg.lower() in text.lower()]
    return ", ".join(set(found)) if found else "Not found"

def extract_skills(text):
    keywords = ["python", "java", "excel", "sql", "power bi", "ml", "nlp",
                "react", "django", "tensorflow", "keras", "pandas", "numpy"]
    skills = [kw for kw in keywords if kw in text.lower()]
    return ", ".join(set(skills)) if skills else "Not found"

# -------------------- Streamlit App -------------------- #

uploaded_file = st.file_uploader("📤 Upload Resume", type=["pdf", "docx", "txt"])

if uploaded_file:
    with st.spinner("Processing..."):
        raw_text = extract_text(uploaded_file)
        cleaned_text = re.sub(r"[^a-zA-Z0-9 ]", " ", raw_text)
        cleaned_text = re.sub(r"\s+", " ", cleaned_text)

        # Extract Info
        name = extract_name(raw_text)
        qual = extract_qualification(raw_text)
        skills = extract_skills(raw_text)

        # Predict Category
        X_vec = tfidf.transform([cleaned_text])
        pred = model.predict(X_vec)[0]

    # Display Results
    st.subheader("📌 Prediction Result")
    st.success(f"Predicted Job Role: **{pred}**")

    st.subheader("📋 Extracted Resume Details")
    col1, col2 = st.columns(2)
    col1.markdown(f"**👤 Name:** {name}")
    col1.markdown(f"**🎓 Qualification:** {qual}")
    col2.markdown(f"**🛠 Skills:** {skills}")
    st.subheader("📝 Raw Text Preview")
    st.text_area("Resume Content", raw_text[:3000], height=250)

    st.markdown("---")
    st.info("Deploy this on Streamlit Cloud after pushing to GitHub with `model.pkl` and `tfidf.pkl`.")

