import streamlit as st
import google.generativeai as genai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fpdf import FPDF
import tempfile
import base64

st.set_page_config(page_title="Plagiarism Detector", page_icon="📝", layout="centered")

with st.sidebar:
    st.title("🔑 API Key Required")
    api_key = st.text_input("Enter API Key", type="password")
    st.markdown("[Get Your API Key](https://makersuite.google.com/app/apikey)")

if api_key:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-pro')
else:
    st.warning("⚠️ Please enter your API key to proceed.")

st.markdown("<h1 style='text-align: center;'>📝 AI-Powered Plagiarism Detector</h1>", unsafe_allow_html=True)
st.write("🔍 **Detect plagiarism** using AI and NLP techniques with instant similarity scoring.")

if "text_input" not in st.session_state:
    st.session_state["text_input"] = ""
if "plagiarism_report" not in st.session_state:
    st.session_state["plagiarism_report"] = ""

def check_plagiarism(input_text):
    prompt = f"Check if the following text is plagiarized. If similar sources exist, return a rewritten version and the percentage of similarity:\n\n{input_text}"
    try:
        response = model.generate_content(prompt)
        return response.text.strip() if response and hasattr(response, 'text') else "No response generated."
    except Exception as e:
        return f"Error: {str(e)}"

def compute_similarity(original, generated):
    if not original.strip() or not generated.strip():
        return 0.0
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([original, generated])
    similarity = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])
    return similarity[0][0] * 100

text_to_check = st.text_area("\ud83d\udccc **Paste Your Text Below:**", value=st.session_state["text_input"], height=200, 
                             placeholder="Enter your text here...", key="text_input")

col1, col2, col3 = st.columns(3)

if col3.button("🚓 Stop", use_container_width=True):
    st.warning("Process stopped!")

if col2.button("❌ Clear Text", use_container_width=True):
    del st.session_state["text_input"]
    del st.session_state["plagiarism_report"]
    st.rerun()

if col1.button("🔎 Check Plagiarism", use_container_width=True):
    if not api_key:
        st.warning("⚠️ Please enter your API key to proceed.")
    elif not text_to_check.strip():
        st.warning("⚠️ Please enter valid text to check for plagiarism.")
    else:
        with st.spinner("🔍 Analyzing text for plagiarism..."):
            ai_result = check_plagiarism(text_to_check)
            similarity_score = compute_similarity(text_to_check, ai_result) if isinstance(ai_result, str) else 0
            st.session_state["plagiarism_report"] = f"Rewritten Text:\n\n{ai_result}\n\nPlagiarism Score: {similarity_score:.2f}%"
            st.success("✅ **Analysis Complete!**")
            st.subheader("🔍 Plagiarism Report")
            st.write(f"**Rewritten Text:**\n\n{ai_result}")
            st.metric(label="🚗 Plagiarism Score", value=f"{similarity_score:.2f}%", delta="Higher means more plagiarism")
            if similarity_score > 50:
                st.error("⚠️ **High plagiarism detected!** Consider rewriting the content.")
            elif 20 <= similarity_score <= 50:
                st.warning("⚠️ **Moderate plagiarism detected.** Some text may be similar.")
            else:
                st.success("✅ **No significant plagiarism detected!**")

def generate_pdf(report_text):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, report_text)
    temp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    pdf.output(temp_pdf.name)
    return temp_pdf.name

if st.button("🌐 Print Report as PDF", use_container_width=True):
    if not st.session_state["plagiarism_report"]:
        st.warning("⚠️ No report available. Check plagiarism first.")
    else:
        pdf_path = generate_pdf(st.session_state["plagiarism_report"])
        with open(pdf_path, "rb") as f:
            pdf_data = f.read()
        b64_pdf = base64.b64encode(pdf_data).decode()
        pdf_link = f'<a href="data:application/octet-stream;base64,{b64_pdf}" download="plagiarism_report.pdf">📅 Download PDF</a>'
        st.markdown(pdf_link, unsafe_allow_html=True)

hide_st_style = """
<style>
#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
header {visibility:hidden;}
.stTextArea textarea {font-size: 18px; padding: 12px;}
.stMetric {text-align: center;}
.stButton>button {border-radius: 8px; font-size: 16px; background-color: #4CAF50; color: white;}
</style>
"""
st.markdown(hide_st_style, unsafe_allow_html=True)
