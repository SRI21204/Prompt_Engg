import streamlit as st
import google.generativeai as genai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Streamlit Page Config
st.set_page_config(page_title="Plagiarism Detector", page_icon="📄", layout="centered")

# Sidebar for API Key
with st.sidebar:
    st.title("🔑 API Key Required")
    api_key = st.text_input("Enter API Key", type="password")
    st.markdown("[Get Your API Key](https://makersuite.google.com/app/apikey)")

# Configure API only if key is provided
if api_key:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-pro')
else:
    st.warning("⚠️ Please enter your API key to proceed.")

# Title Section
st.markdown("<h1 style='text-align: center;'>📝 AI-Powered Plagiarism Detector</h1>", unsafe_allow_html=True)
st.write("🔍 **Detect plagiarism** using AI and NLP techniques with instant similarity scoring.")

# User Input
text_to_check = st.text_area("📌 **Paste Your Text Below:**", height=200, placeholder="Enter your text here...")

# Function to check plagiarism using AI
def check_plagiarism(input_text):
    prompt = f"Check if the following text is plagiarized. If similar sources exist, return a rewritten version and the percentage of similarity:\n\n{input_text}"
    
    try:
        response = model.generate_content(prompt)
        
        # Extract AI response properly
        if response and hasattr(response, 'candidates'):
            return response.candidates[0]['text'].strip()
        elif hasattr(response, 'text'):
            return response.text.strip()
        else:
            return "No response generated."
    
    except Exception as e:
        return f"Error: {str(e)}"

# Function to compute text similarity (Basic NLP)
def compute_similarity(original, generated):
    if not original or not generated:
        return 0.0

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([original, generated])
    similarity = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])
    
    return similarity[0][0] * 100  # Convert to percentage

# Buttons Layout
col1, col2 = st.columns(2)

with col1:
    check_button = st.button("🔎 Check Plagiarism", use_container_width=True)

with col2:
    clear_button = st.button("❌ Clear Text", use_container_width=True)

# Clear Button Functionality
if clear_button:
    text_to_check = ""  # Reset text area

# Check Plagiarism Action
if check_button:
    if not api_key:
        st.warning("⚠️ Please enter your API key to proceed.")
    elif not text_to_check.strip():
        st.warning("⚠️ Please enter valid text to check for plagiarism.")
    else:
        with st.spinner("🔍 Analyzing text for plagiarism..."):
            ai_result = check_plagiarism(text_to_check)

            if isinstance(ai_result, str):
                similarity_score = compute_similarity(text_to_check, ai_result)
            else:
                similarity_score = 0

            # Results Section
            st.success("✅ **Analysis Complete!**")
            st.subheader("🔍 Plagiarism Report")
            st.write(f"**Rewritten Text:**\n\n{ai_result}")

            # Plagiarism Score Display
            st.metric(label="🛑 Plagiarism Score", value=f"{similarity_score:.2f}%", delta="Higher means more plagiarism")

            # Plagiarism Risk Levels
            if similarity_score > 50:
                st.error("⚠️ **High plagiarism detected!** Consider rewriting the content.")
            elif 20 <= similarity_score <= 50:
                st.warning("⚠️ **Moderate plagiarism detected.** Some text may be similar.")
            else:
                st.success("✅ **No significant plagiarism detected!**")

# Hide Streamlit Branding & Improve UI
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
