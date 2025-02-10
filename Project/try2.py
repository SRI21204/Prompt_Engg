import streamlit as st
import google.generativeai as genai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Sidebar for API Key
with st.sidebar:
    api_key = st.text_input("API Key", type="password")
    st.markdown("[Get Your API Key](https://makersuite.google.com/app/apikey)")

# Configure API only if key is provided
if api_key:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-pro')
else:
    st.info("Please add your API key to continue.")

# Title
st.title("📝 AI-Powered Plagiarism Detector")

# User Input
text_to_check = st.text_area("Paste your text here:")

# Function to check plagiarism using AI
def check_plagiarism(input_text):
    prompt = f"Check if the following text is plagiarized. If similar sources exist, return a rewritten version and the percentage of similarity:\n\n{input_text}"
    
    try:
        response = model.generate_content(prompt)
        
        # Extract AI response properly
        if response and hasattr(response, 'text'):
            return response.text.strip()  # Convert to string
        else:
            return "No response generated."
    
    except Exception as e:
        return f"Error: {str(e)}"

# Function to compute text similarity (Basic NLP)
def compute_similarity(original, generated):
    if not original or not generated:  # Avoid errors if input is empty
        return 0.0

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([original, generated])
    similarity = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])
    
    return similarity[0][0] * 100  # Convert to percentage

# Submit Button
if st.button("Check Plagiarism"):
    if not api_key:
        st.warning("Please enter your API key to proceed.")
    elif not text_to_check.strip():
        st.warning("Please enter valid text to check for plagiarism.")
    else:
        with st.spinner("Checking for plagiarism..."):
            ai_result = check_plagiarism(text_to_check)

            # Ensure AI response is a string
            if isinstance(ai_result, str):
                similarity_score = compute_similarity(text_to_check, ai_result)
            else:
                similarity_score = 0  # If AI response fails, assume no similarity

            # Display Results
            st.subheader("🔍 Plagiarism Report")
            st.write(ai_result)

            st.metric(label="🛑 Plagiarism Score", value=f"{similarity_score:.2f}%")
            
            # Plagiarism Risk Levels
            if similarity_score > 50:
                st.error("⚠️ High risk of plagiarism detected!")
            elif 20 <= similarity_score <= 50:
                st.warning("⚠️ Some content might be plagiarized.")
            else:
                st.success("✅ No significant plagiarism detected!")

# Hide Streamlit branding
hide_st_style = """
<style>
#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
header {visibility:hidden;}
</style>
"""
st.markdown(hide_st_style, unsafe_allow_html=True)
