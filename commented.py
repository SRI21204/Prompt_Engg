import streamlit as st  # Import Streamlit library for building web apps
import google.generativeai as genai  # Import Google Generative AI library for AI-based text analysis
import json  # Import JSON library for handling JSON data
import requests  # Import requests library for making HTTP requests
from sklearn.feature_extraction.text import TfidfVectorizer  # Import TfidfVectorizer for text vectorization
from sklearn.metrics.pairwise import cosine_similarity  # Import cosine_similarity for similarity computation
from fpdf import FPDF  # Import FPDF for generating PDF files
import tempfile  # Import tempfile for creating temporary files
import base64  # Import base64 for encoding PDF data

# Set Streamlit page configuration (title, icon, layout)
st.set_page_config(page_title="Plagiarism Detector", page_icon="📝", layout="centered")

# Create a sidebar for API key inputs
with st.sidebar:
    st.title("🔑 API Keys Required")  # Sidebar title
    gemini_api_key = st.text_input("Enter Gemini API Key", type="password")  # Input field for Gemini API key (hidden text)
    st.markdown("[Get Gemini API Key](https://makersuite.google.com/app/apikey)")  # Link to get Gemini API key
    google_api_key = st.text_input("Enter Google Search API Key", type="password")  # Input field for Google Search API key
    google_cse_id = st.text_input("Enter Google CSE ID", type="password")  # Input field for Google Custom Search Engine ID
    st.markdown("[Get Google API Key & CSE ID](https://developers.google.com/custom-search/v1/introduction)")  # Link to get Google API credentials

# Configure Gemini AI if API key is provided
if gemini_api_key:
    genai.configure(api_key=gemini_api_key)  # Set up Gemini API with the provided key
    model = genai.GenerativeModel('gemini-2.0-flash')  # Initialize the Gemini model
else:
    st.warning("⚠️ Please enter your Gemini API key to proceed.")  # Warn user if no Gemini API key is entered

# Warn user if Google Search API credentials are missing
if not (google_api_key and google_cse_id):
    st.warning("⚠️ Please enter your Google Search API Key and CSE ID for full internet search.")

# Display main title and description
st.markdown("<h1 style='text-align: center;'>📝 AI-Powered Plagiarism Detector</h1>", unsafe_allow_html=True)  # Centered title in HTML
st.write("🔍 **Detect plagiarism** across the internet using AI and NLP with instant similarity scoring and source detection.")  # App description

# Initialize session state variables if not already present
if "text_input" not in st.session_state:
    st.session_state["text_input"] = ""  # Store user-entered text
if "plagiarism_report" not in st.session_state:
    st.session_state["plagiarism_report"] = ""  # Store plagiarism report

# Function to search the web using Google Custom Search API
def search_web(query, api_key, cse_id):
    try:
        url = "https://www.googleapis.com/customsearch/v1"  # Google Custom Search API endpoint
        params = {"key": api_key, "cx": cse_id, "q": query[:200]}  # Parameters: API key, CSE ID, and query (limited to 200 chars)
        response = requests.get(url, params=params, timeout=5)  # Send GET request with 5-second timeout
        if response.status_code == 200:  # Check if request was successful
            return response.json().get("items", [])  # Return search results (list of items)
        return []  # Return empty list if request fails
    except Exception as e:
        return [{"link": f"Error in search: {str(e)}", "snippet": ""}]  # Return error message if exception occurs

# Function to check plagiarism using AI and web search
def check_plagiarism(input_text, gemini_api_key, google_api_key, google_cse_id):
    # Define prompt for Gemini AI to analyze text and return JSON
    prompt = (
        "Analyze the following text for plagiarism and return a structured JSON response.\n"
        "Find real matching sources and return a JSON object with 'plagiarism_score' and 'sources'.\n"
        "ONLY return real sources. If no sources are found, return an empty list.\n\n"
        "Ensure response is in **valid JSON format** inside ```json ... ``` markdown blocks.\n\n"
        f"Text:\n{input_text}\n\n"
        "Expected JSON format:\n"
        "```json\n"
        "{\n"
        '  "plagiarism_score": "XX%",\n'
        '  "sources": ["https://example.com", "https://validsource.com"]\n'
        "}\n"
        "```"
    )

    ai_json = {"plagiarism_score": "0%", "sources": []}  # Default AI response if no analysis occurs
    if gemini_api_key:  # If Gemini API key is provided
        try:
            response = model.generate_content(prompt)  # Send prompt to Gemini model
            if response and hasattr(response, 'text'):  # Check if response is valid
                response_text = response.text.strip()  # Get response text and remove whitespace
                if "```json" in response_text:  # Check if response contains JSON block
                    response_text = response_text.split("```json")[1].split("```")[0].strip()  # Extract JSON content
                ai_json = json.loads(response_text)  # Parse JSON response
        except Exception as e:
            ai_json["sources"] = [f"Gemini Error: {str(e)}"]  # Store error if Gemini fails

    web_sources = []  # List to store web search results
    if google_api_key and google_cse_id:  # If Google Search credentials are provided
        web_results = search_web(input_text, google_api_key, google_cse_id)  # Perform web search
        for result in web_results[:5]:  # Limit to top 5 results
            snippet = result.get("snippet", "")  # Get snippet from search result
            similarity = compute_similarity(input_text, snippet)  # Compute similarity score
            if similarity > 20:  # If similarity exceeds 20%
                web_sources.append(result["link"])  # Add source URL to list

    # Extract AI plagiarism score and convert to float
    ai_score = float(ai_json.get("plagiarism_score", "0%").replace("%", ""))
    # Compute max web similarity score (if Google credentials provided)
    web_score = max([0] + [compute_similarity(input_text, r.get("snippet", "")) for r in web_results[:5] if google_api_key and google_cse_id])
    combined_score = max(ai_score, web_score)  # Take the higher score between AI and web
    combined_sources = list(set(ai_json.get("sources", []) + web_sources))  # Combine and deduplicate sources

    # Return combined results
    return {
        "plagiarism_score": f"{combined_score:.1f}",  # Format score to one decimal place
        "sources": combined_sources if combined_sources else ["⚠️ No valid sources found"]  # Sources or fallback message
    }

# Function to verify if a URL exists (returns True if status code is 200)
def verify_source_exists(url):
    try:
        response = requests.head(url, allow_redirects=True, timeout=5)  # Send HEAD request to URL
        return response.status_code == 200  # Return True if URL is accessible
    except requests.RequestException:
        return False  # Return False if request fails

# Function to compute cosine similarity between two texts
def compute_similarity(original, generated):
    if not original.strip() or not generated.strip():  # Check if either text is empty
        return 0.0  # Return 0 if no comparison possible
    vectorizer = TfidfVectorizer()  # Initialize TF-IDF vectorizer
    tfidf_matrix = vectorizer.fit_transform([original, generated])  # Convert texts to TF-IDF matrix
    similarity = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])  # Compute cosine similarity
    return similarity[0][0] * 100  # Return similarity as percentage

# Text area for user to input text to check
text_to_check = st.text_area("\U0001F4CC **Paste Your Text Below:**", value=st.session_state["text_input"], height=200, 
                            placeholder="Enter your text here...", key="text_input")

# Create three columns for buttons
col1, col2, col3 = st.columns(3)

# Stop button (currently just displays a warning)
if col3.button("🛑 Stop", use_container_width=True):
    st.warning("Process stopped!")

# Clear text button (resets session state and reruns app)
if col2.button("❌ Clear Text", use_container_width=True):
    del st.session_state["text_input"]  # Delete text input from session state
    del st.session_state["plagiarism_report"]  # Delete report from session state
    st.rerun()  # Rerun the app to refresh

# Check plagiarism button
if col1.button("🔎 Check Plagiarism", use_container_width=True):
    if not (gemini_api_key or (google_api_key and google_cse_id)):  # Check if at least one API key is provided
        st.warning("⚠️ Please enter at least one API key to proceed.")
    elif not text_to_check.strip():  # Check if text input is empty
        st.warning("⚠️ Please enter valid text to check for plagiarism.")
    else:
        with st.spinner("🔍 Analyzing text for plagiarism across the internet..."):  # Show spinner during analysis
            result = check_plagiarism(text_to_check, gemini_api_key, google_api_key, google_cse_id)  # Run plagiarism check

            # Generate report text
            report = (
                f"Plagiarism Score: {result['plagiarism_score']}%\n"
                f"Matching Sources:\n" + "\n".join([f"- {src}" for src in result['sources']])
            )
            st.session_state["plagiarism_report"] = report  # Store report in session state
            
            st.success("✅ **Analysis Complete!**")  # Display success message
            st.subheader("🔍 Plagiarism Report")  # Report subheader
            st.metric(label="🚗 Plagiarism Score", value=f"{result['plagiarism_score']}%", delta="Higher means more plagiarism")  # Display score
            st.write("📌 **Matching Sources:**")  # Sources label
            for source in result["sources"]:  # List all sources as clickable links
                st.markdown(f"🔗 [{source}]({source})")
            
            score = float(result["plagiarism_score"])  # Convert score to float for comparison
            if score > 50:  # High plagiarism warning
                st.error("⚠️ **High plagiarism detected!** Consider rewriting the content.")
            elif 20 <= score <= 50:  # Moderate plagiarism warning
                st.warning("⚠️ **Moderate plagiarism detected.** Some text may be similar.")
            else:  # Low or no plagiarism
                st.success("✅ **No significant plagiarism detected!**")

# Function to generate a PDF from report text
def generate_pdf(report_text):
    pdf = FPDF()  # Initialize FPDF object
    pdf.add_page()  # Add a page to the PDF
    pdf.set_font("Arial", size=12)  # Set font to Arial, size 12
    pdf.multi_cell(0, 10, report_text)  # Add report text to PDF
    temp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")  # Create a temporary PDF file
    pdf.output(temp_pdf.name)  # Save PDF to temporary file
    return temp_pdf.name  # Return file path

# Button to generate and download PDF report
if st.button("🌐 Print Report as PDF", use_container_width=True):
    if not st.session_state["plagiarism_report"]:  # Check if report exists
        st.warning("⚠️ No report available. Check plagiarism first.")
    else:
        pdf_path = generate_pdf(st.session_state["plagiarism_report"])  # Generate PDF
        with open(pdf_path, "rb") as f:  # Open PDF file in binary mode
            pdf_data = f.read()  # Read PDF content
        b64_pdf = base64.b64encode(pdf_data).decode()  # Encode PDF as base64
        pdf_link = f'<a href="data:application/octet-stream;base64,{b64_pdf}" download="plagiarism_report.pdf">📅 Download PDF</a>'  # Create download link
        st.markdown(pdf_link, unsafe_allow_html=True)  # Display download link

# CSS to customize Streamlit appearance
hide_st_style = """
<style>
#MainMenu {visibility:hidden;}  # Hide Streamlit main menu
footer {visibility:hidden;}  # Hide Streamlit footer
header {visibility:hidden;}  # Hide Streamlit header
.stTextArea textarea {font-size: 18px; padding: 12px;}  # Style text area
.stMetric {text-align: center;}  # Center-align metrics
.stButton>button {border-radius: 8px; font-size: 16px; background-color: #4CAF50; color: white;}  # Style buttons
</style>
"""
st.markdown(hide_st_style, unsafe_allow_html=True)  # Apply custom CSS