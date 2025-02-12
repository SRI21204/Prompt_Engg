Here’s a list of all the installations you need to run the plagiarism detector code successfully. You can install them using pip in your Python environment.

1. Install Streamlit
Streamlit is required to create the web application interface.
$ pip install streamlit

3. Install Google's Generative AI SDK
This is needed to interact with Google’s Gemini API for AI-powered plagiarism detection.
$ pip install google-generativeai

3. Install scikit-learn
Used for Natural Language Processing (NLP) functions like TF-IDF vectorization and cosine similarity calculation.
$ pip install scikit-learn

4. Install fpdf
fpdf is used for generating the plagiarism report in PDF format.
$ pip install fpdf

Install All at Once
To install all dependencies in one command, run:
$ pip install streamlit google-generativeai scikit-learn fpdf

Running the Code
Once everything is installed, you can run the Streamlit app using:
$ streamlit run app.py
