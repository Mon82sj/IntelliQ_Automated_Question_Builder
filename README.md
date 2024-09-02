# IntelliQ_Automated_Question_Builder

**IntelliQ**
Tagline: Empowering Knowledge Through Intelligent Automation

**Overview**

IntelliQ is a web application developed using Streamlit that enables users to generate and download questions based on uploaded documents. It supports CSV, XLSX, and PDF file formats and provides options for creating both objective and subjective questions with varying difficulty levels. IntelliQ integrates with the Together AI API for question generation and offers download options in CSV, Excel, or PDF formats.

**Features**

**File Upload:** Upload CSV, XLSX, or PDF files for content extraction.
**Topic-Based Question Generation:** Generate questions based on a specified topic from the uploaded content.
**Question Types:** Choose between objective and subjective question types.
**Difficulty Levels:** Select the difficulty level of the generated questions (Easy, Medium, Hard).
**Answer Inclusion:** Option to include answers with the generated questions.
**Download Options:** Download generated questions as CSV, Excel, or PDF files.
**Quiz Functionality:** Create and take quizzes based on generated questions.
**Feedback Integration:** Provide feedback to regenerate questions based on user input.

Installation

To run IntelliQ locally, follow these steps:

Clone the Repository

git clone https://github.com/yourusername/intelliq.git
cd intelliq

Create a Virtual Environment

python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

Install Dependencies

pip install -r requirements.txt

Run the Application

streamlit run app.py

Dependencies

streamlit: For creating the web interface.
pandas: For handling CSV and Excel files.
fitz (PyMuPDF): For PDF text extraction.
langchain_community: For integration with Together AI for question generation.
sentence_transformers: For embedding-based content filtering.
reportlab: For PDF generation.
xlsxwriter: For Excel file generation.
csv: For CSV file generation.

Configuration

API Key and Model Name

Update the api_key and model_name variables in the IntelliQ_Program.py file with your Together AI API key and the desired model name.

Environment Variables

For security reasons, you may want to use environment variables for storing sensitive information like API keys.

Usage

Upload File: Upload a CSV, XLSX, or PDF file containing the content.
Enter Topic: Input the topic for which questions need to be generated.
Select Parameters: Choose the number of questions, question type, difficulty level, and whether to include answers.
Generate Questions: Click the "Generate Questions" button to produce the questions.
Download Options: Select a download format (CSV, Excel, PDF) and click "Download" to get the questions in the desired format.
Quiz Interface: Start a quiz to answer the generated questions and get evaluated based on your responses.
Provide Feedback: If you quit the quiz, provide feedback to regenerate questions based on your input.

Example

import streamlit as st
import pandas as pd
from langchain_community.llms import Together
from sentence_transformers import SentenceTransformer
import fitz  # PyMuPDF for PDF processing

# Initialize models and API
api_key = "your_api_key_here"
model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
llm = Together(together_api_key=api_key, model=model_name)
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

# Streamlit app setup
st.title("IntelliQ - Empowering Knowledge Through Intelligent Automation")
uploaded_file = st.file_uploader("Upload a CSV, XLSX, or PDF file", type=["csv", "xlsx", "pdf"])

if uploaded_file:
    # Handle file processing and question generation here
    pass
Contributing
Contributions are welcome! Please fork the repository and submit a pull request with your proposed changes. Ensure that your changes include appropriate tests and documentation.

License
