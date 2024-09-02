import streamlit as st
import pandas as pd
import fitz  # PyMuPDF for PDF processing
from langchain_community.llms import Together
from sentence_transformers import SentenceTransformer, util
from io import StringIO, BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import xlsxwriter
import csv

# API key and model for Together AI
api_key = "replace_this_with_your_api_key"
model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"  # Update this with a valid model name from Together
llm = Together(together_api_key=api_key, model=model_name)

# Initialize the sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize session state variables
if 'quiz_index' not in st.session_state:
    st.session_state.quiz_index = 0a
if 'user_answers' not in st.session_state:
    st.session_state.user_answers = {}
if 'quiz_started' not in st.session_state:
    st.session_state.quiz_started = False
if 'qa_list' not in st.session_state:
    st.session_state.qa_list = []
if 'feedback_active' not in st.session_state:
    st.session_state.feedback_active = False
if 'feedback_query' not in st.session_state:
    st.session_state.feedback_query = None

# Streamlit app title with an eye-catching header
st.markdown("<h1 style='text-align: center;'>Automated Question Builder (AQB)</h1>", unsafe_allow_html=True)

# File uploader
uploaded_file = st.file_uploader("Upload a CSV, XLSX, or PDF file", type=["csv", "xlsx", "pdf"])

# Text extraction from PDF using PyMuPDF
def extract_text_from_pdf(file):
    try:
        doc = fitz.open(stream=file.read(), filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        return ""

# Function to filter content based on the topic using embedding-based similarity
def filter_content_by_topic_embedding(content, topic, threshold=0.5):
    lines = content.split('\n')
    topic_embedding = model.encode(topic, convert_to_tensor=True)
    line_embeddings = model.encode(lines, convert_to_tensor=True)
    
    # Compute cosine similarities
    cosine_scores = util.pytorch_cos_sim(topic_embedding, line_embeddings)[0]
    
    # Filter lines based on similarity threshold
    relevant_lines = [line for line, score in zip(lines, cosine_scores) if score >= threshold]
    
    return "\n".join(relevant_lines) if relevant_lines else content

# Function to generate questions based on the content and topic
def generate_questions(text, topic, num_questions, question_type, difficulty, include_answers=True):
    # Filter content based on the topic using the new embedding-based method
    filtered_text = filter_content_by_topic_embedding(text, topic)
    if not filtered_text.strip():
        st.warning("Filtered content is empty. No relevant content found for the given topic.")
        return []
    
    # Initialize an empty list for storing question-answer pairs
    qa_list = []
    
    # Loop until we get the exact number of questions needed
    while len(qa_list) < num_questions:
        # Calculate the number of questions still needed
        remaining_questions = num_questions - len(qa_list)
        
        # Structured prompt for generating questions
        if question_type == 'Objective':
            prompt = (f"Based on the following content related to the topic '{topic}', "
                      f"generate exactly {remaining_questions} diverse and meaningful objective questions "
                      f"such as True/False, MCQs, Fill in the Blanks, etc., with options and answers if applicable. "
                      f"Format the output as 'Q: <question>' and 'A: <answer>' for each pair if answers are included.\n\n"
                      f"Content:\n{filtered_text}")
        else:  # Subjective
            prompt = (f"Based on the following content related to the topic '{topic}', "
                      f"generate exactly {remaining_questions} diverse and meaningful subjective questions "
                      f"with their answers. Format the output as 'Q: <question>' and 'A: <answer>' for each pair.\n\n"
                      f"Content:\n{filtered_text}")

        try:
            response = llm(prompt)
            
            # Parse questions and answers
            question = None
            answer = None
            qa_lines = response.split('\n')

            for line in qa_lines:
                line = line.strip()
                if line.startswith('Q:'):
                    if question:  # Add previous question with or without answer
                        qa_list.append([question, answer if include_answers else None])
                    question = line.replace('Q:', '').strip()
                    answer = None  # Reset the answer
                elif line.startswith('A:') and include_answers:
                    answer = line.replace('A:', '').strip()

            # Add the last pair
            if question:
                qa_list.append([question, answer if include_answers else None])

        except Exception as e:
            st.error(f"Error generating questions and answers: {e}")
            return []
    
    # Ensure we return exactly the number of questions requested
    return qa_list[:num_questions]

# Create quiz interface
def create_quiz_interface(qa_list):
    if not st.session_state.quiz_started:
        st.session_state.quiz_started = True
        st.session_state.quiz_index = 0
        st.session_state.user_answers = {}
    
    if st.session_state.quiz_started:
        if st.session_state.quiz_index < len(qa_list):
            question, _ = qa_list[st.session_state.quiz_index]
            st.write(f"**Question {st.session_state.quiz_index + 1}:** {question}")

            answer = st.text_input("Your answer:", key=f"answer_{st.session_state.quiz_index}")

            col1, col2 = st.columns(2)
            with col1:
                if st.button("Next", key=f"next_{st.session_state.quiz_index}"):
                    st.session_state.user_answers[st.session_state.quiz_index] = answer
                    st.session_state.quiz_index += 1
            with col2:
                if st.button("Submit Quiz", key="submit_quiz"):
                    st.session_state.user_answers[st.session_state.quiz_index] = answer
                    st.session_state.quiz_started = False
                    st.success("Quiz Completed!")
                    evaluate_quiz(st.session_state.qa_list, st.session_state.user_answers)
                    
                # Quit Quiz Button
                if st.button("Quit Quiz", key="quit_quiz"):
                    st.session_state.quiz_started = False
                    st.session_state.quiz_index = 0
                    st.session_state.user_answers = {}
                    st.session_state.qa_list = []
                    st.session_state.feedback_active = True
                    st.session_state.feedback_query = None
                    st.write("Quiz has been quit. Please provide feedback to regenerate questions based on your input.")
                    st.text_area("Feedback (e.g., what kind of questions would you like to see?):", key="feedback_input")
                    if st.button("Submit Feedback", key="submit_feedback"):
                        feedback = st.session_state.feedback_query
                        regenerate_questions_based_on_feedback(feedback)
                    st.success("Feedback submitted. Generating new questions based on feedback.")
        else:
            st.write("No more questions. Click **Submit Quiz** to finish.")
            if st.button("Submit Quiz", key="submit_quiz_end"):
                st.session_state.user_answers[st.session_state.quiz_index] = answer
                st.session_state.quiz_started = False
                st.success("Quiz Completed!")
                evaluate_quiz(st.session_state.qa_list, st.session_state.user_answers)
    else:
        st.write("Quiz has ended.")

def regenerate_questions_based_on_feedback(feedback):
    if feedback:
        st.session_state.qa_list = generate_questions(text=uploaded_file, topic=feedback, num_questions=num_questions, question_type=question_type, difficulty=difficulty, include_answers=include_answers)
        st.write("New questions generated based on your feedback.")
    else:
        st.warning("No feedback provided. Cannot regenerate questions.")

def evaluate_quiz(qa_list, user_answers):
    st.markdown("### **Quiz Evaluation**")
    for i, (question, correct_answer) in enumerate(qa_list):
        user_answer = user_answers.get(i, "").strip()
        if correct_answer:  # Ensure correct_answer is not None
            is_correct = "Yes" if (user_answer.lower() == correct_answer.lower()) else "No"
        else:
            is_correct = "No"
        st.write(f"**Q{i + 1}:** {question}")
        st.write(f"**Your Answer:** {user_answer}")
        st.write(f"**Correct Answer:** {correct_answer}")
        st.write(f"**Correct:** {is_correct}")
        st.markdown("---")

# Functions to generate files
def generate_csv(qa_list):
    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(["Question", "Answer"])
    for q, a in qa_list:
        writer.writerow([q, a])
    return output.getvalue()

def generate_excel(qa_list):
    output = BytesIO()
    workbook = xlsxwriter.Workbook(output, {'in_memory': True})
    worksheet = workbook.add_worksheet()
    worksheet.write_row(0, 0, ["Question", "Answer"])
    for row_num, (q, a) in enumerate(qa_list, start=1):
        worksheet.write_row(row_num, 0, [q, a])
    workbook.close()
    output.seek(0)
    return output.getvalue()

def generate_pdf(qa_list):
    output = BytesIO()
    c = canvas.Canvas(output, pagesize=letter)
    width, height = letter
    y = height - 40
    c.setFont("Helvetica", 12)
    for i, (q, a) in enumerate(qa_list):
        c.drawString(40, y, f"Q{i + 1}: {q}")
        y -= 20
        if a:
            c.drawString(40, y, f"A: {a}")
            y -= 20
        if y < 40:
            c.showPage()
            y = height - 40
    c.save()
    output.seek(0)
    return output.getvalue()

# Main app logic
if uploaded_file:
    if uploaded_file.type == "text/csv":
        content = pd.read_csv(uploaded_file).to_string()
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
        content = pd.read_excel(uploaded_file).to_string()
    elif uploaded_file.type == "application/pdf":
        content = extract_text_from_pdf(uploaded_file)

    # User input for topic
    topic = st.text_input("Enter the topic for question generation")

    # Options for generating questions
    num_questions = st.number_input("Number of questions to generate", min_value=1, value=5)
    question_type = st.selectbox("Select question type", ["Objective", "Subjective"])
    difficulty = st.selectbox("Select difficulty level", ["Easy", "Medium", "Hard"])

    include_answers = st.checkbox("Include Answers")

    if st.button("Generate Questions"):
        st.session_state.qa_list = generate_questions(content, topic, num_questions, question_type, difficulty, include_answers)
    
    # Display the generated questions
    if st.session_state.qa_list:
        include_answers = st.checkbox("Include Answers", value=True)
        qa_data = [[q[0], q[1] if include_answers else None] for q in st.session_state.qa_list]
        df = pd.DataFrame(qa_data, columns=["Question", "Answer"])
        
        st.write("**Generated Questions:**")
        st.table(df)
        
        # Prompt for download format
        download_format = st.selectbox("Select download format", ["None", "CSV", "Excel", "PDF"])
        
        if st.button("Download"):
            if download_format == "CSV":
                csv_data = generate_csv(st.session_state.qa_list)
                st.download_button("Download CSV", csv_data, "questions.csv", "text/csv")
            elif download_format == "Excel":
                excel_data = generate_excel(st.session_state.qa_list)
                st.download_button("Download Excel", excel_data, "questions.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            elif download_format == "PDF":
                pdf_data = generate_pdf(st.session_state.qa_list)
                st.download_button("Download PDF", pdf_data, "questions.pdf", "application/pdf")
            else:
                st.warning("Please select a download format.")

        # Start quiz if questions are available
        if include_answers:
            create_quiz_interface(st.session_state.qa_list)
    else:
        st.warning("Please upload a file and enter a topic.")

