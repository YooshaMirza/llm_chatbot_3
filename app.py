import streamlit as st
from PyPDF2 import PdfReader
from transformers import BartTokenizer, BartForConditionalGeneration
import tempfile
import os

# Load BART model for Summarization
tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")

# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Function to summarize the extracted text using BART
def summarize_text(text):
    try:
        inputs = tokenizer([text], max_length=1024, return_tensors="pt", truncation=True)
        summary_ids = model.generate(inputs["input_ids"], max_length=150, min_length=30, length_penalty=2.0, num_beams=4, early_stopping=True)
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary
    except Exception as e:
        return f"Error summarizing text: {e}"

# Streamlit app
def main():
    st.set_page_config(page_title="Health Report Chatbot", page_icon="💬", layout="wide")
    st.title("Health Report Chatbot 💬")
    st.write("Upload your health report in PDF format, and the chatbot will summarize it and interact with you based on that information.")

    pdf_file = st.file_uploader("Upload PDF", type="pdf")

    if pdf_file:
        # Save uploaded file to a temporary location to avoid memory overload
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(pdf_file.read())
            tmp_file_path = tmp_file.name

        st.write("Extracting text from the PDF...")
        extracted_text = extract_text_from_pdf(tmp_file_path)
        os.remove(tmp_file_path)  # Clean up the temp file

        st.write("Summarizing the health report...")
        summary = summarize_text(extracted_text)
        st.write("Summary:")
        st.write(summary)

        st.write("You can now chat about the summary.")
        user_input = st.text_input("Type your message here:")
        if user_input:
            # Simple echo response as an example
            st.write("Chatbot Response:")
            st.write(f"You asked: {user_input}\nBased on the summary: {summary}")

if __name__ == '__main__':
    main()
