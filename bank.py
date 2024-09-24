import pdfplumber
import pytesseract
import requests
import pandas as pd
from PIL import Image
# 1. Setup Google Gemini API
API_KEY = 'AIzaSyBzUi4AFJL-yDEGj7kLEOAc1Q_7bPZ72dA'
API_ENDPOINT = 'https://api.geminivisionpro.com/v1/process'
# Function to send requests to Google Gemini LLM
def query_gemini(text_prompt):
    headers = {'x-api-key': API_KEY}
    data = {'prompt': text_prompt}
    
    response = requests.post(API_ENDPOINT, json=data, headers=headers)
    
    if response.status_code == 200:
        return response.json().get('output', '')
    else:
        raise Exception(f"API error: {response.status_code}, {response.text}")
# 2. Extract Text from PDF using pdfplumber
def extract_text_from_pdf(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text
# 3. Extract Text from Image using pytesseract
def extract_text_from_image(image_path):
    image = Image.open(image_path)
    text = pytesseract.image_to_string(image)
    return text
# 4. Process Extracted Text and Query Google Gemini
def process_bank_statement(text):
    # Example query to extract transactions
    prompt = f"""
    Extract the transaction details (dates, descriptions, and amounts) from the following bank statement text:
    
    {text}
    """
    
    extracted_info = query_gemini(prompt)
    
    return extracted_info
# 5. Example Function to Display Transactions
def display_transactions(extracted_info):
    # Example of how you might display structured data
    print("Extracted Transaction Data:")
    print(extracted_info)
# 6. Main Function to Handle PDF or Image Input
def main(input_file, file_type='pdf'):
    if file_type == 'pdf':
        # Extract text from a PDF
        extracted_text = extract_text_from_pdf(input_file)
    elif file_type == 'image':
        # Extract text from an image
        extracted_text = extract_text_from_image(input_file)
    else:
        raise ValueError("Invalid file type. Use 'pdf' or 'image'.")
    # Process the extracted text using Google Gemini
    result = process_bank_statement(extracted_text)
    
    # Display or save the results
    display_transactions(result)
# Example Usage
if __name__ == "__main__":
    # Input file can be either a PDF or an image
    input_file = "bank_statement.pdf"
    main(input_file, file_type='pdf')
