from pypdf import PdfReader
from docx import Document
from PIL import Image
import pytesseract
import os
import re
import google.generativeai as genai
from dotenv import load_dotenv
import json

load_dotenv()

print("Libraries imported successfully.")

api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise ValueError("Google API Key not found. Please set it as a Colab secret named 'GOOGLE_API_KEY' or directly in the code.")


genai.configure(api_key=api_key)

# Initialize the model
model = genai.GenerativeModel("gemma-3-4b-it")

print("Gemini configured and model initialized.")

def extract_text_from_pdf(filepath):
    """Extracts text from a PDF file."""
    text = ""
    try:
        reader = PdfReader(filepath)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    except Exception as e:
        print(f"Error extracting text from PDF {filepath}: {e}")
        return ""
    return text

def extract_text_from_docx(filepath):
    """Extracts text from a DOCX file."""
    text = ""
    try:
        document = Document(filepath)
        for paragraph in document.paragraphs:
            text += paragraph.text + "\n"
    except Exception as e:
        print(f"Error extracting text from DOCX {filepath}: {e}")
        return ""
    return text

def extract_text_from_image(filepath):
    """Extracts text from an image file using OCR."""
    try:
        img = Image.open(filepath)
        text = pytesseract.image_to_string(img)
    except Exception as e:
        print(f"Error extracting text from image {filepath}: {e}")
        return ""
    return text

def clean_text(text):
    """Performs basic cleaning on extracted text."""
    # Remove extra whitespaces and newlines
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_text(filepath):
    """Extracts and cleans text from various file formats (PDF, DOCX, image)."""
    if not os.path.exists(filepath):
        print(f"Error: File not found at {filepath}")
        return ""

    file_extension = os.path.splitext(filepath)[1].lower()
    extracted_text = ""

    if file_extension == '.pdf':
        extracted_text = extract_text_from_pdf(filepath)
    elif file_extension == '.docx':
        extracted_text = extract_text_from_docx(filepath)
    elif file_extension in ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff']:
        extracted_text = extract_text_from_image(filepath)
    elif file_extension == '.txt':
        extracted_text = open(filepath, 'r', encoding='utf-8').read()
    else:
        print(f"Unsupported file type: {file_extension}")
        return ""

    return clean_text(extracted_text)

print("Text extraction module functions defined.")

resume_json_schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string", "description": "Full name of the applicant"},
        "email": {"type": "string", "format": "email", "description": "Contact email address"},
        "phone": {"type": "string", "description": "Contact phone number"},
        "linkedin": {"type": "string", "format": "uri", "description": "LinkedIn profile URL"},
        "experience": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Job title"},
                    "company": {"type": "string", "description": "Company name"},
                    "duration": {"type": "string", "description": "Employment duration (e.g., 'Jan 2020 - Dec 2022')"},
                    "description": {"type": "string", "description": "Key responsibilities and achievements"}
                },
                "required": ["title", "company"]
            },
            "description": "List of work experiences"
        },
        "education": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "degree": {"type": "string", "description": "Degree obtained"},
                    "institution": {"type": "string", "description": "Educational institution"},
                    "year": {"type": "string", "description": "Graduation year or period"}
                },
                "required": ["degree", "institution"]
            },
            "description": "List of educational qualifications"
        },
        "skills": {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of relevant skills"
        }
    },
    "required": ["name", "email"]
}

print("Resume JSON schema defined.")

def parse_resume_with_llm(resume_text, model):
    """
    Parses resume text using an LLM to extract structured information
    based on a predefined JSON schema.
    """

    # Convert the JSON schema to a string for embedding in the prompt
    schema_str = json.dumps(resume_json_schema, indent=2)

    # Construct the prompt for the LLM
    prompt = f"""
    You are an expert resume parsing assistant. Your task is to extract structured information from the provided resume text.
    Strictly adhere to the following JSON schema for your output.

    JSON Schema:
    {schema_str}

    Resume Text:
    {resume_text}

    Please output ONLY the JSON object, do not include any other text or formatting outside the JSON.
    Ensure the output JSON is well-formed and valid according to the schema.
    """

    try:
        # Call the LLM with the constructed prompt
        response = model.generate_content(prompt)

        # Extract the text content from the response
        json_output_str = response.text

        # Use regex to find and extract the JSON object, robustly handling markdown code blocks
        match = re.search(r'```json\n(.*)```', json_output_str, re.DOTALL)
        if match:
            json_output_str = match.group(1).strip()
        else:
            # If no ```json block, try to find a generic ``` block
            match = re.search(r'```\n(.*)```', json_output_str, re.DOTALL)
            if match:
                json_output_str = match.group(1).strip()
            else:
                # Fallback: if no code block, assume the whole string is JSON (after stripping whitespace)
                json_output_str = json_output_str.strip()

        # Attempt to parse the JSON string into a Python dictionary
        parsed_data = json.loads(json_output_str)
        return parsed_data
    except ValueError as ve:
        print(f"Error parsing LLM output as JSON: {ve}")
        print(f"LLM Raw Output: {json_output_str}")
        return None
    except Exception as e:
        print(f"An error occurred during LLM processing: {e}")
        return None

print("parse_resume_with_llm function defined.")


def process(resume_filepath):


  extracted_resume_text = extract_text(resume_filepath)

  if extracted_resume_text:
      print("Text extracted successfully (first 500 characters):\n", extracted_resume_text[:500])
  else:
      print("Failed to extract text from the resume.")

  if extracted_resume_text:
      # Parse the extracted text using the LLM
      parsed_resume_data = parse_resume_with_llm(extracted_resume_text, model)

      if parsed_resume_data:
          print("\nParsed Resume Data:\n", json.dumps(parsed_resume_data, indent=2))


      else:
          print("Failed to parse resume data with LLM.")
  else:
      print("Cannot parse resume data: no text was extracted.")

  output_filename = 'processed_resumes/structured_resume_data.json' #change this to save as {user_id}.json
  final_json_output = json.dumps(parsed_resume_data, indent=4)
  print(final_json_output)
  with open(output_filename, 'w') as f:
      f.write(final_json_output)

  print(f"Successfully saved structured job data to '{output_filename}'")

process('uploads/input1.pdf')