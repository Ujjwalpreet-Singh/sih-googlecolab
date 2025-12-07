import google.generativeai as genai
import os
from google.colab import userdata
import pandas as pd
from google.colab import drive
import json


api_key = userdata.get("GOOGLE_API_KEY")



if not api_key:
    raise ValueError("Google API Key not found. Please set it as a Colab secret named 'GOOGLE_API_KEY' or directly in the code.")

# Configure Gemini
genai.configure(api_key=api_key)

# Initialize the model
model = genai.GenerativeModel("gemini-2.5-flash-lite")

print("Gemini configured and model initialized.")

new_prompt_template_final = """You are a highly analytical expert in talent acquisition and Natural Language Processing (NLP). Your primary goal is to process the provided job description, extract key information, and **strictly normalize the skills and responsibilities for optimal matching using BERT-style embedding models.**

**Normalization Strategy for Embeddings (CRITICAL):**
* **Strict Consolidation & Hierarchy:** **MUST** use the **broadest possible normalized term** to represent a concept. If a skill (e.g., 'Photoshop', 'Time Management', 'Drafting Wills') is a sub-component of a larger, encompassing concept ('Adobe Creative Cloud', 'Organizational Skills', 'Estate Planning'), you **MUST list ONLY the parent concept.**
* **Standardization:** Use concise, consistent phrasing. (e.g., use 'Communication' instead of 'Communication Skills').
* **Responsibility Action Focus:** Responsibilities must be phrased as clear, concise actions (e.g., "Manage client relationships" not "Client relationship management").
* **No Redundancy:** The final lists must be the shortest possible representation of the job's core requirements.

**For the Job Description, follow these steps:**
1.  **Extract Job Title**: Identify the main job title.
2.  **Extract & Normalize Skills**: **Strictly consolidate** and standardize all skills following the Normalization Strategy above to create a concise list of high-level, unique skill terms.
3.  **Extract & Normalize Responsibilities**: **Strictly consolidate** and standardize all responsibilities into a concise list of unique, action-focused terms.

Your output MUST be a JSON object with the following structure:
{{
    "job_title": "<Extracted Job Title>",
    "skills": ["<Normalized Skill 1>", "<Normalized Skill 2>", ...],
    "responsibilities": ["<Normalized Responsibility 1>", "<Normalized Responsibility 2>", ...]
}}

Job Description: {job_description}"""

print("New prompt template optimized for BERT embedding consistency has been created.")


# Mount Google Drive (if not already mounted)
drive.mount('/content/drive', force_remount=True)

# Define the path to your CSV file on Google Drive
# IMPORTANT: Update this path to your actual CSV file location
csv_file_path = '/content/drive/My Drive/SIH-Project/postings.csv'

try:
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(csv_file_path)

    # Ensure the 'description' column exists
    if 'description' not in df.columns:
        raise ValueError("CSV file must contain a 'description' column.")

    # Extract the first 10 job descriptions
    # You can adjust the number of postings or criteria as needed
    job_descriptions = df['description'].head(5).tolist()

    print(f"Successfully loaded {len(job_descriptions)} job descriptions from '{csv_file_path}'.")

    # Print the first job description to verify
    if job_descriptions:
        print("\nFirst loaded job description (first 200 chars):")
        print(job_descriptions[0][:200] + "...")
    else:
        print("No job descriptions loaded.")

except FileNotFoundError:
    print(f"Error: CSV file not found at '{csv_file_path}'. Please check the path.")
    job_descriptions = []
except ValueError as e:
    print(f"Error processing CSV: {e}")
    job_descriptions = []
except Exception as e:
    print(f"An unexpected error occurred: {e}")
    job_descriptions = []

structured_job_data = []
for job_description in job_descriptions:
    # Format the prompt with the current job description and predefined categories
    formatted_prompt = new_prompt_template_final.format(
        job_description=job_description
    )

    try:
        # Generate content using the Gemini model
        response = model.generate_content(formatted_prompt)
        generated_text = response.text

        # Attempt to parse the generated text as JSON
        # Gemini sometimes wraps JSON in markdown code blocks, so we need to extract it.
        if generated_text.startswith('```json') and generated_text.endswith('```'):
            json_str = generated_text[len('```json'):-len('```')].strip()
        else:
            json_str = generated_text.strip()

        parsed_data = json.loads(json_str)
        structured_job_data.append(parsed_data)
    except Exception as e:
        print(f"Error processing job description: {job_description[:50]}...")
        print(f"Error: {e}")
        print(f"Raw response text: {generated_text}")
        # Optionally, append original description with an error message
        structured_job_data.append({
            "original_description": job_description,
            "error": str(e),
            "raw_gemini_response": generated_text
        })

print("Processing complete. First entry of structured_job_data:")
if structured_job_data:
    print(json.dumps(structured_job_data[0], indent=4))
else:
    print("No data processed.")

# Convert the structured_job_data list into a single JSON formatted string
final_json_output = json.dumps(structured_job_data, indent=4)

# Print the final_json_output to display the complete structured data
print(final_json_output)
output_filename = 'structured_job_data.json'

with open(output_filename, 'w') as f:
    f.write(final_json_output)

print(f"Successfully saved structured job data to '{output_filename}'")