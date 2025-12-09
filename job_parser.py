import google.generativeai as genai
import os
from dotenv import load_dotenv
import pandas as pd
import time
import json
import re

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")



if not api_key:
    raise ValueError("Google API Key not found. Please set it as a Colab secret named 'GOOGLE_API_KEY' or directly in the code.")

# Configure Gemini
genai.configure(api_key=api_key)

# Initialize the model
model = genai.GenerativeModel("gemma-3-4b-it")

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

# Define the path to your CSV file on Google Drive
# IMPORTANT: Update this path to your actual CSV file location
def process(file_path,limit):
    print("it has begun")
    print(file_path)
    job_descriptions = []

    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found at '{file_path}'.")

        ext = os.path.splitext(file_path)[1].lower()

        # ✅ CASE 1: CSV FILE
        if ext == ".csv":
            df = pd.read_csv(file_path)

            if "description" not in df.columns:
                raise ValueError("CSV file must contain a 'description' column.")

            job_descriptions = df["description"].head(int(limit)).tolist()

        # ✅ CASE 2: JSON FILE
        elif ext == ".json":
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # ✅ Accepts both list of dicts or dict with key
            if isinstance(data, list):
                job_descriptions = [
                    item["description"]
                    for item in data
                    if isinstance(item, dict) and "description" in item
                ]
            elif isinstance(data, dict) and "description" in data:
                job_descriptions = data["description"]
                if not isinstance(job_descriptions, list):
                    raise ValueError("'description' in JSON must be a list.")
            else:
                raise ValueError("JSON must contain a list or a 'description' key.")

            job_descriptions = job_descriptions[:limit]

        else:
            raise ValueError("Unsupported file format. Only .csv and .json are allowed.")

        print(f"✅ Successfully loaded {len(job_descriptions)} job descriptions from '{file_path}'.")

        if job_descriptions:
            print("\n✅ First loaded job description (first 200 chars):")
            print(job_descriptions[0][:200] + "...")
        else:
            print("⚠️ No job descriptions found.")

    except FileNotFoundError as e:
        print(f"❌ {e}")

    except json.JSONDecodeError:
        print("❌ Invalid JSON file format.")

    except ValueError as e:
        print(f"❌ Data Error: {e}")

    except Exception as e:
        print(f"❌ Unexpected error: {e}")


    structured_job_data = []
    count = 1
    for job_description in job_descriptions:
        print('processing job description no.', count)
        # Format the prompt with the current job description and predefined categories
        formatted_prompt = new_prompt_template_final.format(
            job_description=job_description
        )

        try:
            # Generate content using the Gemini model
            response = model.generate_content(formatted_prompt)
            generated_text = response.text

            match = re.search(r"```json\s*(.*?)\s*```", generated_text, re.DOTALL)

            if match:
                json_str = match.group(1).strip()
            else:
                # ✅ Fallback: extract first {...} anywhere in text
                match = re.search(r"\{.*\}", generated_text, re.DOTALL)
                if not match:
                    raise ValueError("No JSON object found in response.")
                json_str = match.group(0).strip()

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
        time.sleep(5)
        count += 1

    print("Processing complete. First entry of structured_job_data:")
    if structured_job_data:
        print(json.dumps(structured_job_data[0], indent=4))
    else:
        print("No data processed.")

    # Convert the structured_job_data list into a single JSON formatted string
    final_json_output = json.dumps(structured_job_data, indent=4)

    # Print the final_json_output to display the complete structured data
    print(final_json_output)
    output_filename = 'processed_jobs/structured_job_data.json' #change this to save as {internship_id}.json

    with open(output_filename, 'w') as f:
        f.write(final_json_output)

    print(f"Successfully saved structured job data to '{output_filename}'")