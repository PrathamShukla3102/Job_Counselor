import os
import json
import re
import PyPDF2 as pdf
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()  # Make sure GOOGLE_API_KEY3 is in your .env file

# Configure Gemini with your API key
genai.configure(api_key=os.getenv("GOOGLE_API_KEY3"))


def _extract_json_block(text: str) -> str:
    """
    Try to extract a JSON object from the raw model response.
    Handles cases like:
    ```json
    { ... }
    ```
    or text before/after the JSON block.
    """
    if not text:
        return ""

    # Remove code fences if present
    text = text.strip()
    text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"```$", "", text).strip()

    # Try to find the first {...} block
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start:end + 1]

    # Fallback: return original text
    return text


def get_gemini_response(job_description: str, resume_text: str) -> str:
    """
    Call Google's Gemini model (models/gemini-2.5-flash) and get structured JSON feedback.
    Returns a JSON STRING so that downstream code (process_gemini_response) can do .strip() and json.loads().
    """
    model = genai.GenerativeModel("models/gemini-2.5-flash")

    input_prompt = f"""
You are a job mentor and a student comes to you. He wants to become {job_description} 
and he has given you the following resume data:

RESUME:
{resume_text}

Your task:
Evaluate how well this resume matches the job description and suggest improvements.

You MUST respond with ONLY a single valid JSON object.
No explanations, no prose, no markdown, no code fences. Just pure JSON.

The JSON object MUST have exactly these keys:
- "percentage_match": a number between 0 and 100
- "missing_keywords": a non-empty array of strings (missing skills/keywords)
- "suggestions": a non-empty array of strings (resume improvement tips)
- "candidate_name": a non-empty string (name of the candidate; if unknown, use "Candidate")

Rules:
- Do NOT return empty arrays. If everything looks good, still include at least one keyword and one suggestion.
- Do NOT use null or empty string for any value.
- Do NOT include any other keys.
- Return ONLY the JSON object, for example:
{"{"}"percentage_match": 78, "missing_keywords": ["Python", "Docker"], "suggestions": ["Add projects that show real-world ML use cases."], "candidate_name": "Pratham Shukla"{"}"}
"""

    response = model.generate_content(input_prompt)
    raw_text = response.text or ""

  
    # Try to clean and extract a JSON object from the response
    cleaned_text = _extract_json_block(raw_text)

    try:
        data = json.loads(cleaned_text)
    except json.JSONDecodeError:
        # Fallback in case model returns something slightly off
        data = {
            "percentage_match": 0,
            "missing_keywords": ["JSON_PARSE_ERROR"],
            "suggestions": ["Model did not return valid JSON. Please try again."],
            "candidate_name": "Candidate",
        }

    # Extra safety: ensure required keys exist and are non-empty
    if "percentage_match" not in data:
        data["percentage_match"] = 0
    if "missing_keywords" not in data or not data["missing_keywords"]:
        data["missing_keywords"] = ["No missing keywords detected, but resume can be improved."]
    if "suggestions" not in data or not data["suggestions"]:
        data["suggestions"] = ["No suggestions generated. Consider adding measurable achievements."]
    if "candidate_name" not in data or not str(data["candidate_name"]).strip():
        data["candidate_name"] = "Candidate"

    # RETURN JSON STRING so process_gemini_response can .strip() and json.loads()
    return json.dumps(data)


def input_pdf_text(uploaded_file) -> str:
    """
    Extract text from an uploaded PDF file.
    """
    reader = pdf.PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text
