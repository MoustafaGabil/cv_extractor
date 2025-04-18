import os
import json
import sys
import requests
from PIL import Image
import pytesseract
import fitz  # PyMuPDF
from PyPDF2 import PdfReader

# Define the models and paths
MODELS = ["llama3", "mistral", "phi"]
EXTRACTED_DATA_DIR = "extracted_data"
SCANNED_DATA_DIR = "scanned_base_data"  # Updated directory name
TEXT_DATA_DIR = "Text_base_data"        # Updated directory name

def extract_text_from_text_pdf(pdf_path):
    text = ""
    try:
        doc = fitz.open(pdf_path)
        for page in doc:
            text += page.get_text()
        return text
    except Exception:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text

def extract_text_from_scanned_pdf(pdf_path):
    text = ""
    try:
        doc = fitz.open(pdf_path)
        for page_num, page in enumerate(doc):
            pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            page_text = pytesseract.image_to_string(img)
            text += f"\n{page_text}"
        return text
    except Exception as e:
        print(f"Error in PDF processing: {str(e)}")
        return ""

def is_scanned_pdf(pdf_path):
    """
    Determine if a PDF is scanned or text-based by checking text content.
    If the extracted text is very short, it's likely a scanned document.
    """
    doc = fitz.open(pdf_path)
    text_length = 0
    for page in doc:
        text_length += len(page.get_text().strip())
    
    # If less than 100 characters per page on average, consider it a scanned PDF
    threshold = 100
    is_scanned = text_length < threshold
    print(f"PDF text length: {text_length} chars (threshold: {threshold})")
    return is_scanned

def extract_cv_info(cv_text, model_name):
    # Model-specific prompt adjustments
    if model_name == "mistral" or model_name == "phi":
        format_instruction = """
ATTENTION: Your ONLY task is to return information as a valid JSON object.
You MUST follow these rules:
1. ONLY return a valid JSON object, nothing else
2. NO explanatory text before or after the JSON
3. NO code blocks, markdown formatting, or ```json markers
4. NO commentary, explanation, or additional text of any kind
5. The response should start with '{' and end with '}'

Example of CORRECT response format:
{"name": "John Smith", "email": "john.smith@example.com"}

Example of INCORRECT response formats:
1. ```json {"name": "John Smith"} ```
2. Here's the extracted information: {"name": "John Smith"}
3. Any text at all outside the JSON object
"""
        # Add system message for phi and mistral
        system_message = "You are a data extraction tool that ONLY outputs valid JSON objects. Never output explanations or any text outside the JSON."
    else:
        format_instruction = """
IMPORTANT: ONLY RETURN A JSON OBJECT. DO NOT RETURN ANY CODE, EXPLANATIONS, OR TEXT OUTSIDE THE JSON OBJECT.
DO NOT wrap your answer in ```json or any code blocks.
DO NOT include any imports, code samples, or programming logic.
ONLY return a raw, valid JSON object with the information extracted from the CV.
"""
        system_message = ""

    prompt = f"""
    {format_instruction}
    
    You are a CV analysis expert. Extract the following information from the CV:
    
    1. Name
    2. Email
    3. Phone
    4. Education (list of degrees, institutions, years)
    5. Work Experience (list of positions, companies, years)
    6. Projects (list of projects without any explanations)
    7. Skills (technical and soft skills)

    There will be strange characters or incomplete words like e, ec, o, and... so on, discard them.
    Also, there will be some symbols like @ ,! and so on at the beginning of sentences, discard them.  
    Sometimes, you will find irregular expressions or symbols, discard them if there is missing information. leave the field empty " ".

    Return the information in a well-structured JSON format with these exact keys:
    "name", "email", "phone", "education", "work experience", "Projects", "skills"
    
    Here is an example of the expected JSON format:
    
    {{
      "name": "John Smith",
      "email": "john.smith@example.com",
      "phone": "+1 234-567-8901",
      "education": [
        {{
          "degree": "Master of Science in Computer Science",
          "institution": "Stanford University",
          "year": "2018-2020"
        }},
        {{
          "degree": "Bachelor of Engineering in Software Engineering",
          "institution": "MIT",
          "year": "2014-2018"
        }}
      ],
      "work experience": [
        {{
          "position": "Senior Software Engineer",
          "company": "Google",
          "location": "Mountain View, CA",
          "year": "2020-Present"
        }},
        {{
          "position": "Software Developer",
          "company": "Microsoft",
          "location": "Seattle, WA",
          "year": "2018-2020"
        }}
      ],
      "Projects": [
        "Machine Learning Image Recognition System",
        "E-commerce Platform Development",
        "Mobile App for Healthcare"
      ],
      "skills": [
        {{
          "technical": "Python, Java, C++, JavaScript, SQL, AWS, Docker, Kubernetes",
          "frameworks": "React, Node.js, Django, TensorFlow, PyTorch"
        }},
        {{
          "soft": "Leadership, Communication, Problem Solving",
          "languages": "English (Native), Spanish (Fluent), French (Basic)"
        }}
      ]
    }}
    
    CV Text:
    {cv_text}
    
    {format_instruction}
    """
    
    # Model-specific parameters
    temperature = 0.1  # Lower temperature for more deterministic outputs
    if model_name == "phi":
        temperature = 0.0  # Phi seems to need even lower temperature
    
    # Try up to 3 times with different approaches
    max_retries = 3
    for attempt in range(max_retries):
        try:
            request_body = {
                "model": model_name,
                "prompt": prompt,
                "stream": False,
                "temperature": temperature
            }
            
            # Add system parameter for models that support it
            if system_message and (model_name == "mistral" or model_name == "phi"):
                request_body["system"] = system_message
            
            response = requests.post(
                "http://localhost:11434/api/generate",
                json=request_body
            )
            
            if response.status_code == 200:
                result = response.json()
                response_text = result.get("response", "")
                
                # Enhanced JSON extraction
                json_str = response_text.strip()
                
                # Attempt 1: Standard extraction
                if attempt == 0:
                    # Remove any text before the first { and after the last }
                    first_brace = json_str.find('{')
                    last_brace = json_str.rfind('}')
                    
                    if first_brace != -1 and last_brace != -1:
                        json_str = json_str[first_brace:last_brace+1]
                    
                    # Remove markdown code blocks
                    if "```json" in json_str:
                        json_str = json_str.split("```json")[1].split("```")[0].strip()
                    elif "```" in json_str:
                        parts = json_str.split("```")
                        for part in parts:
                            if "{" in part and "}" in part:
                                json_str = part.strip()
                                break
                    
                # Attempt 2: More aggressive cleaning
                elif attempt == 1:
                    # Try to detect and extract just the JSON portion using regex
                    import re
                    json_match = re.search(r'\{.*\}', json_str, re.DOTALL)
                    if json_match:
                        json_str = json_match.group(0)
                
                # Attempt 3: Simplify the prompt and retry with just JSON extraction
                elif attempt == 2:
                    # Use a different temperature in the last attempt
                    temperature = 0.05
                    # Keep the structure but simplify - will use on next loop iteration
                    continue
                
                # Try parsing the cleaned JSON
                parsed_result = json.loads(json_str)
                return parsed_result
                
            else:
                print(f"API error on attempt {attempt+1}: {response.status_code}")
                # Wait briefly before retrying
                import time
                time.sleep(0.5)
                
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON on attempt {attempt+1}: {str(e)}")
            print(f"Raw response (first 100 chars): {response_text[:100]}")
            
            # If this is the last attempt, return error
            if attempt == max_retries - 1:
                return {"error": "Failed to parse JSON response after multiple attempts", 
                        "raw_response": response_text}
    
    # If we got here, all attempts failed
    return {"error": f"Error from Ollama API after {max_retries} attempts", 
            "raw_response": response_text if 'response_text' in locals() else "No response"}

def save_extraction_to_file(data, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    print(f"Saved extraction results to {file_path}")

def process_cv(pdf_path, cv_name):
    try:
        # Determine if PDF is scanned or text-based
        is_scanned = is_scanned_pdf(pdf_path)
        
        if is_scanned:
            print(f"Detected {cv_name} as a scanned PDF. Using OCR...")
            cv_text = extract_text_from_scanned_pdf(pdf_path)
            pdf_type = "scanned"
            base_dir = SCANNED_DATA_DIR
        else:
            print(f"Detected {cv_name} as a text-based PDF. Extracting text...")
            cv_text = extract_text_from_text_pdf(pdf_path)
            pdf_type = "text"
            base_dir = TEXT_DATA_DIR
        
        if not cv_text:
            print(f"Failed to extract text from {cv_name}")
            return False
            
        # Process with each model and save to appropriate directories
        for model in MODELS:
            print(f"Extracting information from {cv_name} using {model}...")
            extraction_result = extract_cv_info(cv_text, model)
            
            # Save to model-specific directory (always)
            model_output_file = os.path.join(EXTRACTED_DATA_DIR, model, f"{cv_name}.json")
            save_extraction_to_file(extraction_result, model_output_file)
            
            # Save to PDF type directory (scanned or text) based on detection result
            # This ensures files are only in one type directory
            type_output_file = os.path.join(base_dir, model, f"{cv_name}.json")
            save_extraction_to_file(extraction_result, type_output_file)
        
        return True
    except Exception as e:
        print(f"Error processing {cv_name}: {str(e)}")
        return False

def main():
    # Create directories for each model
    for model in MODELS:
        os.makedirs(os.path.join(EXTRACTED_DATA_DIR, model), exist_ok=True)
        os.makedirs(os.path.join(SCANNED_DATA_DIR, model), exist_ok=True)
        os.makedirs(os.path.join(TEXT_DATA_DIR, model), exist_ok=True)
    
    # Also create directory for ground truth if it doesn't exist
    os.makedirs("ground_truth", exist_ok=True)
    
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
        if not os.path.exists(pdf_path):
            print(f"Error: File {pdf_path} does not exist")
            return
            
        cv_name = os.path.splitext(os.path.basename(pdf_path))[0]
        process_cv(pdf_path, cv_name)
    else:
        print("Please provide the path to a CV PDF file as a command-line argument")
        print("Example: python save_extraction.py path/to/cv.pdf")

if __name__ == "__main__":
    main() 