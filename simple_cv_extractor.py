import os
import json
import tempfile
import requests
import streamlit as st
from PIL import Image
import fitz  # PyMuPDF
import time
import threading
from streamlit.runtime.scriptrunner import add_script_run_ctx

# Set page title and configuration
st.set_page_config(page_title="CV Extractor", layout="wide")
st.title("CV Information Extractor")

# Global variables
MODELS = ["phi", "llama3", "mistral"]

# Initialize session state for tracking model runs
if 'processing' not in st.session_state:
    st.session_state.processing = False
if 'results' not in st.session_state:
    st.session_state.results = {}

def extract_text_from_pdf(pdf_path):
    """Extract text from PDF file"""
    text = ""
    try:
        doc = fitz.open(pdf_path)
        for page in doc:
            text += page.get_text()
        return text
    except Exception as e:
        st.error(f"Error extracting text: {str(e)}")
        return ""

def is_scanned_pdf(pdf_path):
    """Check if PDF is scanned (low text content)"""
    doc = fitz.open(pdf_path)
    text_length = 0
    for page in doc:
        text_length += len(page.get_text().strip())
    return text_length < 100

def format_cv_data(data):
    """Format CV data as readable text"""
    text = []
    
    if "name" in data and data["name"]:
        text.append(f"## Name\n{data['name']}")
    
    if "email" in data and data["email"]:
        text.append(f"## Email\n{data['email']}")
    
    if "phone" in data and data["phone"]:
        text.append(f"## Phone\n{data['phone']}")
    
    if "education" in data and data["education"]:
        text.append("## Education")
        if isinstance(data["education"], list):
            for edu in data["education"]:
                if isinstance(edu, dict):
                    edu_text = []
                    for key, value in edu.items():
                        edu_text.append(f"{key}: {value}")
                    text.append("- " + " | ".join(edu_text))
                else:
                    text.append(f"- {edu}")
        else:
            text.append(f"- {data['education']}")
    
    if "work_experience" in data and data["work_experience"]:
        text.append("## Work Experience")
        if isinstance(data["work_experience"], list):
            for exp in data["work_experience"]:
                if isinstance(exp, dict):
                    exp_text = []
                    for key, value in exp.items():
                        exp_text.append(f"{key}: {value}")
                    text.append("- " + " | ".join(exp_text))
                else:
                    text.append(f"- {exp}")
        else:
            text.append(f"- {data['work_experience']}")
    
    # Try alternative key "work experience" if "work_experience" not found
    elif "work experience" in data and data["work experience"]:
        text.append("## Work Experience")
        if isinstance(data["work experience"], list):
            for exp in data["work experience"]:
                if isinstance(exp, dict):
                    exp_text = []
                    for key, value in exp.items():
                        exp_text.append(f"{key}: {value}")
                    text.append("- " + " | ".join(exp_text))
                else:
                    text.append(f"- {exp}")
        else:
            text.append(f"- {data['work experience']}")
    
    if "projects" in data and data["projects"]:
        text.append("## Projects")
        if isinstance(data["projects"], list):
            for project in data["projects"]:
                text.append(f"- {project}")
        else:
            text.append(f"- {data['projects']}")
    
    # Try alternative key "Projects" if "projects" not found
    elif "Projects" in data and data["Projects"]:
        text.append("## Projects")
        if isinstance(data["Projects"], list):
            for project in data["Projects"]:
                text.append(f"- {project}")
        else:
            text.append(f"- {data['Projects']}")
    
    if "skills" in data and data["skills"]:
        text.append("## Skills")
        if isinstance(data["skills"], list):
            for skill in data["skills"]:
                if isinstance(skill, dict):
                    for skill_type, skill_value in skill.items():
                        text.append(f"### {skill_type.capitalize()}")
                        text.append(f"{skill_value}")
                else:
                    text.append(f"- {skill}")
        elif isinstance(data["skills"], dict):
            for skill_type, skill_value in data["skills"].items():
                text.append(f"### {skill_type.capitalize()}")
                if isinstance(skill_value, list):
                    for item in skill_value:
                        text.append(f"- {item}")
                else:
                    text.append(f"{skill_value}")
        else:
            text.append(f"- {data['skills']}")
    
    return "\n\n".join(text)

def run_in_thread(func, *args, **kwargs):
    """Run a function in a thread with Streamlit context"""
    thread = threading.Thread(target=func, args=args, kwargs=kwargs)
    add_script_run_ctx(thread)
    thread.start()
    return thread

def extract_cv_info(cv_text, model_name, result_container):
    """Extract CV information using the specified model"""
    # Prepare the prompt
    prompt = f"""
    You are a CV analysis expert. Extract the following information from the CV:
    
    1. Name
    2. Email
    3. Phone
    4. Education (list of degrees, institutions, years)
    5. Work Experience (list of positions, companies, years)
    6. Projects (list of projects)
    7. Skills (technical and soft skills)

    Return the information in a well-structured JSON format with these exact keys:
    "name", "email", "phone", "education", "work_experience", "projects", "skills"
    
    CV Text:
    {cv_text}
    
    IMPORTANT: ONLY RETURN A JSON OBJECT. DO NOT INCLUDE ANY TEXT OUTSIDE THE JSON.
    """
    
    try:
        # Check if Ollama is running
        try:
            requests.get("http://localhost:11434/api/tags", timeout=2)
        except requests.exceptions.ConnectionError:
            result_container.error("Ollama is not running. Please start Ollama with 'ollama serve' in a separate terminal.")
            return None
        except requests.exceptions.Timeout:
            result_container.error("Ollama server is not responding. Please check if it's running correctly.")
            return None
            
        # Stream the response to avoid timeout
        status_text = result_container.empty()
        status_text.info(f"Starting extraction with {model_name}... This may take several minutes for large CVs.")
        
        # Set up streaming
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model_name,
                "prompt": prompt,
                "stream": True,  # Enable streaming
                "temperature": 0.1
            },
            stream=True  # Enable HTTP streaming
        )
        
        if response.status_code != 200:
            result_container.error(f"API error: {response.status_code}")
            return None
            
        # Process streamed response
        full_response = ""
        progress_bar = result_container.progress(0)
        
        # Animated dots for the status message
        dots = 1
        
        for i, line in enumerate(response.iter_lines()):
            if line:
                chunk = json.loads(line)
                if "response" in chunk:
                    full_response += chunk["response"]
                    
                    # Update status with animated dots
                    if i % 20 == 0:
                        dots = (dots % 3) + 1
                        status_text.info(f"Extracting with {model_name}" + "." * dots)
                        progress_value = min(0.99, (i / 1000))  # Arbitrary max size assumption
                        progress_bar.progress(progress_value)
        
        progress_bar.progress(1.0)
        status_text.success(f"Extraction with {model_name} complete!")
        
        # Extract JSON from the response
        json_str = full_response.strip()
        
        # Find JSON boundaries
        first_brace = json_str.find('{')
        last_brace = json_str.rfind('}')
        
        if first_brace != -1 and last_brace != -1:
            json_str = json_str[first_brace:last_brace+1]
        
        # Remove markdown code blocks if present
        if "```json" in json_str:
            json_str = json_str.split("```json")[1].split("```")[0].strip()
        elif "```" in json_str:
            parts = json_str.split("```")
            for part in parts:
                if "{" in part and "}" in part:
                    json_str = part.strip()
                    break
        
        # Parse JSON
        parsed_result = json.loads(json_str)
        
        # Display formatted result
        result_container.markdown(format_cv_data(parsed_result))
        
        # Show JSON
        with result_container.expander("View as JSON"):
            result_container.json(parsed_result)
            
        # Add download button for JSON
        json_str = json.dumps(parsed_result, indent=2)
        result_container.download_button(
            label="Download JSON",
            data=json_str,
            file_name=f"{model_name}_extraction.json",
            mime="application/json"
        )
        
        return parsed_result
            
    except requests.exceptions.ConnectionError:
        result_container.error("Connection error. Make sure Ollama is running at localhost:11434")
        return None
    except json.JSONDecodeError as e:
        result_container.error(f"Failed to parse JSON response: {str(e)}")
        with result_container.expander("View raw response"):
            result_container.text(full_response if 'full_response' in locals() else "No response")
        return None
    except Exception as e:
        result_container.error(f"Unexpected error: {str(e)}")
        return None

# Sidebar
st.sidebar.title("CV Extractor Options")

# Model selection
st.sidebar.subheader("Select Models")
selected_models = []
for model in MODELS:
    if st.sidebar.checkbox(f"Use {model}", value=True):
        selected_models.append(model)

# Main content - Only upload option
st.subheader("Upload CV (PDF format)")
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    # Save to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp:
        temp.write(uploaded_file.read())
        temp_path = temp.name
    
    # Process button
    process_button = st.button("Process CV")
    
    if process_button:
        # Check if scanned
        if is_scanned_pdf(temp_path):
            st.warning("This appears to be a scanned PDF. Text extraction may be incomplete.")
        
        # Extract text
        cv_text = extract_text_from_pdf(temp_path)
        
        # Show raw text
        with st.expander("View extracted raw text"):
            st.text_area("CV Text", cv_text, height=300)
        
        # Process with selected models
        if selected_models:
            # Create tabs for results
            tabs = st.tabs(selected_models)
            
            # Process each model in its own tab
            for i, model in enumerate(selected_models):
                with tabs[i]:
                    result_container = st.container()
                    extract_cv_info(cv_text, model, result_container)
        else:
            st.warning("Please select at least one model from the sidebar.")
    
        # Clean up temp file after processing
        try:
            os.unlink(temp_path)
        except Exception:
            pass

# Instructions
with st.expander("How to use"):
    st.markdown("""
    ## CV Extractor Instructions
    
    1. **Upload CV**: Upload a PDF file containing a CV/resume
    2. **Select Models**: Choose which LLMs to use for extraction from the sidebar
    3. **Process CV**: Click the "Process CV" button to start extraction
    4. **View Results**: See the extracted information for each selected model in tabs
    5. **Download Results**: Use the download buttons to save extraction results if needed
    
    ### Notes:
    - Make sure Ollama is running locally with the required models
    - For best results, use text-based PDFs
    - No files are automatically saved to the filesystem
    - This version has no timeout limits - it will wait as long as needed for the model to complete
    """)

# Setup Information
with st.expander("Setup Requirements"):
    st.markdown("""
    ## Setup Requirements
    
    Before running this application, make sure to:
    
    1. Install Ollama: [https://github.com/ollama/ollama](https://github.com/ollama/ollama)
    
    2. Install the required models:
    ```bash
    ollama pull llama3
    ollama pull mistral
    ollama pull phi
    ```
    
    3. Start Ollama service:
    ```bash
    ollama serve
    ```
    
    ## Troubleshooting
    
    If you encounter issues:
    1. Make sure Ollama is running (`ollama serve` in a separate terminal)
    2. Check if the models are properly installed (`ollama list`)
    3. For very large CVs, the extraction might take several minutes - the app will wait as long as needed
    4. If the app seems stuck, try using a different model or restarting Ollama
    """)

# Footer
st.sidebar.markdown("---")
st.sidebar.info("CV Information Extractor v1.0") 