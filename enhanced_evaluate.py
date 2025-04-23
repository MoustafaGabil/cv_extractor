#!/usr/bin/env python3
import os
import json
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import difflib
from fuzzywuzzy import fuzz
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Constants
MODELS = ["phi", "llama3", "mistral"]
EXTRACTED_DATA_DIR = "extracted_data"
GROUND_TRUTH_DIR = "ground_truth"
REPORTS_DIR = "evaluation_reports"
CV_REPORTS_DIR = os.path.join(REPORTS_DIR, "individual_cv_reports")
MODEL_REPORTS_DIR = os.path.join(REPORTS_DIR, "model_reports")
AGGREGATED_DIR = os.path.join(REPORTS_DIR, "aggregated_results")

# Create reports directory if it doesn't exist
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(CV_REPORTS_DIR, exist_ok=True)
os.makedirs(MODEL_REPORTS_DIR, exist_ok=True)
os.makedirs(AGGREGATED_DIR, exist_ok=True)

def load_json_file(file_path):
    """Load a JSON file and handle errors."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError:
        print(f"Error decoding JSON in {file_path}")
        return {}
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return {}

def normalize_field_names(data):
    """Standardize field names to handle variations."""
    field_mapping = {
        "name": ["name", "full_name", "candidate_name"],
        "email": ["email", "email_address", "contact_email"],
        "phone": ["phone", "phone_number", "contact_phone", "telephone"],
        "education": ["education", "educational_background", "academic_history", "qualifications"],
        "work_experience": ["work_experience", "experience", "professional_experience", "employment_history"],
        "skills": ["skills", "technical_skills", "competencies", "expertise"],
        "projects": ["projects", "personal_projects", "portfolio"]
    }
    
    normalized_data = {}
    
    for standard_field, variations in field_mapping.items():
        for variation in variations:
            if variation in data:
                normalized_data[standard_field] = data[variation]
                break
    
    return normalized_data

def normalize_text(text):
    """Normalize text by converting to lowercase and removing punctuation and extra spaces."""
    if not text or not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation except for email addresses and phone numbers
    text = re.sub(r'[^\w\s@.+-]', ' ', text)
    
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def normalize_phone(phone):
    """Standardize phone number formats."""
    if not phone or not isinstance(phone, str):
        return ""
    
    # Remove all non-digit characters
    digits_only = re.sub(r'\D', '', phone)
    
    return digits_only

def calculate_text_similarity(text1, text2):
    """Calculate similarity between two text strings using various metrics."""
    if not text1 and not text2:
        return 1.0  # Both are empty, consider them matching
    
    if not text1 or not text2:
        return 0.0  # One is empty, the other isn't
    
    # Normalize texts
    norm_text1 = normalize_text(text1)
    norm_text2 = normalize_text(text2)
    
    # Calculate similarity metrics
    sequence_matcher = difflib.SequenceMatcher(None, norm_text1, norm_text2)
    sequence_ratio = sequence_matcher.ratio()
    
    fuzzy_ratio = fuzz.ratio(norm_text1, norm_text2) / 100
    fuzzy_partial_ratio = fuzz.partial_ratio(norm_text1, norm_text2) / 100
    
    # Average of metrics
    avg_similarity = (sequence_ratio + fuzzy_ratio + fuzzy_partial_ratio) / 3
    
    return avg_similarity

def flatten_skills(skills_data):
    """Flatten nested skills structures into a simple list."""
    if not skills_data:
        return []
    
    if isinstance(skills_data, list):
        # Check if it's already a list of strings
        if all(isinstance(item, str) for item in skills_data):
            return skills_data
        
        # It might be a list of objects/dicts
        flattened = []
        for item in skills_data:
            if isinstance(item, dict):
                # Extract skill names from dictionaries
                for key, value in item.items():
                    if isinstance(value, str):
                        flattened.append(value)
                    elif isinstance(value, list):
                        flattened.extend([str(v) for v in value])
            elif isinstance(item, str):
                flattened.append(item)
        return flattened
    
    if isinstance(skills_data, dict):
        # If skills is a dictionary with categories
        flattened = []
        for category, skills in skills_data.items():
            if isinstance(skills, list):
                flattened.extend([str(skill) for skill in skills])
            elif isinstance(skills, str):
                flattened.append(skills)
        return flattened
    
    # If it's a string, split by commas or semicolons
    if isinstance(skills_data, str):
        return [skill.strip() for skill in re.split(r'[,;]', skills_data) if skill.strip()]
    
    return []

def extract_education_fields(edu_item):
    """Extract standardized fields from education item."""
    if isinstance(edu_item, str):
        return {"description": edu_item}
    
    if not isinstance(edu_item, dict):
        return {"description": str(edu_item)}
    
    result = {}
    
    # Common field mappings for education
    field_mapping = {
        "institution": ["institution", "university", "school", "college", "academy"],
        "degree": ["degree", "qualification", "certification", "diploma"],
        "field_of_study": ["field_of_study", "major", "subject", "course"],
        "date": ["date", "period", "duration", "years", "time_period"],
        "start_date": ["start_date", "from", "start"],
        "end_date": ["end_date", "to", "end"],
        "description": ["description", "details", "achievements"]
    }
    
    for standard_field, variations in field_mapping.items():
        for variation in variations:
            if variation in edu_item and edu_item[variation]:
                result[standard_field] = edu_item[variation]
                break
    
    # If we couldn't extract structured fields, use the whole item as description
    if not result:
        result["description"] = str(edu_item)
    
    return result

def extract_work_experience_fields(exp_item):
    """Extract standardized fields from work experience item."""
    if isinstance(exp_item, str):
        return {"description": exp_item}
    
    if not isinstance(exp_item, dict):
        return {"description": str(exp_item)}
    
    result = {}
    
    # Common field mappings for work experience
    field_mapping = {
        "company": ["company", "organization", "employer", "workplace"],
        "position": ["position", "title", "role", "job_title"],
        "date": ["date", "period", "duration", "years", "time_period"],
        "start_date": ["start_date", "from", "start"],
        "end_date": ["end_date", "to", "end"],
        "location": ["location", "place", "city", "country"],
        "description": ["description", "details", "responsibilities", "achievements"]
    }
    
    for standard_field, variations in field_mapping.items():
        for variation in variations:
            if variation in exp_item and exp_item[variation]:
                result[standard_field] = exp_item[variation]
                break
    
    # If we couldn't extract structured fields, use the whole item as description
    if not result:
        result["description"] = str(exp_item)
    
    return result

def calculate_field_precision_recall(extracted, ground_truth, field):
    """Calculate precision, recall, and F1 score for a specific field."""
    # Simple case for scalar fields
    if field in ["name", "email", "phone"]:
        extracted_value = extracted.get(field, "")
        ground_truth_value = ground_truth.get(field, "")
        
        if field == "phone":
            extracted_value = normalize_phone(extracted_value)
            ground_truth_value = normalize_phone(ground_truth_value)
            
        similarity = calculate_text_similarity(extracted_value, ground_truth_value)
        return {
            "precision": similarity,
            "recall": similarity,
            "f1": similarity,
            "extracted": extracted_value,
            "ground_truth": ground_truth_value
        }
    
    # Handle complex fields (lists or nested structures)
    if field == "skills":
        extracted_skills = flatten_skills(extracted.get("skills", []))
        ground_truth_skills = flatten_skills(ground_truth.get("skills", []))
        
        # Normalize skills
        extracted_skills = [normalize_text(skill) for skill in extracted_skills]
        ground_truth_skills = [normalize_text(skill) for skill in ground_truth_skills]
        
        # Remove empty strings
        extracted_skills = [s for s in extracted_skills if s]
        ground_truth_skills = [s for s in ground_truth_skills if s]
        
        if not ground_truth_skills:
            if not extracted_skills:
                return {"precision": 1.0, "recall": 1.0, "f1": 1.0, 
                       "extracted": [], "ground_truth": []}
            return {"precision": 0.0, "recall": 1.0, "f1": 0.0, 
                   "extracted": extracted_skills, "ground_truth": []}
        
        if not extracted_skills:
            return {"precision": 1.0, "recall": 0.0, "f1": 0.0, 
                   "extracted": [], "ground_truth": ground_truth_skills}
        
        # Calculate matches using fuzzy matching
        matches = 0
        for gt_skill in ground_truth_skills:
            # Find best match for this ground truth skill
            best_similarity = max(
                (calculate_text_similarity(gt_skill, ext_skill) for ext_skill in extracted_skills),
                default=0.0
            )
            
            # Count as match if similarity is above threshold
            if best_similarity > 0.8:
                matches += best_similarity
        
        precision = matches / len(extracted_skills) if extracted_skills else 0.0
        recall = matches / len(ground_truth_skills) if ground_truth_skills else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "extracted": extracted_skills,
            "ground_truth": ground_truth_skills
        }
    
    # Handle education and work experience
    if field in ["education", "work_experience"]:
        extracted_items = extracted.get(field, [])
        ground_truth_items = ground_truth.get(field, [])
        
        # Ensure items are in list format
        if not isinstance(extracted_items, list):
            extracted_items = [extracted_items] if extracted_items else []
        if not isinstance(ground_truth_items, list):
            ground_truth_items = [ground_truth_items] if ground_truth_items else []
        
        # Process each item to extract standardized fields
        processor_func = extract_education_fields if field == "education" else extract_work_experience_fields
        extracted_processed = [processor_func(item) for item in extracted_items]
        ground_truth_processed = [processor_func(item) for item in ground_truth_items]
        
        if not ground_truth_processed:
            if not extracted_processed:
                return {"precision": 1.0, "recall": 1.0, "f1": 1.0, 
                       "extracted": [], "ground_truth": []}
            return {"precision": 0.0, "recall": 1.0, "f1": 0.0, 
                   "extracted": extracted_processed, "ground_truth": []}
        
        if not extracted_processed:
            return {"precision": 1.0, "recall": 0.0, "f1": 0.0, 
                   "extracted": [], "ground_truth": ground_truth_processed}
        
        # Calculate item matches
        matches = 0
        for gt_item in ground_truth_processed:
            # Convert item to string for comparison
            gt_item_str = json.dumps(gt_item, sort_keys=True)
            
            # Find best match
            best_similarity = 0
            for ext_item in extracted_processed:
                ext_item_str = json.dumps(ext_item, sort_keys=True)
                similarity = calculate_text_similarity(gt_item_str, ext_item_str)
                best_similarity = max(best_similarity, similarity)
            
            matches += best_similarity
        
        precision = matches / len(extracted_processed) if extracted_processed else 0.0
        recall = matches / len(ground_truth_processed) if ground_truth_processed else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "extracted": extracted_processed,
            "ground_truth": ground_truth_processed
        }
    
    # Default case for other fields
    return {
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "extracted": None,
        "ground_truth": None
    }

def evaluate_extraction(cv_name, model):
    """Evaluate the extraction quality for a CV using a specific model."""
    # Define file paths
    ground_truth_path = os.path.join(GROUND_TRUTH_DIR, f"{cv_name}.json")
    extracted_path = os.path.join(EXTRACTED_DATA_DIR, model, f"{cv_name}.json")
    
    # Check if files exist
    if not os.path.exists(ground_truth_path):
        print(f"Ground truth file not found for {cv_name}")
        return None
    
    if not os.path.exists(extracted_path):
        print(f"Extracted file not found for {cv_name} using {model} model")
        return None
    
    # Load files
    ground_truth = load_json_file(ground_truth_path)
    extracted = load_json_file(extracted_path)
    
    if not ground_truth or not extracted:
        return None
    
    # Normalize field names
    ground_truth = normalize_field_names(ground_truth)
    extracted = normalize_field_names(extracted)
    
    # Fields to evaluate
    fields = ["name", "email", "phone", "education", "work_experience", "skills"]
    
    # Evaluate each field
    field_results = {}
    for field in fields:
        field_results[field] = calculate_field_precision_recall(extracted, ground_truth, field)
    
    # Calculate overall scores
    valid_f1_scores = [result["f1"] for field, result in field_results.items() 
                      if result["f1"] >= 0]
    
    overall_f1 = sum(valid_f1_scores) / len(valid_f1_scores) if valid_f1_scores else 0
    
    # Create error analysis report
    error_report = generate_error_report(cv_name, extracted, ground_truth)
    
    return {
        "cv_name": cv_name,
        "model": model,
        "fields": field_results,
        "overall_f1": overall_f1,
        "error_report": error_report
    }

def generate_comparison_chart(all_results, title, save_path):
    """Generate a bar chart comparing model performance."""
    models = MODELS
    cv_names = list(set(result["cv_name"] for result in all_results if result))
    
    # Prepare data
    model_averages = {model: [] for model in models}
    
    for cv_name in cv_names:
        for model in models:
            cv_results = [r for r in all_results if r and r["cv_name"] == cv_name and r["model"] == model]
            if cv_results:
                model_averages[model].append(cv_results[0]["overall_f1"])
            else:
                model_averages[model].append(0)
    
    # Calculate overall averages
    model_overall_averages = {model: sum(scores)/len(scores) if scores else 0 
                             for model, scores in model_averages.items()}
    
    # Create figure
    plt.figure(figsize=(15, 8))
    
    # Create bar chart
    x = np.arange(len(cv_names))
    width = 0.2
    multiplier = 0
    
    for model, scores in model_averages.items():
        offset = width * multiplier
        plt.bar(x + offset, scores, width, label=model)
        multiplier += 1
    
    # Add labels and title
    plt.xlabel('CV Name')
    plt.ylabel('F1 Score')
    plt.title(title)
    plt.xticks(x + width, cv_names, rotation=45, ha='right')
    plt.legend(loc='upper left')
    
    # Add average line for each model
    for i, model in enumerate(models):
        plt.axhline(y=model_overall_averages[model], color=f'C{i}', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def generate_field_comparison_chart(all_results, cv_name, save_path):
    """Generate a chart comparing model performance for a specific CV across different fields."""
    models = MODELS
    fields = ["name", "email", "phone", "education", "work_experience", "skills"]
    
    # Filter results for this CV
    cv_results = [r for r in all_results if r and r["cv_name"] == cv_name]
    
    if not cv_results:
        return
    
    # Prepare data
    field_scores = {field: [] for field in fields}
    
    for model in models:
        model_results = [r for r in cv_results if r["model"] == model]
        
        if model_results:
            result = model_results[0]
            for field in fields:
                if field in result["fields"]:
                    field_scores[field].append(result["fields"][field]["f1"])
                else:
                    field_scores[field].append(0)
        else:
            for field in fields:
                field_scores[field].append(0)
    
    # Create figure
    plt.figure(figsize=(15, 8))
    
    # Create bar chart
    x = np.arange(len(fields))
    width = 0.2
    multiplier = 0
    
    for i, model in enumerate(models):
        offset = width * multiplier
        scores = [field_scores[field][i] for field in fields]
        plt.bar(x + offset, scores, width, label=model)
        multiplier += 1
    
    # Add labels and title
    plt.xlabel('Field')
    plt.ylabel('F1 Score')
    plt.title(f'Field Performance Comparison for {cv_name}')
    plt.xticks(x + width, fields, rotation=45, ha='right')
    plt.legend(loc='upper left')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def export_to_excel(all_results):
    """Export evaluation results to an Excel file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    excel_path = os.path.join(AGGREGATED_DIR, f"evaluation_results_{timestamp}.xlsx")
    
    # Create a summary DataFrame
    summary_data = []
    
    for result in all_results:
        if not result:
            continue
            
        cv_name = result["cv_name"]
        model = result["model"]
        overall_f1 = result["overall_f1"]
        
        # Get individual field scores
        field_scores = {field: result["fields"][field]["f1"] 
                       for field in result["fields"] if "f1" in result["fields"][field]}
        
        row = {
            "CV Name": cv_name,
            "Model": model,
            "Overall F1": overall_f1
        }
        
        # Add field scores
        for field, score in field_scores.items():
            row[f"{field.capitalize()} F1"] = score
        
        summary_data.append(row)
    
    # Create DataFrame
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        
        # Create detailed DataFrames for each model
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # Group by model
            for model in MODELS:
                model_results = [r for r in all_results if r and r["model"] == model]
                
                if not model_results:
                    continue
                
                # Prepare detailed data
                detailed_data = []
                
                for result in model_results:
                    cv_name = result["cv_name"]
                    
                    for field in result["fields"]:
                        field_result = result["fields"][field]
                        
                        row = {
                            "CV Name": cv_name,
                            "Field": field,
                            "Precision": field_result.get("precision", 0),
                            "Recall": field_result.get("recall", 0),
                            "F1": field_result.get("f1", 0)
                        }
                        
                        detailed_data.append(row)
                
                # Create DataFrame and save
                if detailed_data:
                    detailed_df = pd.DataFrame(detailed_data)
                    detailed_df.to_excel(writer, sheet_name=f'{model} Details', index=False)
        
        print(f"Results exported to {excel_path}")
        return excel_path
    else:
        print("No valid results to export")
        return None

def generate_error_report(cv_name, extracted, ground_truth):
    """Generate a detailed error report for a CV."""
    report = {
        "missing_fields": [],
        "extra_fields": [],
        "field_errors": {}
    }
    
    # Check for missing fields
    for field in ground_truth:
        if field not in extracted or not extracted[field]:
            report["missing_fields"].append(field)
    
    # Check for extra fields
    for field in extracted:
        if field not in ground_truth or not ground_truth[field]:
            report["extra_fields"].append(field)
    
    # Check field content errors
    common_fields = [f for f in ground_truth if f in extracted and ground_truth[f] and extracted[f]]
    
    for field in common_fields:
        gt_value = ground_truth[field]
        ext_value = extracted[field]
        
        # Skip identical values
        if gt_value == ext_value:
            continue
        
        # For simple fields, calculate similarity
        if field in ["name", "email", "phone"]:
            similarity = calculate_text_similarity(str(gt_value), str(ext_value))
            
            if similarity < 0.9:  # Threshold for reporting an error
                report["field_errors"][field] = {
                    "ground_truth": gt_value,
                    "extracted": ext_value,
                    "similarity": similarity
                }
        
        # For complex fields
        elif field in ["education", "work_experience", "skills"]:
            # Convert to lists if not already
            gt_items = gt_value if isinstance(gt_value, list) else [gt_value]
            ext_items = ext_value if isinstance(ext_value, list) else [ext_value]
            
            # Check length differences
            if len(gt_items) != len(ext_items):
                report["field_errors"][field] = {
                    "error_type": "count_mismatch",
                    "ground_truth_count": len(gt_items),
                    "extracted_count": len(ext_items)
                }
            
            # For skills, check missing and extra skills
            if field == "skills":
                gt_skills = flatten_skills(gt_value)
                ext_skills = flatten_skills(ext_value)
                
                # Normalize skills
                gt_skills = [normalize_text(skill) for skill in gt_skills if skill]
                ext_skills = [normalize_text(skill) for skill in ext_skills if skill]
                
                missing_skills = []
                for gt_skill in gt_skills:
                    # Check if any extracted skill is similar
                    best_match = max(
                        (calculate_text_similarity(gt_skill, ext_skill) for ext_skill in ext_skills),
                        default=0
                    )
                    
                    if best_match < 0.8:  # Threshold for considering a skill missing
                        missing_skills.append(gt_skill)
                
                extra_skills = []
                for ext_skill in ext_skills:
                    # Check if any ground truth skill is similar
                    best_match = max(
                        (calculate_text_similarity(ext_skill, gt_skill) for gt_skill in gt_skills),
                        default=0
                    )
                    
                    if best_match < 0.8:  # Threshold for considering a skill extra
                        extra_skills.append(ext_skill)
                
                if missing_skills or extra_skills:
                    report["field_errors"][field] = {
                        "error_type": "content_mismatch",
                        "missing_items": missing_skills,
                        "extra_items": extra_skills
                    }
    
    return report

def main():
    """Run the evaluation."""
    print("Starting CV extraction evaluation...")
    
    # Get all CVs from ground truth directory
    cv_names = []
    for filename in os.listdir(GROUND_TRUTH_DIR):
        if filename.endswith(".json"):
            cv_name = filename[:-5]  # Remove .json extension
            cv_names.append(cv_name)
    
    print(f"Found {len(cv_names)} CVs in ground truth directory")
    
    # Evaluate each CV for each model
    all_results = []
    
    for cv_name in cv_names:
        print(f"\nEvaluating {cv_name}...")
        
        for model in MODELS:
            print(f"  Using {model} model...")
            result = evaluate_extraction(cv_name, model)
            
            if result:
                all_results.append(result)
                print(f"  Overall F1 score: {result['overall_f1']:.4f}")
            else:
                print(f"  Evaluation failed")
    
    # Generate overall comparison chart
    chart_path = os.path.join(AGGREGATED_DIR, "model_comparison.png")
    generate_comparison_chart(all_results, "Model Performance Comparison", chart_path)
    print(f"\nGenerated overall comparison chart: {chart_path}")
    
    # Generate individual CV field comparison charts
    for cv_name in cv_names:
        cv_chart_path = os.path.join(CV_REPORTS_DIR, f"{cv_name}_field_comparison.png")
        generate_field_comparison_chart(all_results, cv_name, cv_chart_path)
        print(f"Generated field comparison chart for {cv_name}: {cv_chart_path}")
    
    # Export results to Excel
    excel_path = export_to_excel(all_results)
    
    print("\nEvaluation complete!")
    if excel_path:
        print(f"Results saved to {excel_path}")
    print(f"Charts saved to evaluation_reports directory and subdirectories")

if __name__ == "__main__":
    main() 