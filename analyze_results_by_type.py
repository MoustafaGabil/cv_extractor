#!/usr/bin/env python3
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
from datetime import datetime

# Define categories
SCANNED_CVS = ["John_sc", "Ihor_sc", "soha_sc"]
TEXT_BASED_CVS = ["Mohammed", "Amr", "Majid", "Adheeba_Resume", "Nicolaas", "Urson Callens", "Wouter", "Levin Boon"]
MODELS = ["phi", "llama3", "mistral"]
FIELDS = ["name", "email", "phone", "education", "work_experience", "skills"]

# Directories
REPORTS_DIR = "evaluation_reports"
ANALYSIS_DIR = os.path.join(REPORTS_DIR, "type_analysis")
os.makedirs(ANALYSIS_DIR, exist_ok=True)

def get_latest_excel_report():
    """Get the most recent evaluation Excel report."""
    excel_files = glob.glob(os.path.join(REPORTS_DIR, "evaluation_results_*.xlsx"))
    if not excel_files:
        print("No evaluation reports found.")
        return None
    
    latest_file = max(excel_files, key=os.path.getmtime)
    print(f"Using latest report: {latest_file}")
    return latest_file

def analyze_by_cv_type(excel_path):
    """Analyze results by CV type (scanned vs text-based)."""
    # Read the Excel file
    summary_df = pd.read_excel(excel_path, sheet_name='Summary')
    
    # Print actual CV names in the report
    print("\nAvailable CV names in the report:")
    print(summary_df['CV Name'].unique())
    
    # Print CV models in the report
    print("\nAvailable models in the report:")
    print(summary_df['Model'].unique())
    
    # Extract data by CV type
    scanned_df = summary_df[summary_df['CV Name'].isin(SCANNED_CVS)]
    text_based_df = summary_df[summary_df['CV Name'].isin(TEXT_BASED_CVS)]
    
    # Print data counts
    print(f"\nScanned CVs count: {len(scanned_df)}")
    print(f"Text-based CVs count: {len(text_based_df)}")
    
    # Check if all models have data for each CV type
    print("\nModel distribution in scanned CVs:")
    print(scanned_df['Model'].value_counts())
    
    print("\nModel distribution in text-based CVs:")
    print(text_based_df['Model'].value_counts())
    
    # Group by model and calculate average scores
    scanned_model_avg = scanned_df.groupby('Model')['Overall F1'].mean().reset_index()
    text_based_model_avg = text_based_df.groupby('Model')['Overall F1'].mean().reset_index()
    all_model_avg = summary_df.groupby('Model')['Overall F1'].mean().reset_index()
    
    # Print summary statistics
    print("\n--- SUMMARY STATISTICS ---")
    print("\nScanned CVs - Average F1 Score by Model:")
    print(scanned_model_avg)
    
    print("\nText-based CVs - Average F1 Score by Model:")
    print(text_based_model_avg)
    
    print("\nAll CVs - Average F1 Score by Model:")
    print(all_model_avg)
    
    # Create comparison chart
    create_type_comparison_chart(scanned_model_avg, text_based_model_avg, all_model_avg)
    
    # Analyze field-specific performance
    field_columns = [col for col in summary_df.columns if col.endswith(' F1') and col != 'Overall F1']
    if field_columns:
        analyze_field_performance(summary_df, scanned_df, text_based_df, field_columns)

def create_type_comparison_chart(scanned_avg, text_based_avg, all_avg):
    """Create a chart comparing model performance by CV type."""
    plt.figure(figsize=(12, 8))
    
    # Set up bar chart
    x = np.arange(len(MODELS))
    width = 0.25
    
    # Create bars
    plt.bar(x - width, scanned_avg['Overall F1'], width, label='Scanned CVs')
    plt.bar(x, text_based_avg['Overall F1'], width, label='Text-based CVs')
    plt.bar(x + width, all_avg['Overall F1'], width, label='All CVs')
    
    # Add labels and formatting
    plt.xlabel('Model')
    plt.ylabel('Average F1 Score')
    plt.title('Model Performance by CV Type')
    plt.xticks(x, MODELS)
    plt.ylim(0, 1.0)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Save chart
    chart_path = os.path.join(ANALYSIS_DIR, "cv_type_comparison.png")
    plt.savefig(chart_path)
    plt.close()
    print(f"\nSaved CV type comparison chart: {chart_path}")

def analyze_field_performance(summary_df, scanned_df, text_based_df, field_columns):
    """Analyze and visualize field-specific performance by CV type."""
    # Create clean field names for display
    clean_field_names = [col.replace(' F1', '') for col in field_columns]
    
    # Calculate averages by model and field
    results = {}
    
    # For each model
    for model in MODELS:
        model_results = {
            'scanned': {},
            'text_based': {},
            'all': {}
        }
        
        # Filter data for this model
        model_all = summary_df[summary_df['Model'] == model]
        model_scanned = scanned_df[scanned_df['Model'] == model]
        model_text_based = text_based_df[text_based_df['Model'] == model]
        
        # Calculate field averages
        for field_col, field_name in zip(field_columns, clean_field_names):
            model_results['scanned'][field_name] = model_scanned[field_col].mean() if not model_scanned.empty else 0
            model_results['text_based'][field_name] = model_text_based[field_col].mean() if not model_text_based.empty else 0
            model_results['all'][field_name] = model_all[field_col].mean() if not model_all.empty else 0
        
        results[model] = model_results
        
        # Create field performance chart for this model
        create_field_chart(model, model_results, clean_field_names)
    
    # Create overall field performance chart (averaged across models)
    create_overall_field_chart(results, clean_field_names)

def create_field_chart(model, model_results, field_names):
    """Create a chart showing field performance for a specific model by CV type."""
    plt.figure(figsize=(14, 8))
    
    # Set up bar chart
    x = np.arange(len(field_names))
    width = 0.25
    
    # Extract values for each category
    scanned_values = [model_results['scanned'].get(field, 0) for field in field_names]
    text_based_values = [model_results['text_based'].get(field, 0) for field in field_names]
    all_values = [model_results['all'].get(field, 0) for field in field_names]
    
    # Create bars
    plt.bar(x - width, scanned_values, width, label='Scanned CVs')
    plt.bar(x, text_based_values, width, label='Text-based CVs')
    plt.bar(x + width, all_values, width, label='All CVs')
    
    # Add labels and formatting
    plt.xlabel('Field')
    plt.ylabel('Average F1 Score')
    plt.title(f'Field Performance by CV Type - {model} Model')
    plt.xticks(x, field_names, rotation=45, ha='right')
    plt.ylim(0, 1.0)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save chart
    chart_path = os.path.join(ANALYSIS_DIR, f"{model}_field_performance.png")
    plt.savefig(chart_path)
    plt.close()
    print(f"Saved field performance chart for {model}: {chart_path}")

def create_overall_field_chart(results, field_names):
    """Create a chart showing overall field performance across all models by CV type."""
    plt.figure(figsize=(14, 8))
    
    # Set up bar chart
    x = np.arange(len(field_names))
    width = 0.25
    
    # Calculate average values across all models
    scanned_values = []
    text_based_values = []
    all_values = []
    
    for field in field_names:
        scanned_avg = sum(results[model]['scanned'].get(field, 0) for model in MODELS) / len(MODELS)
        text_based_avg = sum(results[model]['text_based'].get(field, 0) for model in MODELS) / len(MODELS)
        all_avg = sum(results[model]['all'].get(field, 0) for model in MODELS) / len(MODELS)
        
        scanned_values.append(scanned_avg)
        text_based_values.append(text_based_avg)
        all_values.append(all_avg)
    
    # Create bars
    plt.bar(x - width, scanned_values, width, label='Scanned CVs')
    plt.bar(x, text_based_values, width, label='Text-based CVs')
    plt.bar(x + width, all_values, width, label='All CVs')
    
    # Add labels and formatting
    plt.xlabel('Field')
    plt.ylabel('Average F1 Score (All Models)')
    plt.title('Overall Field Performance by CV Type')
    plt.xticks(x, field_names, rotation=45, ha='right')
    plt.ylim(0, 1.0)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save chart
    chart_path = os.path.join(ANALYSIS_DIR, "overall_field_performance.png")
    plt.savefig(chart_path)
    plt.close()
    print(f"Saved overall field performance chart: {chart_path}")

def create_summary_excel(excel_path):
    """Create a summary Excel file with CV type analysis."""
    # Read the Excel file
    summary_df = pd.read_excel(excel_path, sheet_name='Summary')
    
    # Add a 'CV Type' column
    summary_df['CV Type'] = summary_df['CV Name'].apply(
        lambda x: 'Scanned' if x in SCANNED_CVS else 'Text-based'
    )
    
    # Create a pivot table for model performance by CV type
    pivot_df = summary_df.pivot_table(
        index='Model',
        columns='CV Type',
        values='Overall F1',
        aggfunc='mean'
    ).reset_index()
    
    # Add an 'All' column
    pivot_df['All'] = summary_df.groupby('Model')['Overall F1'].mean().values
    
    # Field-specific analysis
    field_columns = [col for col in summary_df.columns if col.endswith(' F1') and col != 'Overall F1']
    field_dfs = []
    
    for field_col in field_columns:
        field_name = field_col.replace(' F1', '')
        
        field_pivot = summary_df.pivot_table(
            index='Model',
            columns='CV Type',
            values=field_col,
            aggfunc='mean'
        ).reset_index()
        
        field_pivot['All'] = summary_df.groupby('Model')[field_col].mean().values
        field_pivot['Field'] = field_name
        
        # Reorder columns
        field_pivot = field_pivot[['Field', 'Model', 'Scanned', 'Text-based', 'All']]
        field_dfs.append(field_pivot)
    
    # Combine all field analyses
    fields_df = pd.concat(field_dfs, ignore_index=True)
    
    # Save to Excel
    output_path = os.path.join(ANALYSIS_DIR, "cv_type_analysis.xlsx")
    with pd.ExcelWriter(output_path) as writer:
        pivot_df.to_excel(writer, sheet_name='Overall F1 by CV Type', index=False)
        fields_df.to_excel(writer, sheet_name='Field F1 by CV Type', index=False)
        summary_df.to_excel(writer, sheet_name='Full Data', index=False)
    
    print(f"\nSaved summary Excel file: {output_path}")

def main():
    """Main function to run the analysis."""
    print("Starting CV type analysis...")
    
    # Get the latest evaluation report
    excel_path = get_latest_excel_report()
    if not excel_path:
        return
    
    # Analyze by CV type
    analyze_by_cv_type(excel_path)
    
    # Create summary Excel file
    create_summary_excel(excel_path)
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main() 