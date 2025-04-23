#!/usr/bin/env python3
import os
import pandas as pd
import glob
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
plt.style.use('ggplot')
sns.set_palette("tab10")

# Directories
REPORTS_DIR = "evaluation_reports"
AGGREGATED_DIR = os.path.join(REPORTS_DIR, "aggregated_results")
MODEL_REPORTS_DIR = os.path.join(REPORTS_DIR, "model_reports")
CV_REPORTS_DIR = os.path.join(REPORTS_DIR, "individual_cv_reports")

# Make sure directories exist
os.makedirs(AGGREGATED_DIR, exist_ok=True)
os.makedirs(MODEL_REPORTS_DIR, exist_ok=True)
os.makedirs(CV_REPORTS_DIR, exist_ok=True)

# Models to analyze
MODELS = ["phi", "llama3", "mistral"]
# Model colors for consistent visualization
MODEL_COLORS = {
    "phi": "#ff9900",
    "llama3": "#0099ff",
    "mistral": "#00cc66"
}

# Function to automatically categorize CVs
def categorize_cvs(df):
    """
    Automatically categorize CVs as scanned or text-based based on naming convention or file metadata.
    
    By default:
    - CVs with "_sc" in their name are considered scanned
    - All others are considered text-based
    
    Returns a dictionary with 'scanned' and 'text_based' lists of CV names.
    """
    all_cvs = df['CV Name'].unique()
    
    # Default categorization based on naming convention
    # Files with "_sc" suffix are treated as scanned
    scanned_cvs = [cv for cv in all_cvs if "_sc" in cv.lower()]
    text_based_cvs = [cv for cv in all_cvs if cv not in scanned_cvs]
    
    print(f"\nAutomatically categorized CVs based on filename:")
    print(f"Scanned CVs ({len(scanned_cvs)}): {', '.join(scanned_cvs)}")
    print(f"Text-based CVs ({len(text_based_cvs)}): {', '.join(text_based_cvs)}")
    
    return {
        'scanned': scanned_cvs,
        'text_based': text_based_cvs
    }

# Get latest Excel report
def get_latest_excel_report():
    # Check in aggregated_results directory first
    excel_files = glob.glob(os.path.join(AGGREGATED_DIR, "evaluation_results_*.xlsx"))
    
    # If not found, check in the main evaluation_reports directory
    if not excel_files:
        excel_files = glob.glob(os.path.join(REPORTS_DIR, "evaluation_results_*.xlsx"))
    
    if not excel_files:
        print("No evaluation reports found in either directory.")
        return None
    
    latest_file = max(excel_files, key=os.path.getmtime)
    print(f"Using latest report: {latest_file}")
    return latest_file

# Function to create and save charts
def create_charts(pivot, field_summary_df):
    # Create overall model comparison chart
    plt.figure(figsize=(12, 6))
    
    # Set up bar positions
    models = pivot['Model'].tolist()
    x = np.arange(len(models))
    width = 0.25
    
    # Get columns without 'Model'
    categories = [col for col in pivot.columns if col != 'Model']
    
    # Create bars for each category
    for i, category in enumerate(categories):
        plt.bar(x + (i - len(categories)/2 + 0.5) * width, 
                pivot[category], 
                width, 
                label=category,
                color=sns.color_palette("husl", len(categories))[i])
    
    # Add labels and formatting
    plt.xlabel('Model')
    plt.ylabel('Average F1 Score')
    plt.title('Model Performance by CV Type')
    plt.xticks(x, models)
    plt.ylim(0, 1.0)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Save the chart
    plt.tight_layout()
    chart_path = os.path.join(AGGREGATED_DIR, "cv_type_comparison.png")
    plt.savefig(chart_path)
    print(f"\nSaved model comparison chart to {chart_path}")
    plt.close()
    
    # Create field performance chart
    # Group field summary by Field and calculate average across models
    field_avg = field_summary_df.groupby('Field')[['Scanned', 'Text-based', 'All']].mean().reset_index()
    
    plt.figure(figsize=(14, 8))
    
    # Set up bar positions
    fields = field_avg['Field'].tolist()
    x = np.arange(len(fields))
    width = 0.25
    
    # Get columns without 'Field'
    categories = ['Scanned', 'Text-based', 'All']
    
    # Create bars for each category
    for i, category in enumerate(categories):
        plt.bar(x + (i - len(categories)/2 + 0.5) * width, 
                field_avg[category], 
                width, 
                label=category,
                color=sns.color_palette("husl", len(categories))[i])
    
    # Add labels and formatting
    plt.xlabel('Field')
    plt.ylabel('Average F1 Score')
    plt.title('Field Performance by CV Type (All Models)')
    plt.xticks(x, fields, rotation=45, ha='right')
    plt.ylim(0, 1.0)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Save the chart
    plt.tight_layout()
    chart_path = os.path.join(AGGREGATED_DIR, "overall_field_performance.png")
    plt.savefig(chart_path)
    print(f"Saved field performance chart to {chart_path}")
    plt.close()
    
    # Create model-specific field performance charts
    for model in MODELS:
        model_fields = field_summary_df[field_summary_df['Model'] == model]
        
        plt.figure(figsize=(12, 6))
        
        # Set up bar positions
        fields = model_fields['Field'].tolist()
        x = np.arange(len(fields))
        width = 0.25
        
        # Get columns without 'Field' and 'Model'
        categories = ['Scanned', 'Text-based', 'All']
        
        # Create bars for each category
        for i, category in enumerate(categories):
            plt.bar(x + (i - len(categories)/2 + 0.5) * width, 
                    model_fields[category], 
                    width, 
                    label=category,
                    color=sns.color_palette("husl", len(categories))[i])
        
        # Add labels and formatting
        plt.xlabel('Field')
        plt.ylabel('Average F1 Score')
        plt.title(f'Field Performance - {model}')
        plt.xticks(x, fields, rotation=45, ha='right')
        plt.ylim(0, 1.0)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Save the chart
        plt.tight_layout()
        chart_path = os.path.join(MODEL_REPORTS_DIR, f"{model}_field_performance.png")
        plt.savefig(chart_path)
        print(f"Saved field performance chart for {model} to {chart_path}")
        plt.close()

# Main function
def main():
    # Get the Excel file
    excel_path = get_latest_excel_report()
    if not excel_path:
        return
    
    # Read the data
    df = pd.read_excel(excel_path, sheet_name='Summary')
    
    # Print column names
    print("\nColumns in the Excel file:")
    print(df.columns.tolist())
    
    # Automatically categorize CVs
    cv_categories = categorize_cvs(df)
    
    # Add a CV Type column
    df['CV Type'] = df['CV Name'].apply(
        lambda x: 'Scanned' if x in cv_categories['scanned'] else 'Text-based'
    )
    
    # Print how many of each type we have
    print("\nCV Type counts:")
    print(df['CV Type'].value_counts())
    
    # Print number of CVs by type and model
    print("\nNumber of CVs by type and model:")
    print(pd.crosstab(df['Model'], df['CV Type']))
    
    # Group by Model and CV Type
    grouped = df.groupby(['Model', 'CV Type'])['Overall F1'].mean().reset_index()
    
    # Reshape to have Model as rows and CV Type as columns
    pivot = grouped.pivot(index='Model', columns='CV Type', values='Overall F1').reset_index()
    
    # Add an 'All' column (average across all CVs)
    model_averages = df.groupby('Model')['Overall F1'].mean().reset_index()
    
    # Merge to ensure all models are included
    pivot = pd.merge(pivot, model_averages, on='Model', how='outer')
    pivot.rename(columns={'Overall F1': 'All'}, inplace=True)
    
    # Fill NaN values with 0
    pivot = pivot.fillna(0)
    
    # Print the results
    print("\n--- MODEL PERFORMANCE BY CV TYPE ---")
    print("Average F1 scores by model and CV type:")
    print(pivot.to_string(index=False))
    
    # Calculate field-specific performance
    field_columns = [col for col in df.columns if col.endswith(' F1') and col != 'Overall F1']
    
    # Placeholder for field summary data
    field_summary = []
    
    if field_columns:
        print("\n--- FIELD PERFORMANCE BY CV TYPE ---")
        
        for field_col in field_columns:
            field_name = field_col.replace(' F1', '')
            print(f"\nPerformance for {field_name} field:")
            
            # Group by Model and CV Type for this field
            field_grouped = df.groupby(['Model', 'CV Type'])[field_col].mean().reset_index()
            
            # Reshape to have Model as rows and CV Type as columns
            field_pivot = field_grouped.pivot(index='Model', columns='CV Type', values=field_col).reset_index()
            
            # Add an 'All' column
            field_all_avg = df.groupby('Model')[field_col].mean().reset_index()
            
            # Merge to ensure all models are included
            field_pivot = pd.merge(field_pivot, field_all_avg, on='Model', how='outer')
            field_pivot.rename(columns={field_col: 'All'}, inplace=True)
            
            # Fill NaN values with 0
            field_pivot = field_pivot.fillna(0)
            
            print(field_pivot.to_string(index=False))
            
            # Add to summary
            for _, row in field_pivot.iterrows():
                field_summary.append({
                    'Field': field_name,
                    'Model': row['Model'],
                    'Scanned': row.get('Scanned', 0),
                    'Text-based': row.get('Text-based', 0),
                    'All': row['All']
                })
        
        # Create and print overall field performance
        field_summary_df = pd.DataFrame(field_summary)
        
        print("\n--- OVERALL FIELD PERFORMANCE SUMMARY ---")
        print("Average F1 scores by field, model, and CV type:")
        
        # Make the table more readable
        for model in MODELS:
            model_fields = field_summary_df[field_summary_df['Model'] == model]
            
            print(f"\nModel: {model}")
            print(model_fields[['Field', 'Scanned', 'Text-based', 'All']].to_string(index=False))
    
    # Print overall conclusions
    print("\n--- SUMMARY CONCLUSIONS ---")
    
    # Find best model for each CV type
    best_model_scanned = pivot.loc[pivot['Scanned'].idxmax()]['Model'] if 'Scanned' in pivot.columns else "N/A"
    best_model_text = pivot.loc[pivot['Text-based'].idxmax()]['Model'] if 'Text-based' in pivot.columns else "N/A"
    best_model_overall = pivot.loc[pivot['All'].idxmax()]['Model']
    
    if 'Scanned' in pivot.columns:
        print(f"Best model for Scanned CVs: {best_model_scanned} (F1 score: {pivot.loc[pivot['Scanned'].idxmax()]['Scanned']:.4f})")
    if 'Text-based' in pivot.columns:
        print(f"Best model for Text-based CVs: {best_model_text} (F1 score: {pivot.loc[pivot['Text-based'].idxmax()]['Text-based']:.4f})")
    print(f"Best model overall: {best_model_overall} (F1 score: {pivot.loc[pivot['All'].idxmax()]['All']:.4f})")
    
    # Create visualization charts
    field_summary_df = pd.DataFrame(field_summary) if field_columns else None
    if field_summary_df is not None:
        create_charts(pivot, field_summary_df)
    
    # Save the results to Excel
    output_path = os.path.join(AGGREGATED_DIR, "cv_type_analysis.xlsx")
    
    # Create workbook
    with pd.ExcelWriter(output_path) as writer:
        # Overall performance
        pivot.to_excel(writer, sheet_name='Overall Performance', index=False)
        
        # Field performance
        if field_columns:
            field_summary_df.to_excel(writer, sheet_name='Field Performance', index=False)
        
        # Full data with categories
        df.to_excel(writer, sheet_name='Full Data', index=False)
    
    print(f"\nResults saved to {output_path}")

if __name__ == "__main__":
    main() 