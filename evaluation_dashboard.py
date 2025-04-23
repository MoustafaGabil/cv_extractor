import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import glob
import json
from PIL import Image
import io

# Set page configuration
st.set_page_config(
    page_title="CV Extraction Evaluation Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Define directories
REPORTS_DIR = "evaluation_reports"
AGGREGATED_DIR = os.path.join(REPORTS_DIR, "aggregated_results")
MODEL_REPORTS_DIR = os.path.join(REPORTS_DIR, "model_reports")
CV_REPORTS_DIR = os.path.join(REPORTS_DIR, "individual_cv_reports")
GROUND_TRUTH_DIR = "ground_truth"
EXTRACTED_DATA_DIR = "extracted_data"

# Define models and their display colors
MODELS = {
    "phi": "#ff9900",
    "llama3": "#0099ff",
    "mistral": "#00cc66"
}

# Function to automatically categorize CVs
def categorize_cvs(df):
    """
    Automatically categorize CVs as scanned or text-based based on naming convention.
    
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
    
    return {
        'scanned': scanned_cvs,
        'text_based': text_based_cvs
    }

# Function to get the latest Excel report
def get_latest_excel_report():
    """Get the most recent evaluation Excel report."""
    excel_files = glob.glob(os.path.join(AGGREGATED_DIR, "evaluation_results_*.xlsx"))
    if not excel_files:
        st.error("No evaluation reports found in the aggregated_results directory.")
        return None
    
    latest_file = max(excel_files, key=os.path.getmtime)
    return latest_file

# Function to load summary data
def load_summary_data():
    """Load and prepare summary data from Excel report."""
    excel_path = get_latest_excel_report()
    if not excel_path:
        return None, None, None
    
    # Read the summary data
    summary_df = pd.read_excel(excel_path, sheet_name='Summary')
    
    # Add CV Type column
    cv_categories = categorize_cvs(summary_df)
    summary_df['CV Type'] = summary_df['CV Name'].apply(
        lambda x: 'Scanned' if x in cv_categories['scanned'] else 'Text-based'
    )
    
    # Create overall performance data
    performance_pivot = summary_df.pivot_table(
        index='Model',
        columns='CV Type',
        values='Overall F1',
        aggfunc='mean'
    ).reset_index()
    
    # Add overall average
    all_avg = summary_df.groupby('Model')['Overall F1'].mean().reset_index()
    performance_pivot = pd.merge(performance_pivot, all_avg, on='Model', how='outer')
    performance_pivot.rename(columns={'Overall F1': 'All'}, inplace=True)
    
    # Create field performance data
    field_columns = [col for col in summary_df.columns if col.endswith(' F1') and col != 'Overall F1']
    
    field_data = []
    for field_col in field_columns:
        field_name = field_col.replace(' F1', '')
        
        # Get field performance by model and CV type
        field_pivot = summary_df.pivot_table(
            index='Model',
            columns='CV Type',
            values=field_col,
            aggfunc='mean'
        ).reset_index()
        
        # Add overall average
        field_all = summary_df.groupby('Model')[field_col].mean().reset_index()
        field_pivot = pd.merge(field_pivot, field_all, on='Model', how='outer')
        field_pivot.rename(columns={field_col: 'All'}, inplace=True)
        field_pivot['Field'] = field_name
        
        field_data.append(field_pivot)
    
    field_df = pd.concat(field_data, ignore_index=True) if field_data else None
    
    return summary_df, performance_pivot, field_df

def create_model_comparison_chart(pivot_df):
    """Create a bar chart comparing model performance."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Prepare data
    models = pivot_df['Model'].tolist()
    x = np.arange(len(models))
    width = 0.25
    
    # Get column categories
    categories = [col for col in pivot_df.columns if col != 'Model']
    
    # Create bars for each category
    for i, category in enumerate(categories):
        values = pivot_df[category].tolist()
        ax.bar(x + (i - len(categories)/2 + 0.5) * width, values, width, 
               label=category, color=sns.color_palette("husl", len(categories))[i])
    
    # Add labels and formatting
    ax.set_xlabel('Model')
    ax.set_ylabel('Average F1 Score')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Set y-axis limits
    ax.set_ylim(0, 1.0)
    
    plt.tight_layout()
    
    # Convert to image
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    
    return buf

def create_field_performance_chart(field_df, model=None):
    """Create a bar chart showing field performance for a specific model."""
    if model:
        model_fields = field_df[field_df['Model'] == model]
    else:
        # Group by field and calculate average across models
        model_fields = field_df.groupby('Field').mean().reset_index()
    
    fields = model_fields['Field'].unique()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Prepare data
    x = np.arange(len(fields))
    width = 0.25
    
    # Get column categories (excluding Model and Field)
    categories = [col for col in model_fields.columns if col not in ['Model', 'Field']]
    
    # Set up bars for each category
    for i, category in enumerate(categories):
        values = []
        for field in fields:
            field_data = model_fields[model_fields['Field'] == field]
            values.append(field_data[category].values[0] if not field_data.empty else 0)
        
        ax.bar(x + (i - len(categories)/2 + 0.5) * width, values, width, 
               label=category, color=sns.color_palette("husl", len(categories))[i])
    
    # Add labels and formatting
    ax.set_xlabel('Field')
    ax.set_ylabel('Average F1 Score')
    title = f'Field Performance - {model}' if model else 'Field Performance (All Models)'
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(fields, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Set y-axis limits
    ax.set_ylim(0, 1.0)
    
    plt.tight_layout()
    
    # Convert to image
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    
    return buf

def compare_extracted_data(cv_name):
    """Compare extracted data from different models against ground truth."""
    
    # Load ground truth data
    ground_truth_path = os.path.join(GROUND_TRUTH_DIR, f"{cv_name}.json")
    if not os.path.exists(ground_truth_path):
        st.error(f"Ground truth file not found for {cv_name}")
        return None
    
    with open(ground_truth_path, 'r', encoding='utf-8') as f:
        ground_truth = json.load(f)
    
    # Load extracted data for each model
    extracted_data = {}
    for model in MODELS.keys():
        model_data_path = os.path.join(EXTRACTED_DATA_DIR, model, f"{cv_name}.json")
        if os.path.exists(model_data_path):
            with open(model_data_path, 'r', encoding='utf-8') as f:
                try:
                    extracted_data[model] = json.load(f)
                except json.JSONDecodeError:
                    st.error(f"Error decoding JSON for {model} - {cv_name}")
                    extracted_data[model] = {"error": "Invalid JSON format"}
        else:
            extracted_data[model] = {"error": "Data not available"}
    
    return ground_truth, extracted_data

def display_dataset_info():
    """Display information about the dataset."""
    # Count ground truth files
    ground_truth_files = glob.glob(os.path.join(GROUND_TRUTH_DIR, "*.json"))
    
    # Get CV types
    cv_names = [os.path.splitext(os.path.basename(f))[0] for f in ground_truth_files]
    scanned_cvs = [cv for cv in cv_names if "_sc" in cv.lower()]
    text_based_cvs = [cv for cv in cv_names if cv not in scanned_cvs]
    
    st.subheader("Dataset Statistics")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total CVs", len(ground_truth_files))
    
    with col2:
        st.metric("Scanned CVs", len(scanned_cvs))
    
    with col3:
        st.metric("Text-based CVs", len(text_based_cvs))
    
    # Display CV list
    st.subheader("Available CVs")
    
    tab1, tab2 = st.tabs(["Scanned CVs", "Text-based CVs"])
    
    with tab1:
        if scanned_cvs:
            for cv in sorted(scanned_cvs):
                st.write(f"- {cv}")
        else:
            st.info("No scanned CVs found")
    
    with tab2:
        if text_based_cvs:
            for cv in sorted(text_based_cvs):
                st.write(f"- {cv}")
        else:
            st.info("No text-based CVs found")

def main():
    st.title("📊 CV Information Extraction Evaluation")
    
    # Load data
    summary_df, performance_pivot, field_df = load_summary_data()
    
    # Add a sidebar with navigation options
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", [
        "Overview", 
        "Model Comparison", 
        "Field Analysis", 
        "CV Comparison",
        "Dataset Info",
        "View Raw Data"
    ])
    
    if page == "Overview":
        st.header("Evaluation Overview")
        
        st.markdown("""
        This dashboard presents the results of evaluating different language models on their ability to
        extract structured information from CVs. The models compared are:
        
        - **Phi**: Microsoft's lightweight language model
        - **Llama3**: Meta's open source large language model
        - **Mistral**: Mistral AI's high-performance model
        
        The evaluation measures the extraction accuracy for various fields such as name, email, education, etc.
        across different types of CVs (scanned and text-based).
        """)
        
        if summary_df is not None:
            # Display overview statistics
            st.subheader("Key Performance Metrics")
            
            # Create columns for metrics
            col1, col2, col3 = st.columns(3)
            
            # Get best models
            if 'Scanned' in performance_pivot.columns:
                best_scanned = performance_pivot.loc[performance_pivot['Scanned'].idxmax()]
                col1.metric(
                    "Best Model for Scanned CVs", 
                    best_scanned['Model'], 
                    f"F1: {best_scanned['Scanned']:.4f}"
                )
            
            if 'Text-based' in performance_pivot.columns:
                best_text = performance_pivot.loc[performance_pivot['Text-based'].idxmax()]
                col2.metric(
                    "Best Model for Text-based CVs", 
                    best_text['Model'], 
                    f"F1: {best_text['Text-based']:.4f}"
                )
            
            best_overall = performance_pivot.loc[performance_pivot['All'].idxmax()]
            col3.metric(
                "Best Model Overall", 
                best_overall['Model'], 
                f"F1: {best_overall['All']:.4f}"
            )
            
            # Display model comparison chart
            st.subheader("Model Performance Overview")
            chart_buf = create_model_comparison_chart(performance_pivot)
            st.image(chart_buf, use_column_width=True)
            
            # Display field performance chart
            st.subheader("Field Performance Overview")
            field_chart_buf = create_field_performance_chart(field_df)
            st.image(field_chart_buf, use_column_width=True)
        else:
            st.warning("No evaluation data available. Please run the evaluation first.")
            st.info("You can run the evaluation by selecting option 2 in the main script.")
    
    elif page == "Model Comparison":
        st.header("Model Comparison")
        
        if summary_df is not None:
            # Display comparison metrics
            st.subheader("Performance by CV Type")
            st.dataframe(performance_pivot, use_container_width=True)
            
            # Display model comparison chart
            chart_buf = create_model_comparison_chart(performance_pivot)
            st.image(chart_buf, use_column_width=True)
            
            # Show best model for each field
            st.subheader("Best Model by Field")
            
            if field_df is not None:
                # Get best model for each field
                best_models = field_df.groupby('Field').apply(
                    lambda x: x.loc[x['All'].idxmax()]
                ).reset_index(drop=True)
                
                for _, row in best_models.iterrows():
                    field_name = row['Field']
                    model_name = row['Model']
                    score = row['All']
                    
                    st.write(f"**{field_name}**: {model_name} (F1: {score:.4f})")
        else:
            st.warning("No evaluation data available. Please run the evaluation first.")
    
    elif page == "Field Analysis":
        st.header("Field Performance Analysis")
        
        if field_df is not None:
            # Add model selector
            model = st.selectbox(
                "Select Model",
                ["All"] + list(MODELS.keys())
            )
            
            # Display field performance
            if model == "All":
                # Show overall field performance
                chart_buf = create_field_performance_chart(field_df)
                st.image(chart_buf, use_column_width=True)
                
                # Display field performance table
                field_avg = field_df.groupby('Field').mean().reset_index()
                st.dataframe(field_avg, use_container_width=True)
            else:
                # Show model-specific field performance
                chart_buf = create_field_performance_chart(field_df, model)
                st.image(chart_buf, use_column_width=True)
                
                # Display field performance table
                model_field_data = field_df[field_df['Model'] == model]
                st.dataframe(model_field_data, use_container_width=True)
        else:
            st.warning("No field performance data available. Please run the evaluation first.")
    
    elif page == "CV Comparison":
        st.header("Compare Extracted Data")
        
        # Get list of CVs
        ground_truth_files = glob.glob(os.path.join(GROUND_TRUTH_DIR, "*.json"))
        cv_names = [os.path.splitext(os.path.basename(f))[0] for f in ground_truth_files]
        
        if cv_names:
            # Add CV selector
            selected_cv = st.selectbox("Select CV", sorted(cv_names))
            
            # Compare extracted data
            comparison_data = compare_extracted_data(selected_cv)
            
            if comparison_data:
                ground_truth, extracted_data = comparison_data
                
                # Display ground truth and extracted data
                st.subheader("Ground Truth vs. Extracted Data")
                
                # Create tabs for each field
                fields = ground_truth.keys()
                field_tabs = st.tabs(fields)
                
                for i, field in enumerate(fields):
                    with field_tabs[i]:
                        # Display ground truth
                        st.subheader("Ground Truth")
                        if isinstance(ground_truth[field], (list, dict)):
                            st.json(ground_truth[field])
                        else:
                            st.write(ground_truth[field])
                        
                        # Display extracted data from each model
                        for model in MODELS.keys():
                            st.subheader(f"{model.capitalize()} Extraction")
                            if model in extracted_data:
                                if "error" in extracted_data[model]:
                                    st.error(extracted_data[model]["error"])
                                elif field in extracted_data[model]:
                                    if isinstance(extracted_data[model][field], (list, dict)):
                                        st.json(extracted_data[model][field])
                                    else:
                                        st.write(extracted_data[model][field])
                                else:
                                    st.info(f"Field '{field}' not found in {model} extraction")
                            else:
                                st.info(f"No data available for {model}")
        else:
            st.warning("No ground truth files found. Please add ground truth data first.")
    
    elif page == "Dataset Info":
        st.header("Dataset Information")
        display_dataset_info()
    
    elif page == "View Raw Data":
        st.header("Raw Evaluation Data")
        
        if summary_df is not None:
            st.dataframe(summary_df, use_container_width=True)
            
            # Add download button
            csv = summary_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Summary Data as CSV",
                data=csv,
                file_name="cv_extraction_summary.csv",
                mime="text/csv",
            )
        else:
            st.warning("No evaluation data available. Please run the evaluation first.")
    
    # Add footer
    st.sidebar.markdown("---")
    st.sidebar.info(
        "This dashboard visualizes the evaluation results of different language models "
        "on their ability to extract structured information from CVs."
    )
    st.sidebar.markdown("Generated: " + pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"))

if __name__ == "__main__":
    main() 