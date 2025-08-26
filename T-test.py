#
# This script performs an ANOVA test on the 'participant_info.csv' file and exports the results to an Excel spreadsheet.
#
# IMPORTANT: Before running this code, you MUST install the required libraries by running the following commands in your terminal:
# pip install pandas
# pip install scipy
# pip install openpyxl
#
# You MUST also change the file_path variable to the exact location of your CSV file.
# Example: file_path = r'C:\Users\YourUser\Documents\DREAMT\DREAMT\participant_info.csv'
#

import pandas as pd
from scipy import stats

def perform_anova_and_export_to_excel(file_path, output_excel_path):
    """
    Performs a one-way ANOVA test on BMI data across AHI rating groups
    and exports the results to a new Excel file.

    Args:
        file_path (str): The full path to the input CSV file ('participant_info.csv').
        output_excel_path (str): The full path to the output Excel file.
    """
    try:
        # Step 1: Load the data from the specified CSV file
        df = pd.read_csv(file_path)

        # Step 2: Clean the column names and check for required columns
        df.columns = df.columns.str.strip()
        if 'AHI' not in df.columns or 'BMI' not in df.columns:
            print("Error: The CSV file does not contain the required 'AHI' or 'BMI' columns.")
            return

        # Step 3: Convert AHI and BMI to numeric data types and handle missing values
        df['AHI'] = pd.to_numeric(df['AHI'], errors='coerce')
        df['BMI'] = pd.to_numeric(df['BMI'], errors='coerce')
        df.dropna(subset=['AHI', 'BMI'], inplace=True)

        # Step 4: Categorize participants based on AHI ratings
        bins = [-1, 5, 15, 30, float('inf')]
        labels = ['Normal', 'Mild', 'Moderate', 'Severe']
        df['AHI_Rating'] = pd.cut(df['AHI'], bins=bins, labels=labels, right=False)

        # Check if all groups have data
        if df['AHI_Rating'].nunique() < len(labels):
            print("Warning: Some AHI rating groups are empty. ANOVA may not be meaningful.")
        
        # Step 5: Extract BMI data for each group for the ANOVA test
        normal_group = df[df['AHI_Rating'] == 'Normal']['BMI']
        mild_group = df[df['AHI_Rating'] == 'Mild']['BMI']
        moderate_group = df[df['AHI_Rating'] == 'Moderate']['BMI']
        severe_group = df[df['AHI_Rating'] == 'Severe']['BMI']

        # Step 6: Perform the one-way ANOVA test
        f_statistic, p_value = stats.f_oneway(
            normal_group, mild_group, moderate_group, severe_group
        )

        # Step 7: Create a DataFrame to hold the ANOVA results
        results_df = pd.DataFrame({
            'Metric': ['F-statistic', 'P-value', 'Significance Level'],
            'Value': [f_statistic, p_value, 0.05]
        })

        # Step 8: Create a separate DataFrame for group summary statistics
        summary_df = df.groupby('AHI_Rating')['BMI'].agg(['count', 'mean', 'std']).reset_index()
        summary_df.rename(columns={
            'AHI_Rating': 'AHI Group',
            'count': 'Number of Participants',
            'mean': 'Mean BMI',
            'std': 'Standard Deviation of BMI'
        }, inplace=True)

        # Step 9: Write the results to an Excel file with multiple sheets
        with pd.ExcelWriter(output_excel_path, engine='openpyxl') as writer:
            results_df.to_excel(writer, sheet_name='ANOVA Results', index=False)
            summary_df.to_excel(writer, sheet_name='Group Summary', index=False)
            
        print(f"✅ Analysis complete. Results have been saved to '{output_excel_path}'")

    except FileNotFoundError:
        print(f"❌ Error: The file was not found at the specified path: {file_path}. Please check the path and try again.")
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")

# IMPORTANT: You must change the paths below before running this script.
# Replace with the actual path to your input CSV file
input_file_path = r'E:\Muttakee\Polysomnography\DREAMT\DREAMT\participant_info.csv'
# Replace with your desired path and name for the output Excel file
output_excel_path = r'E:\Muttakee\Polysomnography\DREAMT\DREAMT\ANOVA_Results.xlsx'

# Call the function to run the analysis and export the Excel file
perform_anova_and_export_to_excel(input_file_path, output_excel_path)
