#
# This script calculates the Pearson correlation coefficient and p-value
# between BMI and AHI and exports the results to an Excel spreadsheet.
#
# IMPORTANT: Before running this code, you MUST install the required libraries
# by running the following commands in your terminal:
# pip install pandas
# pip install scipy
# pip install openpyxl
#
# You MUST also change the file_path variable to the exact location of your CSV file.
# Example: input_file_path = r'C:\Users\YourUser\Documents\DREAMT\DREAMT\participant_info.csv'
#

import pandas as pd
from scipy import stats

def perform_correlation_and_export_to_excel(input_file_path, output_excel_path):
    """
    Calculates the Pearson correlation coefficient between AHI and BMI and exports
    the results, including a p-value, to a new Excel file.

    Args:
        input_file_path (str): The full path to the input CSV file ('participant_info.csv').
        output_excel_path (str): The full path to the output Excel file.
    """
    try:
        # Step 1: Load the data from the specified CSV file
        df = pd.read_csv(input_file_path)

        # Step 2: Clean the column names and check for required columns
        df.columns = df.columns.str.strip()
        if 'AHI' not in df.columns or 'BMI' not in df.columns:
            print("Error: The CSV file does not contain the required 'AHI' or 'BMI' columns.")
            return

        # Step 3: Convert AHI and BMI to numeric data types and handle missing values
        # The errors='coerce' argument will replace any non-numeric values with NaN
        df['AHI'] = pd.to_numeric(df['AHI'], errors='coerce')
        df['BMI'] = pd.to_numeric(df['BMI'], errors='coerce')
        df.dropna(subset=['AHI', 'BMI'], inplace=True)

        # Step 4: Perform the Pearson correlation test
        # pearsonr returns the correlation coefficient and the two-tailed p-value
        correlation_coefficient, p_value = stats.pearsonr(df['AHI'], df['BMI'])

        # Step 5: Create a DataFrame to hold the correlation results
        results_df = pd.DataFrame({
            'Metric': ['Correlation Coefficient', 'P-value', 'Sample Size'],
            'Value': [correlation_coefficient, p_value, len(df)]
        })

        # Step 6: Create a separate DataFrame to show a sample of the cleaned data
        sample_df = df[['AHI', 'BMI']].head(10)

        # Step 7: Write the results to an Excel file with multiple sheets
        with pd.ExcelWriter(output_excel_path, engine='openpyxl') as writer:
            results_df.to_excel(writer, sheet_name='Correlation Results', index=False)
            sample_df.to_excel(writer, sheet_name='Cleaned Data Sample', index=False)
            
        print(f"✅ Analysis complete. Results have been saved to '{output_excel_path}'")

    except FileNotFoundError:
        print(f"❌ Error: The file was not found at the specified path: {input_file_path}. Please check the path and try again.")
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")

# IMPORTANT: You must change the paths below before running this script.
# Replace with the actual path to your input CSV file
input_file_path = r'E:\Muttakee\Polysomnography\DREAMT\DREAMT\participant_info.csv'
# Replace with your desired path and name for the output Excel file
output_excel_path = r'E:\Muttakee\Polysomnography\DREAMT\DREAMT\Correlation_Results.xlsx'

# Call the function to run the analysis and export the Excel file
perform_correlation_and_export_to_excel(input_file_path, output_excel_path)
