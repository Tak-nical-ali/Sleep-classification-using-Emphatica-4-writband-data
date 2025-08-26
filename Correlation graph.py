#
# This script loads data, performs a correlation analysis, and generates a
# scatter plot to visualize the relationship between BMI and AHI.
#
# IMPORTANT: Before running this code, you MUST install the required libraries
# by running the following commands in your terminal:
# pip install pandas
# pip install matplotlib
# pip install scipy
#
# You MUST also change the file_path variable to the exact location of your CSV file.
# Example: file_path = r'C:\Users\YourUser\Documents\DREAMT\DREAMT\participant_info.csv'
#

import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

def plot_correlation(file_path):
  
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
        
        # Step 4: Calculate the Pearson correlation for display on the plot
        correlation_coefficient, p_value = stats.pearsonr(df['AHI'], df['BMI'])

        # Step 5: Create the scatter plot
        plt.figure(figsize=(10, 6)) # Set the figure size for better readability
        plt.scatter(df['BMI'], df['AHI'], alpha=0.6, color='b')

        # Step 6: Add a line of best fit (linear regression line)
        # np.polyfit() is used to find the coefficients of the best-fit line
        # The line is then plotted over the scatter plot
        m, b = stats.linregress(df['BMI'], df['AHI'])[:2]
        plt.plot(df['BMI'], m * df['BMI'] + b, color='red', linestyle='--', label='Line of Best Fit')

        # Step 7: Add labels, title, and grid to the plot
        plt.title(f'BMI vs. AHI Correlation (r = {correlation_coefficient:.2f}, p = {p_value:.4f})', fontsize=16)
        plt.xlabel('Body Mass Index (BMI)', fontsize=12)
        plt.ylabel('Apnea-Hypopnea Index (AHI)', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend()
        
        # Step 8: Display the plot
        plt.show()

        print("Plot generation complete. A new window with the graph should have appeared.")

    except FileNotFoundError:
        print(f" Error: The file was not found at the specified path: {file_path}. Please check the path and try again.")
    except Exception as e:
        print(f" An unexpected error occurred: {e}")

# IMPORTANT: You must change the path below before running this script.
# Replace with the actual path to your input CSV file
input_file_path = r'E:\Muttakee\Polysomnography\DREAMT\DREAMT\participant_info.csv'

# Call the function to run the analysis and generate the plot
plot_correlation(input_file_path)

