# Basketball Data Analysis (Adaptable for Any Player)

A Python-based data analysis and machine learning project focused on analyzing player performance, using Tyrese Haliburton's 2023/2024 season data as an example. This program can be adapted for any basketball player by following the same steps.

```bash
# Prerequisites

# Ensure you have Python and the necessary libraries installed
pip install scikit-learn easygui pandas matplotlib

# Data Collection

# Download Data:
# To analyze a player's performance, you need to collect two key datasets:
# 1. Full Season/Playoffs Data for the player (Tyrese Haliburton used here) from Basketballreference.com.
# 2. Historical Lines for the player from Bettingpros.com.

# For the second dataset, you'll need to scrape the table using the Table Capture extension:
# Steps to scrape:
# 1. Navigate to Extensions > Tables
# 2. Choose the largest table (around 85x15 in this case).
# 3. Press the blue sheet with the green plus icon to copy the data.
# 4. Paste the data into the top-left cell of an Excel sheet.
# 5. Save the file for later use.

# Data Cleaning

# Follow these steps to clean the data properly:
# 1. Delete rows with DNP/DND/Missing data fields: Select entire rows, right-click, and delete them to ensure no NaN values remain.
# 2. Sort Data by Date:
#    - Ctrl + A to select all data
#    - Right-click > Sort > Custom Sort
#    - Ensure the "My data has headers" option is checked
#    - Sort by Date (newest to oldest)
# 3. Merge Scraped Data:
#    - Copy the Prop Line column from the scraped table.
#    - Paste it into the right side of the combined season CSV file. The last three columns should be GmSc, +/- , and Line.
# 4. Delete rows where 'NL' is listed, and ensure the column naming is consistent:
#    - Change column 'F' to LOC
#    - Change column 'H' to W/L
#    - Change column 'AE' from Prop Line to Line

# Move the CSV files into a folder (for example, the 'data' folder in the main project directory). The program will prompt you to select the correct file.

# Usage

# 1. Run the Program:
#    Execute main.py (or .exe if compiled from source).

# 2. Follow Prompts:
#    - Option 1: Visualizations – view scatter plots and the processed pandas data table.
#    - Option 2: Accuracy/Classification Report – shows the accuracy report for the trained model.
#    - Option 3: Re-train the model – retrain the model after tweaking any settings.
#    - Option 4: Enter custom query – input custom data to predict and classify a future match. Example queries are provided for datasets like t_haliburton_23-24_regszn.csv.
#    - Option 5: Settings – view or tweak settings for model training.
#    - Option 6: Label Mappings – see the encoded labels that the preprocessing algorithm generated (team IDs, W/L, and Home/Away mapping).
