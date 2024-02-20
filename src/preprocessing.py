# Import necessary libraries for data manipulation and visualization
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from helpers import records_to_consider, convert_YearsCodePro, convert_OrgSize

# Suppress any warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')

# Load the survey dataset into a pandas DataFrame
df = pd.read_csv("../survey_results_public.csv")

# Filter countries with at least 1000 data points using a custom helper function
countries_shortlisted = records_to_consider(df['Country'].value_counts())

# Keep only data from shortlisted countries
df = df[df['Country'].isin(countries_shortlisted)]

# Select the below columns for analysis
required_cols = ['MainBranch', 'Age','Employment','RemoteWork','EdLevel','YearsCode',
                 'YearsCodePro', 'DevType','Country','CompTotal','ConvertedCompYearly',
                 'WorkExp','Industry','Currency','OrgSize','ICorPM']

# Create a subset DataFrame with the selected columns
df_subset = df[required_cols]

# Drop all rows with any NaN values to ensure data quality
df_final = df_subset.dropna()

# Apply custom functions to format 'OrgSize' and 'YearsCodePro' columns for consistency
df_final['OrgSize'] = df_final['OrgSize'].apply(convert_OrgSize)
df_final['YearsCodePro'] = df_final['YearsCodePro'].apply(convert_YearsCodePro)

# Rename Education levels for clarity and brevity.
renaming_education_level = {"Bachelor's degree (B.A., B.S., B.Eng., etc.)" : 'Bachelors',
       "Master's degree (M.A., M.S., M.Eng., MBA, etc.)" : 'Masters',
       'Some college/university study without earning a degree' : 'Some College',
       'Professional degree (JD, MD, Ph.D, Ed.D, etc.)' : 'PhD, Postdoc',
       'Secondary school (e.g. American high school, German Realschule or Gymnasium, etc.)' : 'Secondary School',
       'Associate degree (A.A., A.S., etc.)' : 'Associate degree',
       'Primary/elementary school' : 'Primary School',
       'Something else' : 'Something else'
       }
df_final.loc[:,'EdLevel'] = df_final['EdLevel'].map(renaming_education_level)

# Filter the dataset to include only respondents from the United States
df_usa = df_final[df_final['Country'] == 'United States of America']

# Identify and remove any records with discrepancies between 'CompTotal' & 'ConvertedCompYearly'
mismatched_records = df_usa[df_usa['CompTotal'] != df_usa['ConvertedCompYearly']]
df_usa = df_usa[~df_usa.index.isin(mismatched_records.index)]

# Copy the cleaned USA dataset for further operations
df_oa = df_usa.copy()

# Exclude specific categories from 'MainBranch', 'Age', 'EdLevel', and 'OrgSize' for more focused analysis
df_oa = df_oa[~df_oa['MainBranch'].isin(['I am not primarily a developer, but I write code sometimes as part of my work/studies'])]
df_oa = df_oa[~df_oa['Age'].isin(['Prefer not to say', 'Under 18 years old'])]
df_oa = df_oa[~df_oa['EdLevel'].isin(['Something else'])]
df_oa = df_oa[~df_oa['OrgSize'].isin(["I don't know", "Frelancer/Sole Proprietor"])]

# Filter 'DevType' and 'Industry' to include only those with at least 50 data points
df_oa = df_oa[df_oa['DevType'].map(df_oa['DevType'].value_counts()) >= 50]
df_oa = df_oa[df_oa['Industry'].map(df_oa['Industry'].value_counts()) >= 50]

# Shortlist employment types based on a custom threshold using a helper function
employment_shortlisted = records_to_consider(df_oa['Employment'].value_counts(), threshold=10)
df_oa = df_oa[df_oa['Employment'].isin(employment_shortlisted)]

# Filter records to include only those with a yearly compensation between $40,000 and $300,000
df_oa = df_oa[(df_oa['ConvertedCompYearly'] >= 40000) & (df_oa['ConvertedCompYearly'] <= 300000)]

# Define salary brackets based on quantiles and label them accordingly
quantiles = [0, 0.2, 0.4, 0.6, 0.8, 1]
bin_labels = ['Low', 'Low-Mid', 'Mid', 'Mid-High', 'High']
df_oa['SalaryBracket'] = pd.qcut(df_oa['ConvertedCompYearly'], q=quantiles, labels=bin_labels)


bin_ranges = {label: (df_oa.loc[df_oa['SalaryBracket'] == label, 'ConvertedCompYearly'].min(), \
                       df_oa.loc[df_oa['SalaryBracket'] == label, 'ConvertedCompYearly'].max()) for label in bin_labels}
# Reference
salary_ranges = {
    'Low' : '40k - 105k',
    'Low-Mid' : '105.5k - 135k',
    'Mid' : '135.5k - 162k',
    'Mid-High' : '162.5k - 200k',
    'High' : '200.4k - 300k',
}

clean_df = df_oa.copy()

# Drop irrelevant columns for the final clean dataset
cols_to_drop = ['MainBranch','Country','CompTotal','WorkExp','Currency','YearsCode']
clean_df = clean_df.drop(cols_to_drop,axis = 1).reset_index(drop = True)

# Convert categorical columns to 'category' dtype for efficient storage and processing
categorical_columns = ['Age', 'RemoteWork', 'EdLevel', 'DevType', 'Industry', 'OrgSize', 'ICorPM','Employment']
for col in categorical_columns:
    clean_df[col] = clean_df[col].astype('category')

# Save the cleaned and processed DataFrame to a new CSV file
clean_df.to_csv("../stack-overflow-developer-survey-results-2023/survey_results_clean_usa.csv",index=False)
