import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from helpers import renaming_education_level, percentile
from helpers import records_to_consider, convert_OrgSize, convert_YearsCodePro

import warnings
warnings.filterwarnings('ignore')


df = pd.read_csv("../stack-overflow-developer-survey-results-2023/survey_results_public.csv")

def survey_responses_count(df):
    # Considering only countries having atleast 1000 data points.
    countries_shortlisted = records_to_consider(df['Country'].value_counts(), threshold=1000)

    # Finalizing data points only from these shortlisted countries.
    df = df[df['Country'].isin(countries_shortlisted)]

    # Survey Responses count according to the country
    sorted_df = df['Country'].value_counts().reset_index().sort_values(by='count', ascending=False)

    return sorted_df


def preprocess_till_Ed_Level(df):
    # Considering only countries having atleast 1000 data points.
    countries_shortlisted = records_to_consider(df['Country'].value_counts(), threshold=1000)

    # Finalizing data points only from these shortlisted countries.
    df = df[df['Country'].isin(countries_shortlisted)]

    # Selecting the below columns
    required_cols = ['MainBranch', 'Age','Employment','RemoteWork','EdLevel',
                    'YearsCodePro', 'DevType','Country','CompTotal','ConvertedCompYearly',
                    'WorkExp','Industry','Currency','OrgSize','ICorPM','LanguageHaveWorkedWith','DatabaseHaveWorkedWith']

    df_subset = df[required_cols]

    # Dropping all the NaN entries
    df_final = df_subset.dropna()

    # Formatting the values into a concise field and appropriate data type
    df_final['OrgSize'] = df_final['OrgSize'].apply(convert_OrgSize)
    df_final['YearsCodePro'] = df_final['YearsCodePro'].apply(convert_YearsCodePro)

    # Rename Education levels for clarity and brevity.
    df_final.loc[:,'EdLevel'] = df_final['EdLevel'].map(renaming_education_level)

    return df_final

def median_compensation(df):
    df_final = preprocess_till_Ed_Level(df)

    salary_stats_df = df_final.groupby('Country')['ConvertedCompYearly'].agg({'mean','median','min','max',
                                                                                        percentile(0.25), percentile(0.75),
                                                                                        percentile(0.90), percentile(0.99)})
    salary_stats_df = salary_stats_df.sort_values(by = 'median',ascending = False)
    salary_stats_df.rename(index={'United Kingdom of Great Britain and Northern Ireland': 'United Kingdom'},inplace=True)

    return salary_stats_df

def process_usa_data(df):
    df_final = preprocess_till_Ed_Level(df)

    df_usa = df_final[df_final['Country'] == 'United States of America']

    mismatched_records = df_usa[df_usa['CompTotal'] != df_usa['ConvertedCompYearly']]

    df_usa = df_usa[~df_usa.index.isin(mismatched_records.index)]

    return df_usa

def languages_worked_with(df):
    df_final = preprocess_till_Ed_Level(df)

    df_final['LanguageHaveWorkedWith'] = df_final['LanguageHaveWorkedWith'].str.split(';')

    return df_final

def databases_worked_with(df):
    df_final = preprocess_till_Ed_Level(df)

    df_final['DatabaseHaveWorkedWith'] = df_final['DatabaseHaveWorkedWith'].str.split(';')

    return df_final

def dev_type(df):
    df_usa = process_usa_data(df)

    median_salary_df = df_usa.groupby('DevType')['ConvertedCompYearly'].median().reset_index().sort_values(
                                                by='ConvertedCompYearly', ascending=False).reset_index(drop=True)
    return median_salary_df

def industry_salaries(df):
    df_usa = process_usa_data(df)

    median_salary_df = df_usa.groupby('Industry')['ConvertedCompYearly'].median().reset_index().sort_values(
                                                by='ConvertedCompYearly', ascending=False).reset_index(drop=True)
    return median_salary_df


def years_WorkExp(df):
    df_usa = process_usa_data(df)

    salary_stats = df_usa.groupby('WorkExp')['ConvertedCompYearly'].agg(['median', 'std']).reset_index()

    return salary_stats

def median_salary_Ed_Level(df):
    df_usa = process_usa_data(df)

    median_salary_by_edlevel = df_usa.groupby('EdLevel')['ConvertedCompYearly'].median().reset_index()

    # Sort the results in descending order of median salary
    median_salary_by_edlevel_sorted = median_salary_by_edlevel.sort_values(by='ConvertedCompYearly', ascending=False)

    return median_salary_by_edlevel_sorted