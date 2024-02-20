import pickle
import pandas as pd

renaming_education_level = {'Bachelor’s degree (B.A., B.S., B.Eng., etc.)' : 'Bachelors',
       'Master’s degree (M.A., M.S., M.Eng., MBA, etc.)' : 'Masters',
       'Some college/university study without earning a degree' : 'Some College',
       'Professional degree (JD, MD, Ph.D, Ed.D, etc.)' : 'PhD, Postdoc',
       'Secondary school (e.g. American high school, German Realschule or Gymnasium, etc.)' : 'Secondary School',
       'Associate degree (A.A., A.S., etc.)' : 'Associate degree',
       'Primary/elementary school' : 'Primary School',
       'Something else' : 'Something else'}

OrgSize_keys = ['Less than 20 employees', '20 to 99 employees','100 to 499 employees', '500 to 999 employees','1,000 to 4,999 employees',
  '5,000 to 9,999 employees','10,000 or more employees']

Industry_keys = ['Information Services, IT, Software Development, or other Technology','Other', 'Financial Services', 'Healthcare',
              'Manufacturing, Transportation, or Supply Chain', 'Retail and Consumer Services', 'Insurance', 'Higher Education','Advertising Services']

DevType_keys = ['Developer, full-stack', 'Developer, back-end', 'Developer, front-end','Developer, desktop or enterprise applications',
                'Developer, embedded applications or devices', 'Engineering manager', 'Developer, mobile', 'Engineer, data',
                'Cloud infrastructure engineer', 'DevOps specialist','Senior Executive (C-Suite, VP, etc.)',
                'Data scientist or machine learning specialist','Research & Development role',
                'Engineer, site reliability', 'Other (please specify):']

EdLevel_keys = ["Primary School", "Secondary School", "Some College", "Associate Degree",
                                                    "Bachelors", "Masters", "PhD, Postdoc"]

Age_keys = ["18-24 years old","25-34 years old", "35-44 years old", "45-54 years old", "55-64 years old","65 years or older"]

RemoteWork_keys = ["Remote", "In-person", "Hybrid (some remote, some in-person)"]

IcorPM_keys = ["Individual contributor", "People manager"]

Employment_keys = ["Employed, full-time", "Employed, full-time;Independent contractor, freelancer, or self-employed","Employed, part-time",
                    "Independent contractor, freelancer, or self-employed","Employed, full-time;Employed, part-time"]

salary_ranges = {
    'Low' : r'40,000 - 105,000',
    'Low-Mid' : r'105,500 - 135,000',
    'Mid' : r'135,500 - 162,000',
    'Mid-High' : r'162,500 - 200,000',
    'High' : r'200,400 - 300,000',
}

slider_ranges = {
    'Low': (40000, 105000),
    'Low-Mid': (105500, 135000),
    'Mid': (135500, 162000),
    'Mid-High': (162500, 200000),
    'High': (200400, 300000),
}


# Calculating various statistics (mean, median, min, max, percentiles) for yearly converted compensation grouped by country. 
def percentile(n):
    def percentile_(x):
        return x.quantile(n)
    percentile_.__name__ = 'percentile_{:02.0f}'.format(n*100)
    return percentile_

def records_to_consider(column_counts, threshold = 1000):
    """
    Filters entities with counts above a specified threshold.
    
    This function takes a pandas Series object (column_counts) where the index 
    represents unique entities (e.g., countries) and the values represent the 
    count of occurrences of each entity. It returns a list of entities where 
    their counts are greater than or equal to a specified threshold.
    
    Parameters:
    - column_counts (pd.Series): A pandas Series object with index as entities and values as counts.
    - threshold (int, optional): The count threshold for filtering entities. Default is 1000.
    
    Returns:
    - list: A list of entities that meet or exceed the specified threshold.
    """
    result = []
    column_list = zip(column_counts.index, column_counts.values)
    for entity, count in column_list:
        if count >= threshold:
            result.append(entity)
    return result

def convert_YearsCodePro(experience):
    """
    Converts a string representation of professional coding experience into a numeric value.
    
    This function handles specific string values representing coding experience and converts
    them into corresponding numeric values. It is designed to standardize the representation 
    of professional experience in years for analysis.
    
    Parameters:
    - experience (str): A string representing the number of years of professional coding experience.
    
    Returns:
    - float: A numeric value representing the years of professional coding experience. Special 
    cases 'More than 50 years' and 'Less than 1 year' are converted to 50 and 0.5, respectively.
    """
    if experience == 'More than 50 years':
        return 50
    elif experience == 'Less than 1 year':
        return 0.5
    else:
        return float(experience)


def convert_OrgSize(size):
    """
    Converts organization size categories into a more generalized category.
    
    This function simplifies the representation of organization size by grouping 
    certain categories together and renaming others for consistency and ease of analysis.
    
    Parameters:
    - size (str): A string representing the size of an organization.
    
    Returns:
    - str: A string representing the generalized category of organization size. 
    Specific sizes '2 to 9 employees' and '10 to 19 employees' are grouped into 
    'Less than 20 employees'. The 'Just me - I am a freelancer, sole proprietor, etc.' 
    category is renamed to 'Freelancer/Sole Proprietor'.
    """
    if size in ['2 to 9 employees', '10 to 19 employees']:
        return 'Less than 20 employees'
    if size == 'Just me - I am a freelancer, sole proprietor, etc.':
        return 'Frelancer/Sole Proprietor'
    return size


def perform_encoding(input_features):
    # Load the encoders
    with open('../saved_weights/encoders.pkl', 'rb') as file:
        encoders = pickle.load(file)

    # Load the mappings
    with open('../saved_weights/education_level_map.pkl', 'rb') as file:
        education_level_map = pickle.load(file)

    # Convert input features to DataFrame
    input_df = pd.DataFrame(input_features)
    
    # Apply LabelEncoders to categorical columns
    for column, encoder in encoders.items():
        input_df[column] = encoder.transform(input_df[column])
    
    # Map ordinal values directly
    input_df['EdLevel'] = input_df['EdLevel'].map(education_level_map)

    # Return processed DataFrame ready for model input
    desired_order = ['Age', 'Employment', 'RemoteWork', 'EdLevel', 'YearsCodePro', 'DevType', 'Industry', 'OrgSize', 'ICorPM']
    input_df = input_df[desired_order]

    return input_df