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

