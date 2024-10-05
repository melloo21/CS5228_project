import re

def extract_make(title, compiled_regex):
    """
    Extract the car make from the title using the compiled regex.
    :param title: str - The title from which to extract the make
    :param compiled_regex: regex pattern - Precompiled regex pattern
    :return: str - Extracted make or empty string if no match
    """
    title = title.lower()
    matches = compiled_regex.findall(title)
    return matches[0] if matches else None

def compile_make_pattern(make_list):
    """
    Compile a regex pattern from the list of makes.
    :param make_list: list - List of car makes
    :return: regex pattern - Compiled regex pattern to find car makes
    """
    # Combine patterns into a single regex
    combined_pattern = r'\b(' + '|'.join(make_list) + r')\b'
    # Precompile the regex
    return re.compile(combined_pattern)

def apply_make_extraction(df, compiled_regex):
    """
    Apply the extraction function to the 'title' column of the dataframe.
    :param df: DataFrame - The dataframe containing the 'title' column
    :param compiled_regex: regex pattern - Precompiled regex pattern
    :return: DataFrame - The dataframe with an 'extracted_make' column
    """
    df['extracted_make'] = df['title'].apply(lambda x: extract_make(x, compiled_regex))
    return df