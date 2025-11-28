import pandas as pd

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names to lowercase with whitespace trimmed.
    
    Args:
        df: Input DataFrame
    
    Returns:
        DataFrame with normalized column names
    """
    df.columns = [col.strip().lower() for col in df.columns]
    return df

def _validate_columns(df: pd.DataFrame, required_columns: set) -> pd.DataFrame:
    """Validate that all required columns exist and return filtered DataFrame.
    
    Normalizes columns and checks for required column presence.
    Raises error if any required columns are missing.
    
    Args:
        df: Input DataFrame to validate
        required_columns: Set of column names that must be present
    
    Returns:
        DataFrame containing only the required columns in order
    
    Raises:
        ValueError: If any required columns are missing
    """
    df = _normalize_columns(df)
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df[list(required_columns)]

def validate_data(df: pd.DataFrame, dataset_name) -> pd.DataFrame:
    """Validate dataset schema based on dataset type.
    
    Applies different validation rules for 'details' and 'nutrition' datasets.
    Details require: code, name, group, type
    Nutrition require: code, nutrient/ingredient, value (per 100gr)
    
    Args:
        df: Input DataFrame to validate
        dataset_name: Type of dataset ('details' or 'nutrition')
    
    Returns:
        Validated DataFrame with required columns only
    
    Raises:
        ValueError: If dataset_name is unknown or required columns missing
    """
    if dataset_name == "details":
        required_columns = ["code", "name", "group", "type"]
    elif dataset_name == "nutrition":
        required_columns = ["code", "nutrient/ingridient", "value (per 100gr)"]
    else:
        raise ValueError(f"Unknown dataset name: '{dataset_name}'")
    
    return _validate_columns(df, required_columns)