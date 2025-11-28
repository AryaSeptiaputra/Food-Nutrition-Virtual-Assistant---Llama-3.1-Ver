import pandas as pd

def load_data(file_path) -> pd.DataFrame:
    """Load CSV data from file with flexible separator detection.
    
    Attempts to load CSV with pipe separator first (|), then falls back to default CSV parsing.
    This allows handling both pipe-delimited and comma-delimited formats.
    
    Args:
        file_path: Path to the CSV file to load
    
    Returns:
        DataFrame containing the loaded CSV data
    
    Raises:
        Exception: If file cannot be read with either separator format
    """
    try:
        return pd.read_csv(file_path, sep = "|")
    except Exception:
        return pd.read_csv(file_path)