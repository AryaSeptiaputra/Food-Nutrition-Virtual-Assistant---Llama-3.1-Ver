import pandas as pd

def _clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates()
    df = df.dropna(how="all")
    return df

def _normalize_data(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [col.strip().lower() for col in df.columns]
    return df

def _get_magnitude(unit: str) -> str:

    unit = str(unit).lower()
    mass_units = ['g', 'mg', 'mcg', 'µg']
    energy_units = ['kal', 'kcal', 'kj']
        
    if unit in mass_units:
        return 'Massa'
    elif unit in energy_units:
        return 'Energi'
    else:
        return 'Lainnya'
    
def _normalize_value(df: pd.DataFrame) -> pd.DataFrame:
    target_col = 'value (per 100gr)'

    # 1. Ekstrak Angka dan Satuan
    regex = r"([0-9.,]+)\s*([a-zA-Zµ]*)"
    extracted_data = df[target_col].astype(str).str.extract(regex)

    df['nilai'] = extracted_data[0]
    df['satuan'] = extracted_data[1].str.lower().str.strip().fillna('')

    cleaned_nilai = df['nilai'].str.replace(',', '', regex=False)
    
    # Konversi ke numerik
    df['nilai'] = pd.to_numeric(cleaned_nilai, errors='coerce')

    df['tipe_besaran'] = df['satuan'].apply(_get_magnitude)
        
    return df

def preprocess(df: pd.DataFrame, is_nutrition_data: bool = False) -> pd.DataFrame:
    df = _clean_data(df)
    df = _normalize_data(df)

    if is_nutrition_data:
            if 'value (per 100gr)' in df.columns:
                df = _normalize_value(df)
            else:
                raise ValueError("Kolom 'value (per 100gr)' tidak ditemukan untuk data nutrisi.")

    return df