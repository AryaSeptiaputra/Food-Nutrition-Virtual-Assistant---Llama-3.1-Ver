import pandas as pd

def _create_nutrition_dict(group):
    """Convert a nutrient DataFrame group into a nested nutrition dictionary.
    
    Transforms each row into {nutrient_name: {nilai: value, satuan: unit}} format.
    
    Args:
        group: DataFrame group indexed by nutrient/ingredient
    
    Returns:
        Dictionary mapping nutrient names to their value and unit
    """
    return {
        row['nutrient/ingridient']: {
            "nilai": row["nilai"],
            "satuan": row["satuan"]
        }
        for _, row in group.iterrows()
    }

def merge_data(df_food_details: pd.DataFrame, df_food_nutritions: pd.DataFrame) -> list:
    """Merge food details with nutrition data and standardize schema.
    
    Joins food details with nutrition information, renames columns to English,
    adds standard serving size (100g), and returns as list of dictionaries.
    
    Args:
        df_food_details: DataFrame with food metadata (code, name, group, type)
        df_food_nutritions: DataFrame with nutrient data (code, nutrient, value, unit)
    
    Returns:
        List of dictionaries with merged food and nutrition data
    """
    
    nutrition_map = (
        df_food_nutritions.groupby('code')
                          .apply(_create_nutrition_dict, include_groups=False)
                          .rename('nutrisi')
    )

    merged_df = df_food_details.set_index('code').join(nutrition_map)
    
    merged_df['nutrisi'] = merged_df['nutrisi'].apply(lambda d: d if isinstance(d, dict) else {})

    merged_df["berat"] = "100"
    merged_df["satuan"] = "gram"
    
    merged_df = merged_df.reset_index().rename(columns={
        "code": "kode",
        "name": "nama",
        "group": "kelompok",
        "type": "tipe"
    })
    
    final_columns = [
        "kode", "nama", "berat", "satuan", 
        "kelompok", "tipe", "nutrisi"
    ]
    final_df = merged_df[final_columns]

    return final_df.to_dict('records')