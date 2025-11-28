def _build_synthetic_sentence(food_item: dict) -> str:
    """Generate a natural language description of a food item and its nutrition.
    
    Creates a descriptive sentence highlighting top nutrients and nutritional role.
    Falls back to basic description if nutrition data unavailable.
    
    Args:
        food_item: Dictionary with keys: nama, kelompok, tipe, berat, satuan, nutrisi
    
    Returns:
        Natural language description of the food and its nutritional content
    """
    name = food_item.get("nama", "")
    group = food_item.get("kelompok", "")
    food_type = food_item.get("tipe", "")
    weight = food_item.get("berat", "")
    unit = food_item.get("satuan", "")
    nutrition_data = food_item.get("nutrisi", {})

    valid_nutrients = {k: v for k, v in nutrition_data.items() if float(v.get("nilai", 0)) > 0}

    sorted_nutrients = sorted(
        valid_nutrients.items(), 
        key=lambda x: float(x[1].get("nilai", 0)), 
        reverse=True
    )

    if not sorted_nutrients:
        return f"{name} adalah makanan kelompok {group} dengan tipe {food_type}, namun data nutrisi tidak tersedia."

    top_nutrients = sorted_nutrients[:3]
    additional_nutrients = sorted_nutrients[3:7]

    top_str = ", ".join(
        f"{v['nilai']} {v['satuan']} {k.split('(')[0].strip().lower()}"
        for k, v in top_nutrients
    )
    additional_str = ", ".join(
        k.split('(')[0].strip().lower()
        for k, _ in additional_nutrients
    )

    role_hint = ""
    for k, v in top_nutrients:
        if "energi" in k.lower() or "kal" in v["satuan"].lower():
            role_hint = "energi"
            break 
        elif "protein" in k.lower():
            role_hint = "protein"
        elif "karbohidrat" in k.lower():
            role_hint = "karbohidrat"
        elif "lemak" in k.lower():
            role_hint = "lemak"
    
    if not role_hint:
        for k, v in sorted_nutrients:
            if "energi" in k.lower() or "kal" in v["satuan"].lower():
                role_hint = "energi"
                break
        if not role_hint:
            role_hint = "zat gizi"

    sentence = (
        f"{name} merupakan makanan dari kelompok {group} dan termasuk ke dalam tipe {food_type}. "
        f"Setiap {weight} {unit} {name.lower()} mengandung berbagai nutrisi penting dengan jumlah yang besar, "
        f"di antaranya {top_str}. "
    )

    if additional_str:
        sentence += (
            f"Selain itu, {name.lower()} juga memiliki kandungan nutrisi lain seperti {additional_str}, "
            f"yang turut mendukung nilai gizi totalnya. "
        )

    sentence += (
        f"Kandungan tersebut menjadikan {name.lower()} sebagai sumber {role_hint} "
        f"yang bermanfaat bagi tubuh."
    )

    return sentence

def generate_descriptions(knowledge_data: list[dict]) -> list[dict]:
    """Generate synthetic descriptions for all food items in the knowledge base.
    
    Iterates through knowledge data and adds 'deskripsi_sintetis' field to each item
    containing a natural language description generated from food metadata and nutrition.
    Gracefully handles errors without stopping processing.
    
    Args:
        knowledge_data: List of food dictionaries with nutrition information
    
    Returns:
        List of food dictionaries with 'deskripsi_sintetis' field added
    """
    if not isinstance(knowledge_data, list):
        return knowledge_data

    processed_count = 0
    try:
        for item in knowledge_data:
            if not isinstance(item, dict):
                continue
            
            description = _build_synthetic_sentence(item) 
            
            item["deskripsi_sintetis"] = description
            processed_count += 1

        return knowledge_data
    
    except Exception as e:
        ValueError(f"Gagal menghasilkan deskripsi sintetis: {e}")
        return knowledge_data