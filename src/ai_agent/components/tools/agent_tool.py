import os
import json
from langchain_core.tools import tool

from src.ai_agent.components.database.DBclient import ChromaDBClient
from src.ai_agent.components.retriever.knowledge_semantic import SemanticRetriever
from src.ai_agent.components.vision.inferance import VisionInference
from src.config.settings import Settings

settings = Settings()

_semantic_retriever = None
_vision_inference = None

def initialize_retriever(db_client: ChromaDBClient, collection_name: str, embedding_function):
    """Initialize the Semantic Retriever before using the tools."""
    global _semantic_retriever
    try:
        _semantic_retriever = SemanticRetriever(
            db_client=db_client, 
            collection_name=collection_name,
            embedding_function=embedding_function
        )
        print("✅ Semantic Retriever initialized successfully")
    except Exception as e:
        print(f"❌ Error initializing retriever: {e}")
        raise

def initialize_vision_inference(model_path: str, raw_dir: str, cache_dir: str, annotated_dir: str, temp_dir: str):
    """Initialize the Vision Inference before using the tools."""
    global _vision_inference
    try:
        _vision_inference = VisionInference(
            model_path=model_path,
            raw_dir=raw_dir,
            cache_dir=cache_dir,
            annotated_dir=annotated_dir,
            temp_dir=temp_dir
        )
        print("✅ Vision Inference initialized successfully")
    except Exception as e:
        print(f"❌ Error initializing vision inference: {e}")
        raise

@tool
def search_food_nutrition_info(query_text: str) -> str:
    """Search food nutrition information from the knowledge base."""
    global _semantic_retriever

    if _semantic_retriever is None:
        error_msg = "Semantic retriever not initialized"
        print(f"❌ {error_msg}")
        return json.dumps({"error": error_msg}, ensure_ascii=False)

    try:
        if not isinstance(query_text, str):
            return json.dumps({"error": "Input must be a text string, not an object."}, ensure_ascii=False)
        
        clean_text = query_text.strip()
        
        if not clean_text:
            return json.dumps({"error": "Empty query"}, ensure_ascii=False)

        results_list = _semantic_retriever.search(query_text=clean_text, n_results=5)
        
        if results_list:
            cleaned_results = []
            for metadata_dict in results_list:
                formatted_metadata = {}

                for k, v in metadata_dict.items():
                    if not k.startswith('nutrisi_') and k not in ['score', 'distance', 'similarity', 'kode', 'tipe', 'kelompok']:
                        formatted_metadata[k] = v

                nutrition_keys = [
                    k for k in metadata_dict.keys()
                    if k.startswith('nutrisi_') and not k.endswith('_unit')
                ]

                for key in nutrition_keys:
                    value = metadata_dict.get(key)
                    unit_key = f"{key}_unit"
                    unit = metadata_dict.get(unit_key, "")
                    display_key = key.replace("nutrisi_", "").replace("_", " ").title()
                    formatted_metadata[display_key] = f"{value} {unit}".strip()

                cleaned_results.append(formatted_metadata)
            
            response = {
                "query": clean_text,
                "results": cleaned_results,
                "count": len(cleaned_results)
            }
        else:
            response = {
                "query": clean_text,
                "error": f"No data found for: {clean_text}",
                "count": 0
            }

        return json.dumps(response, ensure_ascii=False, indent=2)

    except Exception as e:
        error_msg = f"Internal error: {str(e)}"
        print(f"❌ {error_msg}")
        return json.dumps({"error": error_msg}, ensure_ascii=False)

@tool
def detect_food_in_image(image_path: str) -> str:
    """Detect food type in an image using a YOLO model."""
    global _vision_inference

    if _vision_inference is None:
        return json.dumps({
            "error": "Vision inference not initialized"
        }, ensure_ascii=False)

    try:
        detection_result = _vision_inference.detected(image_path=image_path)
        return json.dumps(detection_result, ensure_ascii=False, indent=2)

    except Exception as e:
        print(f"Error in detect_food_in_image: {e}")
        return json.dumps({
            "error": f"Internal error: {str(e)}"
        }, ensure_ascii=False)