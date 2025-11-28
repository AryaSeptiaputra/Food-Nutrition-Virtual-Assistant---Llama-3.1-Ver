import os
import sys
import json
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

sys.path.append(os.getcwd())

def run_real_db_test():
    """Test semantic retriever with real ChromaDB database and food knowledge base.
    
    Performs end-to-end testing of the semantic search functionality:
    1. Loads configuration from settings
    2. Connects to ChromaDB with embedding function
    3. Initializes semantic retriever tool
    4. Accepts user query input or uses default
    5. Executes search_food_nutrition_info tool
    6. Validates and displays results
    7. Checks data formatting (removes technical fields like score/distance)
    
    Handles all errors gracefully with informative messages.
    Supports user input or defaults to 'nasi' (rice) for testing.
    """
    print("Starting Real Database Test with Actual Data\n")

    try:
        from src.config.settings import Settings
        from src.ai_agent.components.database.DBclient import ChromaDBClient
        from src.ai_agent.components.tools.agent_tool import (
            initialize_retriever, 
            search_food_nutrition_info
        )
        
        settings = Settings()
        print("Settings loaded.")
        print(f"DB Path Relative: {settings.CHROMA_DB_PATH}")
        print(f"Collection: {settings.CHROMA_COLLECTION_NAME}")
        print(f"Embedding Model: {settings.EMBEDDING_TEXT_MODEL_NAME}")
        
    except ImportError as e:
        print("Import Error: Ensure this script runs from project root folder.")
        print(f"Detail: {e}")
        return
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return

    print("\nConnecting to ChromaDB...")
    try:
        chroma_path = os.path.join(settings.PROJECT_ROOT, settings.CHROMA_DB_PATH).replace("\\", "/")
        
        if not os.path.exists(chroma_path):
            print(f"WARNING: DB folder not found at: {chroma_path}")
        
        db_client = ChromaDBClient(db_path=chroma_path)
        
        embedding_function = SentenceTransformerEmbeddingFunction(
            model_name=settings.EMBEDDING_TEXT_MODEL_NAME
        )
    except Exception as e:
        print(f"Error connecting to DB or loading Embedding Model: {e}")
        return

    print("Initializing Retriever Tool...")
    try:
        initialize_retriever(
            db_client=db_client,
            collection_name=settings.CHROMA_COLLECTION_NAME,
            embedding_function=embedding_function
        )
        print("Semantic Retriever initialized successfully")
    except Exception as e:
        print(f"Error initializing tool: {e}")
        return

    print("-" * 50)
    query_keyword = input("Enter food name to search (e.g., rice): ").strip()
    
    if not query_keyword:
        query_keyword = "rice"
        print(f"   Using default: '{query_keyword}'")
    
    print(f"\nSearching for: '{query_keyword}'...")
    
    try:
        json_output = search_food_nutrition_info.invoke(query_keyword)
    except Exception as e:
        print(f"\nError invoking tool: {e}")
        return

    print("\nJSON Output from Tool:")
    print("-" * 50)
    print(json_output)
    print("-" * 50)

    try:
        data = json.loads(json_output)
        
        if "error" in data:
            print("\nTool returned ERROR message.")
            print(f"   Message: {data['error']}")
        else:
            result_count = data.get('count', 0)
            print(f"\nSuccess! Found {result_count} results.")
            
            if result_count > 0:
                first_result = data['results'][0]
                print("\nData Format Verification (Filtering):")
                
                has_score = 'score' in first_result
                has_distance = 'distance' in first_result
                has_similarity = 'similarity' in first_result
                
                if not any([has_score, has_distance, has_similarity]):
                    print("   [OK] Technical fields (score/distance) successfully removed.")
                else:
                    print("   [FAIL] Technical fields still present!")
                    
                print(f"\n   First Result Example: {first_result.get('nama_makanan', 'No Name')}")
                print(f"   Calories: {first_result.get('kalori', 'N/A')}")
                    
    except Exception as e:
        print(f"\nFailed to parse JSON output: {e}")

if __name__ == "__main__":
    run_real_db_test()