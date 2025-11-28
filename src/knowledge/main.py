import os

from src.config.settings import Settings
from src.ai_agent.components.database.DBclient import ChromaDBClient
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# Import pipeline modules
from .ingestion.loader import load_data
from .ingestion.validator import validate_data
from .processing.preprocessing import preprocess
from .processing.merger import merge_data
from .processing.builder import generate_descriptions
from .persistence.chroma import ChromaUploader

settings = Settings()
details_path = os.path.join(settings.PROJECT_ROOT, settings.FOOD_DETAILS_PATH).replace("\\", "/")
nutritions_path = os.path.join(settings.PROJECT_ROOT, settings.FOOD_NUTRITIONS_PATH).replace("\\", "/")

chroma_path = os.path.join(settings.PROJECT_ROOT, settings.CHROMA_DB_PATH).replace("\\", "/")
collection_name = settings.CHROMA_COLLECTION_NAME

embedding_model_name = settings.EMBEDDING_TEXT_MODEL_NAME

def run_etl_pipeline():
    """Execute the complete ETL (Extract-Transform-Load) pipeline for the Knowledge Base.
    
    This function orchestrates the entire data processing workflow:
    1. Extracts food details and nutrition data from CSV files
    2. Validates data schema and structure
    3. Preprocesses data for consistency and quality
    4. Merges food details with nutrition information
    5. Generates synthetic descriptions for semantic search
    6. Loads processed data into ChromaDB vector database
    
    The pipeline handles:
    - Multiple data sources (food details and nutrition CSVs)
    - Schema validation for different dataset types
    - Data merging and enrichment
    - Vector embedding and semantic indexing
    - Error handling with informative messages
    
    Raises:
        FileNotFoundError: If input CSV files are not found at configured paths
        ValueError: If data validation fails due to missing or invalid columns
        ImportError: If required dependencies are not installed
        Exception: For any other unexpected errors during pipeline execution
    """
    try:
        print(f"Reading details data from: {details_path}")
        df_details = load_data(details_path)
        
        print(f"Reading nutrition data from: {nutritions_path}")
        df_nutrition = load_data(nutritions_path)
        print("Data loaded successfully.")

        print("Validating data schema...")
        df_details_valid = validate_data(df_details, "details")
        df_nutrition_valid = validate_data(df_nutrition, "nutrition")
        print("Data validated successfully.")

        print("Starting data preprocessing...")
        df_details_preprocessed = preprocess(df_details_valid, is_nutrition_data=False)
        df_nutrition_preprocessed = preprocess(df_nutrition_valid, is_nutrition_data=True)
        print("Data preprocessing completed.")

        print("Merging details and nutrition data...")
        knowledge_data = merge_data(df_details_preprocessed, df_nutrition_preprocessed)
        
        if not knowledge_data:
            print("Data is empty after merge. Pipeline stopped.")
            return
            
        print(f"Data merged successfully. Total items: {len(knowledge_data)}")

        print("Creating synthetic descriptions for data...")
        described_data = generate_descriptions(knowledge_data)
        print("Synthetic descriptions created successfully.")

        print("Setting up ChromaDB connection...")
        db_client = ChromaDBClient(chroma_path)
        
        embed_fn = SentenceTransformerEmbeddingFunction(
            model_name=embedding_model_name
        )
        
        uploader = ChromaUploader(
            db_client=db_client,
            collection_name=collection_name,
            embedding_function=embed_fn
        )
        print(f"Connection to collection '{collection_name}' established.")

        print("Starting data upload/upsert to ChromaDB...")
        upload_result = uploader.upload_data(described_data)
        
        print(f"Upload process completed.")
        print(f"   Items processed: {upload_result['processed']}")
        print(f"   Total data in collection: {upload_result['total_in_collection']}")
        
        print("Knowledge Base ETL Pipeline completed successfully.")

    except FileNotFoundError as e:
        print(f"Error: File not found. Ensure settings path is correct. {e}")
        raise
    except ValueError as e:
        print(f"Error: Data validation or value error. Check file columns. {e}")
        raise
    except ImportError as e:
        print(f"Error: Failed to import module. Ensure all dependencies are installed. {e}")
        raise
    except Exception as e:
        print(f"Unexpected error occurred in pipeline: {e}")
        raise

if __name__ == "__main__":
    run_etl_pipeline()