import os
import sys

from src.ai_agent.components.database.DBclient import ChromaDBClient
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from src.config.settings import Settings

from src.ai_agent.core.orchestrator import AIAgentOrchestrator
from src.ai_agent.components.tools.agent_tool import (
    initialize_retriever, 
    initialize_vision_inference,
    search_food_nutrition_info, 
    detect_food_in_image
)


def main():
    """Initialize AI nutritional assistant system and run interactive chat loop.
    
    Performs complete system setup:
    1. Loads configuration from settings
    2. Initializes ChromaDB client and semantic retriever for food knowledge base
    3. Initializes YOLO vision inference for food detection (with graceful fallback)
    4. Creates AI agent orchestrator with ReAct pattern
    5. Starts interactive chat loop for user queries
    
    The system provides two main capabilities:
    - Search food nutrition information via semantic search
    - Detect foods in images using YOLO object detection
    
    User interaction:
    - Interactive loop processes user queries in real-time
    - Agent uses semantic search and vision tools to formulate responses
    - Type 'exit', 'quit', or 'q' to terminate
    - Handles KeyboardInterrupt gracefully
    
    Raises:
        Exception: Caught and printed during chat loop; does not crash the program
    """
    settings = Settings()
    
    chroma_path = os.path.join(settings.PROJECT_ROOT, settings.CHROMA_DB_PATH).replace("\\", "/")
    template_prompt_path = os.path.join(settings.PROJECT_ROOT, settings.TEMPLATE_PROMPT_PATH).replace("\\", "/")
    
    print("Initializing dependencies...")

    db_client = ChromaDBClient(db_path=chroma_path)

    embedding_function = SentenceTransformerEmbeddingFunction(
        model_name=settings.EMBEDDING_TEXT_MODEL_NAME
    )

    print("Preparing Tools for AI Agent...")

    initialize_retriever(
        db_client=db_client,
        collection_name=settings.CHROMA_COLLECTION_NAME,
        embedding_function=embedding_function
    )

    yolo_model_path = os.path.join(settings.PROJECT_ROOT, settings.YOLO_MODEL_PATH).replace("\\", "/")

    if os.path.exists(yolo_model_path):
        initialize_vision_inference(
            model_path=yolo_model_path,
            raw_dir=settings.RAW_DIR,
            cache_dir=settings.CACHE_DIR,
            annotated_dir=settings.ANNOTATED_DIR,
            temp_dir=settings.TEMP_DIR
        )
    else:
        print(f"[WARNING] YOLO model not found: {yolo_model_path}")
        print("Food detection might not work if called.")

    tools_list = [
        search_food_nutrition_info,
        detect_food_in_image
    ]

    orchestrator = AIAgentOrchestrator(
        model_name=settings.LLM_MODEL_NAME,
        tools=tools_list,
        mongo_path=settings.MONGO_CONNECTION_STRING,
        template_prompt_path=template_prompt_path
    )

    session_id = "test_session_vision_01"
    
    print("\n" + "="*60)
    print("🤖 Nutritional Assistant - Llama 3.1 Ver - Debug Mode")
    print(" - Type 'exit' to stop")
    print("="*60)

    while True:
        try:
            user_query = input("\nYou: ").strip()

            if user_query.lower() in ["exit", "quit", "q"]:
                print("\nThank you for using this nutrition assistant!")
                break

            result = orchestrator.invoke(session_id=session_id, user_message=user_query)
            
            print(f"\nAssistant: {result.get('final_answer', 'Sorry, no answer can be produced.')}")
        
        except KeyboardInterrupt:
            print("\n\n[INTERRUPTED] Program stopped.")
            break
        except Exception as e:
            print(f"\n[ERROR] {e}")
            print("Please try again.")

if __name__ == "__main__":
    main()
