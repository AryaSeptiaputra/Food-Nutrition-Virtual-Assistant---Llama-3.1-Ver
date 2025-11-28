from src.ai_agent.components.database.DBclient import ChromaDBClient

class SemanticRetriever:
    """Semantic search interface for ChromaDB collections.
    
    Provides high-level methods for semantic similarity search and metadata-based filtering
    against a ChromaDB collection. Handles query formatting and result conversion.
    """
    def __init__(self, db_client: ChromaDBClient, collection_name: str, embedding_function):
        """Initialize the semantic retriever with a ChromaDB collection.
        
        Args:
            db_client: ChromaDBClient instance for database operations
            collection_name: Name of the ChromaDB collection to query
            embedding_function: Embedding function object for semantic search
        
        Raises:
            Exception: If collection cannot be accessed or created
        """
        self.client = db_client

        try:
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                embedding_function=embedding_function
            )
        except Exception as e:
            print(f"Error getting collection in SemanticRetriever: {e}")
            raise

    def search(self, query_text: str, n_results: int = 5, filters: dict = None):
        """Execute semantic similarity search on the ChromaDB collection.
        
        Performs vector-based semantic search to find documents most similar to the query.
        Results are filtered by optional metadata conditions and formatted for output.
        
        Args:
            query_text: Search query text to embed and match against documents
            n_results: Maximum number of results to return (default: 5)
            filters: Optional ChromaDB where clause for metadata filtering
        
        Returns:
            List of dictionaries containing metadata for matching documents.
            Empty list if no results found or query is empty.
        """
        if not query_text:
            return []

        try:
            results = self.collection.query(
                query_texts=[query_text],
                n_results=n_results,
                where=filters,
                include=["metadatas"]
            )

            return self.format_results(results)

        except Exception as e:
            print(f"Error during semantic search: {e}")
            return []

    def get_by_filter(self, filters: dict, n_results: int = 5):
        """Retrieve collection entries filtered by metadata criteria only.
        
        Performs metadata-based filtering without semantic search. Useful for
        exact matching or filtering by specific metadata fields without embedding queries.
        
        Args:
            filters: ChromaDB where clause specifying metadata filter conditions
            n_results: Maximum number of results to return (default: 5)
        
        Returns:
            List of dictionaries containing metadata for filtered documents.
            Empty list if no results found or filters are empty.
        """
        if not filters:
            return []

        try:
            results = self.collection.get(
                where=filters,
                limit=n_results,
                include=["metadatas"]
            )

            return self.format_get_results(results)

        except Exception as e:
            print(f"Error during get_by_filter: {e}")
            return []

    def format_results(self, results: dict) -> list[dict]:
        """Convert ChromaDB query output to a clean list of metadata dictionaries.
        
        Transforms the nested ChromaDB query response format into a flat list of
        metadata dictionaries. Handles missing or empty result structures gracefully.
        
        Args:
            results: ChromaDB query response containing 'metadatas' key
        
        Returns:
            List of metadata dictionaries from the first result batch.
            Empty list if results are malformed or empty.
        """
        if not results or 'metadatas' not in results or not results['metadatas']:
            return []

        return results.get('metadatas', [[]])[0]

    def format_get_results(self, results: dict) -> list[dict]:
        """Convert ChromaDB get output to a clean list of metadata dictionaries.
        
        Transforms the ChromaDB get() response format into a flat list of metadata.
        Handles missing or empty result structures gracefully.
        
        Args:
            results: ChromaDB get response containing 'metadatas' key
        
        Returns:
            List of metadata dictionaries from the results.
            Empty list if results are malformed or empty.
        """
        if not results or 'metadatas' not in results:
            return []

        return results.get('metadatas', [])