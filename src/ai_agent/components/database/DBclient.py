import chromadb

class ChromaDBClient:
    def __init__(self, db_path: str, in_memory: bool = False):
        """Initialize ChromaDB client with optional persistent or in-memory storage.
        
        Creates a ChromaDB client instance. Can be configured to use persistent storage
        (saved to disk) or in-memory storage (temporary, lost on exit).
        
        Args:
            db_path: File system path where ChromaDB data will be persisted (ignored if in_memory=True)
            in_memory: If True, use in-memory storage; if False, use persistent storage (default: False)
        
        Raises:
            Exception: If ChromaDB client initialization fails
        """
        self.db_path = db_path

        try:
            if in_memory:
                self.client = chromadb.Client()
            else:
                self.client = chromadb.PersistentClient(path=self.db_path)
        except Exception:
            raise

    def get_or_create_collection(self, name: str, embedding_function):
        """Get an existing collection or create a new one if it doesn't exist.
        
        Attempts to retrieve a collection by name. If the collection doesn't exist,
        it will be created with the provided embedding function.
        
        Args:
            name: Name of the collection to get or create
            embedding_function: Embedding function object (e.g., SentenceTransformerEmbeddingFunction)
        
        Returns:
            ChromaDB collection object ready for add/query operations
        
        Raises:
            Exception: If collection retrieval/creation fails
        """
        try:
            collection = self.client.get_or_create_collection(
                name=name,
                embedding_function=embedding_function
            )
            return collection
        except Exception:
            raise

    def get_collection(self, name: str, embedding_function):
        """Retrieve an existing collection from ChromaDB.
        
        Fetches an existing collection by name. The collection must already exist.
        Use get_or_create_collection() if the collection might not exist.
        
        Args:
            name: Name of the collection to retrieve
            embedding_function: Embedding function object (must match the collection's embedding function)
        
        Returns:
            ChromaDB collection object for querying and retrieval
        
        Raises:
            Exception: If collection does not exist or retrieval fails
        """
        try:
            collection = self.client.get_collection(
                name=name,
                embedding_function=embedding_function
            )
            return collection
        except Exception:
            raise