import pandas as pd
from src.ai_agent.components.database.DBclient import ChromaDBClient

class ChromaUploader:
    """
    Meng-upload data knowledge base ke collection ChromaDB.
    """
    def __init__(self, db_client: ChromaDBClient, collection_name: str, embedding_function):
        self.client = db_client
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=embedding_function
        )

    def _prepare_data(self, knowledge_data: list[dict]):
        """
        Mengubah list of dicts (dari Proses 5) menjadi format ChromaDB.
        (IDs, Documents, Metadatas)
        """
        ids = []
        documents = []
        metadatas = []

        for item in knowledge_data:

            item_id = item.get("kode")
            if not item_id:
                continue

            doc_text = item.get("deskripsi_sintetis")
            if not doc_text:
                continue

            meta = item.copy()
            meta.pop("deskripsi_sintetis", None)
            
            nutrisi_dict = meta.pop("nutrisi", {}) 
            
            try:
                meta['berat'] = float(meta.get('berat', 0.0))
            except (ValueError, TypeError):
                meta['berat'] = 0.0

            if nutrisi_dict:
                for key, val_dict in nutrisi_dict.items():
                    try:
                        nilai = val_dict.get('nilai')
                        satuan = val_dict.get('satuan', '') # 1. Ambil data satuan
                        
                        if pd.isna(nilai):
                            nilai = 0.0
                        
                        # Bersihkan key (misal: "Energi (Energy)" -> "energi")
                        clean_key_root = key.split('(')[0].strip().lower().replace(' ', '_')
                        meta_key_val = f"nutrisi_{clean_key_root}"
                        
                        # 2. Simpan NILAI sebagai float (agar bisa difilter/dihitung)
                        meta[meta_key_val] = float(nilai)

                        # 3. Simpan SATUAN sebagai metadata terpisah
                        # Contoh key baru: "nutrisi_energi_unit"
                        meta_key_unit = f"{meta_key_val}_unit"
                        meta[meta_key_unit] = str(satuan)
                        
                    except Exception:
                        pass

            ids.append(item_id)
            documents.append(doc_text)
            metadatas.append(meta)

        return ids, documents, metadatas

    def upload_data(self, knowledge_data: list[dict]):
        """
        Melakukan 'upsert' data ke ChromaDB.
        Upsert = Update jika ID sudah ada, Insert jika belum.
        """
        if not knowledge_data:
            return {"processed": 0, "total_in_collection": self.collection.count()}

        ids, documents, metadatas = self._prepare_data(knowledge_data)

        if not ids:
            return {"processed": 0, "total_in_collection": self.collection.count()}

        try:
            self.collection.upsert(
                ids=ids,
                documents=documents,
                metadatas=metadatas
            )
            
            total_count = self.collection.count()
            
            return {
                "processed": len(ids),
                "total_in_collection": total_count
            }
        except Exception as e:
            raise