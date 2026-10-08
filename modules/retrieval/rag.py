import chromadb
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any

class RAGRetriever:
    def __init__(self, collection_name: str = "legal_documents"):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.client = chromadb.Client()
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def chunk_text(self, text: str, chunk_size: int = 500) -> List[str]:
        words = text.split()
        if not words:
            return []
        return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

    def add_pages(self, pages: List[Dict[str, Any]]) -> None:
        documents = []
        ids = []
        metadatas = []

        for page in pages:
            page_num = page.get("page", 0)
            chunks = self.chunk_text(page.get("text", ""))

            for i, chunk in enumerate(chunks):
                documents.append(chunk)
                ids.append(f"page_{page_num}_chunk_{i}")
                metadatas.append({"page": page_num})

        if not documents:
            return

        embeddings = self.model.encode(documents).tolist()

        self.collection.add(
            documents=documents,
            embeddings=embeddings,
            ids=ids,
            metadatas=metadatas
        )

    def search(self, question: str, top_k: int = 3) -> List[Dict[str, Any]]:
        count = self.collection.count()
        if count == 0:
            return []

        actual_k = min(top_k, count)
        query_embedding = self.model.encode([question]).tolist()

        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=actual_k
        )

        sources = []
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]

        for doc, meta in zip(docs, metas):
            sources.append({
                "page": meta["page"],
                "text": doc
            })

        return sources