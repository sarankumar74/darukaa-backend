import os
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

DB_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'faiss_index')

class RAGService:
    def __init__(self):
        self.embeddings = GoogleGenerativeAIEmbeddings(model="text-embedding-004")
        if os.path.exists(DB_DIR):
            self.vectorstore = FAISS.load_local(DB_DIR, self.embeddings, allow_dangerous_deserialization=True)
            self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})
        else:
            self.vectorstore = None
            self.retriever = None

    def retrieve(self, query: str):
        if not self.retriever:
            return []
        docs = self.retriever.invoke(query)
        return [{"content": doc.page_content, "source": doc.metadata.get("source", "Unknown")} for doc in docs]

rag_service = RAGService()
