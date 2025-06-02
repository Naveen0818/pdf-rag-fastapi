import os
import faiss
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from typing import List, Tuple
from utils import load_pdf_text, chunk_text

class RAGAssistant:
    def __init__(self, gemini_api_key: str):
        if not gemini_api_key:
            raise ValueError("❌ Gemini API key is missing.")
        genai.configure(api_key=gemini_api_key)

        # ✅ Use only a currently supported model
        self.gemini_model_name = "models/gemini-1.5-flash"
        self.gemini_model = genai.GenerativeModel(self.gemini_model_name)
        print(f"✅ Using Gemini model: {self.gemini_model_name}")

        # ✅ Initialize SBERT for embeddings
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

    def build_vector_store(self, pdf_path: str) -> Tuple[faiss.IndexFlatL2, List[str]]:
        text = load_pdf_text(pdf_path)
        chunks = chunk_text(text)
        embeddings = self.embedder.encode(chunks, convert_to_numpy=True, normalize_embeddings=True).astype("float32")
        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings)
        print(f"✅ Indexed {len(chunks)} chunks from '{os.path.basename(pdf_path)}'")
        return index, chunks

    def get_similar_chunks(self, index: faiss.IndexFlatL2, chunks: List[str], question: str, top_k: int = 3) -> List[str]:
        q_embedding = self.embedder.encode([question], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
        distances, indices = index.search(q_embedding, top_k)
        return [chunks[i] for i in indices[0]]

    def generate_answer(self, context_chunks: List[str], question: str) -> str:
        context = "\n\n".join(context_chunks)
        prompt = f"""You are an AI assistant. Use the following context to answer the user's question.

Context:
{context}

Question: {question}
Answer:"""

        print("🔍 Prompt sent to Gemini:\n", prompt[:300])
        try:
            response = self.gemini_model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            print("❌ Gemini API error:", e)
            return "Gemini failed to generate an answer."
