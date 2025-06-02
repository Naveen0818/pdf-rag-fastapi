from fastapi import FastAPI, UploadFile, File, Form
from rag_engine import RAGAssistant
import os
from dotenv import load_dotenv

# Load API key from .env
load_dotenv()
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")

# Initialize FastAPI app and Gemini-based assistant
app = FastAPI()
assistant = RAGAssistant(gemini_api_key=GEMINI_API_KEY)

index = None
chunks = []

@app.post("/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    global index, chunks
    pdf_path = f"data/{file.filename}"
    with open(pdf_path, "wb") as f:
        f.write(await file.read())
    index, chunks = assistant.build_vector_store(pdf_path)
    return {"message": "PDF processed and vector DB created."}

@app.post("/ask/")
async def ask_question(question: str = Form(...)):
    if index is None:
        return {"error": "Please upload a PDF first."}
    retrieved = assistant.get_similar_chunks(index, chunks, question)
    answer = assistant.generate_answer(retrieved, question)
    return {"answer": answer}
