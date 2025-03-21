import os

from fastapi import FastAPI, UploadFile
import pdfplumber
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from openai import OpenAI

app = FastAPI()
vector_store = Chroma("pdf_db")

API_KEY = os.getenv("OPENAI_API_KEY")

@app.post("/upload/")
async def upload_pdf(file: UploadFile):
    with pdfplumber.open(file.file) as pdf:
        text = " ".join(page.extract_text() for page in pdf.pages)
        chunks = __chunk_text(text)
        vector_store.add_texts(chunks)
    return {"message": "PDF uploaded successfully"}

@app.post("/query/")
async def query_rag(query: str):
    retrieved_docs = vector_store.similarity_search(query)
    context = "\n".join([doc.page_content for doc in retrieved_docs])

    client = OpenAI(api_key="API_KEY")
    completion = client.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "system", "content": f"Context: {context}"},
                  {"role": "user", "content": query}]
    )
    
    return {"response": completion['choices'][0]['message']['content']}


def __chunk_text(text, chunk_size=1000):
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
