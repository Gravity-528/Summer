from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_qdrant import Qdrant, QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
from pathlib import Path
import os
import re

load_dotenv()

app = FastAPI()

client = OpenAI()
_embedding = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))

_qdrant_client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)

collection_name = "medical_diagnosis"

class ChatRequest(BaseModel):
    query: str

def clean_text(text: str) -> str:
    return re.sub(r'[\ud800-\udfff]', '', text)

@app.get("/")
def read_root():
    return {"message": "Welcome to the Medical RAG Server!"}

@app.post("/ingestion")
def read_ingestion():
    pdf_folder = (Path(__file__).parent / "./Data for DR").resolve()
    print(f"Loading PDF files from: {pdf_folder}")

    all_docs = []

    for pdf_file in pdf_folder.glob("*.pdf"):
        print(f"Loading: {pdf_file.name}")
        loader = PyPDFLoader(str(pdf_file))
        docs = loader.load()
        for doc in docs:
            doc.page_content = clean_text(doc.page_content)  
        all_docs.extend(docs)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=400
    )
    split_docs = text_splitter.split_documents(all_docs)

    if not _qdrant_client.collection_exists(collection_name):
        _qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config={
                "size": len(_embedding.embed_query("test")),
                "distance": "Cosine"
            }
        )

    vector_db = Qdrant(
        client=_qdrant_client,
        collection_name=collection_name,
        embeddings=_embedding  
    )

    vector_db.add_documents(split_docs)  

    print("Indexing of all PDFs is complete.")
    return {"message": "Ingestion and indexing complete."}

@app.post("/chat")
def read_chat(request: ChatRequest):
    query = request.query

    vector_db = Qdrant(
        client=_qdrant_client,
        collection_name=collection_name,
        embeddings=_embedding
    )

    search_results = vector_db.similarity_search(query=query, k=5)

    context = "\n\n\n".join([
        f"Page Content: {doc.page_content}\nPage Number: {doc.metadata.get('page_label', 'N/A')}\nFile Location: {doc.metadata.get('source', 'N/A')}"
        for doc in search_results
    ])

    system_prompt = f"""
    You are a helpful AI Assistant who answers user queries based on the available context
    retrieved from a PDF file along with page contents and page number.

    You should only answer the user based on the following context and guide them
    to open the right page number to know more.

    Context:
    {context}
    """

    chat_completion = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]
    )

    answer = chat_completion.choices[0].message.content
    return {"response": answer}
