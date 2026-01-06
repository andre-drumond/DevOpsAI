import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List

# LangChain imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain.chains import RetrievalQA

app = FastAPI(title="DevOpsAI Backend")

# Configurações
PERSIST_DIRECTORY = "/app/chroma_db" # Caminho absoluto para garantir o volume
OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
MODEL_NAME = "llama3.2"

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def get_vectorstore():
    return Chroma(
        persist_directory=PERSIST_DIRECTORY, 
        embedding_function=embeddings
    )

def get_qa_chain():
    vectorstore = get_vectorstore()
    llm = ChatOllama(base_url=OLLAMA_URL, model=MODEL_NAME)
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3})
    )

class QueryRequest(BaseModel):
    question: str

@app.get("/documents", response_model=List[str])
async def list_documents():
    """Lista os nomes únicos dos arquivos PDF indexados."""
    try:
        # Acessa diretamente a coleção do ChromaDB para pegar metadados
        vectorstore = get_vectorstore()
        
        # Pega todos os metadados disponíveis
        data = vectorstore.get()
        
        if not data or not data['metadatas']:
            return []

        # Extrai a fonte (source) e desduplica
        sources = set()
        for metadata in data['metadatas']:
            if metadata and 'source' in metadata:
                # Limpa o caminho do arquivo para mostrar apenas o nome (ex: "manual.pdf")
                filename = os.path.basename(metadata['source'])
                # Remove o prefixo "temp_" se existir
                if filename.startswith("temp_"):
                    filename = filename[5:]
                sources.add(filename)
        
        return list(sources)
    except Exception as e:
        # Se o banco não existir ainda, retorna lista vazia
        return []

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    # Salva o arquivo temporariamente
    temp_filename = f"temp_{file.filename}"
    with open(temp_filename, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        loader = PyPDFLoader(temp_filename)
        documents = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)

        # Atualiza o Vector Store persistente
        vectorstore = get_vectorstore()
        vectorstore.add_documents(chunks)
        
        return {"message": "PDF processado com sucesso!", "chunks": len(chunks)}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

@app.post("/chat")
async def chat(request: QueryRequest):
    try:
        qa_chain = get_qa_chain()
        response = qa_chain.invoke({"query": request.question})
        return {"answer": response["result"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao processar: {str(e)}")