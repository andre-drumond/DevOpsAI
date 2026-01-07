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
from langchain_classic.chains import RetrievalQA
try:
    from langchain_core.prompts import PromptTemplate
    from langchain_core.documents import Document
except ImportError:
    from langchain.prompts import PromptTemplate
    from langchain.schema import Document

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

def extract_keywords(question: str) -> List[str]:
    """Extrai palavras-chave significativas da pergunta (palavras com mais de 3 caracteres)."""
    words = question.lower().split()
    # Remove palavras muito comuns e mantém apenas palavras significativas
    stop_words = {'o', 'a', 'os', 'as', 'um', 'uma', 'de', 'da', 'do', 'em', 'na', 'no', 'sobre', 'para', 'com', 'que', 'me', 'você'}
    keywords = [w for w in words if len(w) > 3 and w not in stop_words]
    return keywords

def find_relevant_documents(vectorstore, question: str, keywords: List[str], max_docs: int = 20) -> List[Document]:
    """Busca documentos relevantes usando busca semântica e filtro por palavras-chave."""
    relevant_docs = []
    
    # Primeira tentativa: busca semântica usando similarity_search diretamente
    try:
        # Usa similarity_search diretamente no vectorstore
        semantic_docs = vectorstore.similarity_search(question, k=max_docs)
    except Exception:
        # Fallback: tenta usar o retriever com invoke
        try:
            retriever = vectorstore.as_retriever(search_kwargs={"k": max_docs})
            semantic_docs = retriever.invoke(question)
        except Exception:
            semantic_docs = []
    
    # Filtra documentos que contêm pelo menos uma palavra-chave
    question_lower = question.lower()
    for doc in semantic_docs:
        content_lower = doc.page_content.lower()
        # Verifica se o documento contém palavras-chave da pergunta
        if any(keyword in content_lower for keyword in keywords) or any(word in content_lower for word in question_lower.split() if len(word) > 3):
            relevant_docs.append(doc)
    
    # Se não encontrou documentos relevantes, busca em todos os documentos
    if not relevant_docs:
        all_data = vectorstore.get()
        if all_data.get('documents'):
            for i, content in enumerate(all_data['documents']):
                content_lower = content.lower()
                # Verifica se contém palavras-chave
                if any(keyword in content_lower for keyword in keywords):
                    metadata = all_data['metadatas'][i] if all_data.get('metadatas') and i < len(all_data['metadatas']) else {}
                    doc = Document(page_content=content, metadata=metadata)
                    relevant_docs.append(doc)
                    if len(relevant_docs) >= max_docs:
                        break
    
    return relevant_docs

def get_qa_chain(retriever=None):
    """Cria uma cadeia de Q&A com retriever customizado ou padrão."""
    vectorstore = get_vectorstore()
    llm = ChatOllama(base_url=OLLAMA_URL, model=MODEL_NAME)
    
    # Prompt template mais restritivo que instrui o modelo a usar APENAS a base de conhecimento
    prompt_template = """Você é um assistente especializado em responder perguntas baseado APENAS no contexto fornecido da base de conhecimento interna.

INSTRUÇÕES CRÍTICAS:
- Use EXCLUSIVAMENTE as informações do contexto abaixo
- Se a informação não estiver no contexto, diga claramente: "Não encontrei essa informação na base de conhecimento"
- NÃO invente informações
- NÃO use conhecimento externo ou informações gerais que não estejam no contexto
- NÃO mencione tecnologias ou ferramentas que não estejam explicitamente no contexto fornecido
- Foque APENAS no que está documentado no contexto

Contexto da Base de Conhecimento:
{context}

Pergunta do usuário: {question}

Resposta (baseada EXCLUSIVAMENTE no contexto acima):"""
    
    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    
    # Se não foi fornecido um retriever, usa o padrão
    if retriever is None:
        retriever = vectorstore.as_retriever(
            search_kwargs={"k": 15},  # Busca mais documentos
            search_type="similarity"
        )
    
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": PROMPT},
        return_source_documents=True
    )

class QueryRequest(BaseModel):
    question: str

@app.get("/health")
async def health():
    """Endpoint de healthcheck para verificar se o backend está pronto."""
    return {"status": "healthy"}

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
        # Verifica se há documentos na base de conhecimento
        vectorstore = get_vectorstore()
        data = vectorstore.get()
        
        if not data or not data.get('ids') or len(data['ids']) == 0:
            return {
                "answer": "Não há documentos na base de conhecimento ainda. Por favor, faça upload de PDFs primeiro usando a opção na barra lateral."
            }
        
        # Extrai palavras-chave da pergunta
        keywords = extract_keywords(request.question)
        
        # Busca documentos relevantes usando busca híbrida (semântica + palavras-chave)
        relevant_docs = find_relevant_documents(vectorstore, request.question, keywords, max_docs=20)
        
        if not relevant_docs:
            return {
                "answer": "Não encontrei informações relevantes na base de conhecimento para responder sua pergunta. Tente reformular a pergunta ou verifique se os documentos contêm essa informação."
            }
        
        # Limita o número de documentos para não exceder o contexto do modelo
        relevant_docs = relevant_docs[:15]
        
        # Cria um retriever customizado com os documentos relevantes encontrados
        # Para isso, vamos usar uma abordagem direta: construir o contexto e chamar o LLM
        context_text = "\n\n".join([doc.page_content for doc in relevant_docs])
        
        llm = ChatOllama(base_url=OLLAMA_URL, model=MODEL_NAME)
        prompt_template = """Você é um assistente especializado em responder perguntas baseado APENAS no contexto fornecido da base de conhecimento interna.

INSTRUÇÕES CRÍTICAS:
- Use EXCLUSIVAMENTE as informações do contexto abaixo
- Se a informação não estiver no contexto, diga claramente: "Não encontrei essa informação na base de conhecimento"
- NÃO invente informações
- NÃO use conhecimento externo ou informações gerais que não estejam no contexto
- NÃO mencione tecnologias ou ferramentas que não estejam explicitamente no contexto fornecido
- Foque APENAS no que está documentado no contexto

Contexto da Base de Conhecimento:
{context}

Pergunta do usuário: {question}

Resposta (baseada EXCLUSIVAMENTE no contexto acima):"""
        
        PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        formatted_prompt = PROMPT.format(context=context_text, question=request.question)
        llm_response = llm.invoke(formatted_prompt)
        answer = llm_response.content if hasattr(llm_response, 'content') else str(llm_response)
        
        return {"answer": answer}
        
    except Exception as e:
        import traceback
        return {
            "answer": f"Erro ao processar: {str(e)}\n\nDetalhes: {traceback.format_exc()}"
        }