import streamlit as st
import requests
import os

# Configurações
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="DevOpsAI", page_icon="🤖")

st.title("🤖 DevOpsAI - RAG com PDFs")

# --- FUNÇÕES AUXILIARES ---
def get_indexed_documents():
    """Busca a lista de documentos indexados no backend."""
    try:
        response = requests.get(f"{BACKEND_URL}/documents")
        if response.status_code == 200:
            return response.json()
        return []
    except:
        return []

# --- SIDEBAR ---
with st.sidebar:
    st.header("📂 Gerenciar Documentos")
    
    # Upload
    uploaded_file = st.file_uploader("Novo arquivo PDF", type="pdf")
    if uploaded_file is not None:
        if st.button("Processar PDF"):
            with st.spinner("Indexando..."):
                files = {"file": (uploaded_file.name, uploaded_file, "application/pdf")}
                try:
                    response = requests.post(f"{BACKEND_URL}/upload", files=files)
                    if response.status_code == 200:
                        st.success("Indexado!")
                    else:
                        st.error(f"Erro: {response.text}")
                except Exception as e:
                    st.error(f"Erro: {e}")
    
    st.divider()
    
    # Listagem de Documentos
    st.subheader("📚 Base de Conhecimento")
    if st.button("🔄 Atualizar Lista"):
        st.rerun() # Recarrega a página para puxar os dados
        
    docs = get_indexed_documents()
    if docs:
        for doc in docs:
            st.text(f"• {doc}")
    else:
        st.caption("Nenhum documento indexado ainda.")

# --- CHAT ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Pergunte algo sobre a documentação..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Consultando base de conhecimento..."):
            try:
                payload = {"question": prompt}
                response = requests.post(f"{BACKEND_URL}/chat", json=payload)
                if response.status_code == 200:
                    answer = response.json().get("answer", "Sem resposta.")
                else:
                    answer = f"Erro no backend: {response.text}"
            except Exception as e:
                answer = f"Erro de conexão: {e}"
        
        st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})