import streamlit as st
import requests
import os
import json

# Configurações
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(
    page_title="DevOpsAI", 
    page_icon="🤖", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS customizado para melhorar a aparência
st.markdown("""
<style>
    .model-card {
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid rgba(250, 250, 250, 0.2);
        margin-bottom: 0.75rem;
        background-color: rgba(38, 39, 48, 0.5);
    }
    .model-name {
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 0.25rem;
    }
    .model-info {
        font-size: 0.85rem;
        color: rgba(250, 250, 250, 0.7);
        margin-bottom: 0.5rem;
    }
    .error-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: rgba(255, 75, 75, 0.1);
        border-left: 4px solid #ff4b4b;
        margin-top: 0.5rem;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: rgba(0, 200, 83, 0.1);
        border-left: 4px solid #00c853;
        margin-top: 0.5rem;
    }
    .info-box {
        padding: 0.75rem;
        border-radius: 0.5rem;
        background-color: rgba(33, 150, 243, 0.1);
        border-left: 4px solid #2196f3;
        margin-top: 0.5rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
</style>
""", unsafe_allow_html=True)

st.title("🤖 DevOpsAI - RAG com Múltiplos Formatos")

# --- FUNÇÕES AUXILIARES ---
def get_indexed_documents():
    """Busca a lista de documentos indexados no backend."""
    try:
        response = requests.get(f"{BACKEND_URL}/documents")
        if response.status_code == 200:
            return response.json()
        return []
    except Exception as e:
        st.error(f"Erro ao buscar documentos: {e}")
        return []

def delete_document(filename: str):
    """Deleta um documento da base de conhecimento."""
    try:
        response = requests.delete(f"{BACKEND_URL}/documents/{filename}")
        if response.status_code == 200:
            return True, response.json().get("message", "Documento deletado")
        return False, response.json().get("detail", "Erro ao deletar")
    except Exception as e:
        return False, str(e)

def get_available_models():
    """Busca modelos disponíveis."""
    try:
        response = requests.get(f"{BACKEND_URL}/models")
        if response.status_code == 200:
            return response.json()
        return {"available_models": [], "installed_models": [], "default_model": "deepseek-r1:1.5b"}
    except:
        return {"available_models": [], "installed_models": [], "default_model": "deepseek-r1:1.5b"}

def get_installed_models():
    """Busca modelos instalados."""
    try:
        response = requests.get(f"{BACKEND_URL}/models/installed")
        if response.status_code == 200:
            return response.json()
        return {"installed_models": [], "count": 0}
    except:
        return {"installed_models": [], "count": 0}

def pull_model(model_name: str, progress_callback=None):
    """Baixa um modelo com feedback de progresso."""
    try:
        # Timeout maior para downloads (30 minutos)
        response = requests.post(
            f"{BACKEND_URL}/models/pull",
            json={"model": model_name},
            stream=True,
            timeout=1800  # 30 minutos
        )
        if response.status_code == 200:
            last_status = None
            has_data = False
            
            for line in response.iter_lines():
                if line:
                    line_str = line.decode('utf-8', errors='ignore')
                    has_data = True
                    
                    # Processa linhas que começam com "data: "
                    if line_str.startswith('data: '):
                        try:
                            data_str = line_str[6:].strip()
                            if data_str:
                                data = json.loads(data_str)
                                
                                # Atualiza callback de progresso
                                if progress_callback:
                                    try:
                                        progress_callback(data)
                                    except Exception as e:
                                        # Continua mesmo se o callback falhar
                                        pass
                                
                                # Verifica status
                                status = data.get("status", "")
                                if status:
                                    last_status = status
                                
                                # Verifica se há erro
                                if data.get("error"):
                                    error_msg = data.get("error", "Erro desconhecido")
                                    # Mensagens mais amigáveis para erros comuns
                                    if "digest mismatch" in error_msg.lower():
                                        return False, "Download interrompido ou corrompido. Por favor, tente baixar novamente."
                                    elif "timeout" in error_msg.lower():
                                        return False, "Timeout no download. O modelo pode ser muito grande. Tente novamente."
                                    elif "connect" in error_msg.lower():
                                        return False, "Não foi possível conectar com o Ollama. Verifique se o serviço está rodando."
                                    else:
                                        return False, f"Erro: {error_msg}"
                                
                                # Verifica se foi sucesso
                                if status == "success":
                                    return True, "Modelo baixado com sucesso!"
                                    
                        except json.JSONDecodeError as e:
                            # Ignora linhas que não são JSON válido
                            continue
                        except Exception as e:
                            # Log do erro mas continua
                            continue
                    # Também processa linhas que são JSON direto (sem "data: ")
                    else:
                        try:
                            data = json.loads(line_str)
                            if progress_callback:
                                try:
                                    progress_callback(data)
                                except:
                                    pass
                            
                            status = data.get("status", "")
                            if status == "success":
                                return True, "Modelo baixado com sucesso!"
                            elif data.get("error"):
                                error_msg = data.get("error", "Erro desconhecido")
                                if "digest mismatch" in error_msg.lower():
                                    return False, "Download interrompido ou corrompido. Por favor, tente baixar novamente."
                                return False, f"Erro: {error_msg}"
                        except:
                            # Ignora linhas que não são JSON
                            continue
            
            # Se chegou aqui sem sucesso explícito
            if not has_data:
                return False, "Nenhuma resposta recebida do servidor. Verifique a conexão."
            elif last_status == "success":
                return True, "Download concluído"
            else:
                return False, "Download interrompido ou incompleto. Tente novamente."
        else:
            error_text = response.text if hasattr(response, 'text') else "Erro desconhecido"
            return False, f"Erro HTTP {response.status_code}: {error_text}"
    except requests.exceptions.Timeout:
        return False, "Timeout: O download demorou muito. Tente novamente ou escolha um modelo menor."
    except requests.exceptions.ConnectionError:
        return False, "Erro de conexão: Não foi possível conectar com o backend."
    except Exception as e:
        return False, f"Erro inesperado: {str(e)}"

def delete_model(model_name: str):
    """Remove um modelo."""
    try:
        response = requests.delete(f"{BACKEND_URL}/models/{model_name}")
        if response.status_code == 200:
            return True, response.json().get("message", "Modelo removido")
        return False, response.json().get("detail", "Erro ao remover modelo")
    except Exception as e:
        return False, str(e)

# Inicializa session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "model" not in st.session_state:
    st.session_state.model = None
if "temperature" not in st.session_state:
    st.session_state.temperature = 0.7
if "top_p" not in st.session_state:
    st.session_state.top_p = 0.9
if "stream_enabled" not in st.session_state:
    st.session_state.stream_enabled = False

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Configurações")
    
    # Busca modelos uma vez
    models_info = get_available_models()
    installed_models = models_info.get("installed_models", [])
    available_models_list = models_info.get("available_models", [])
    default_model = models_info.get("default_model", "deepseek-r1:1.5b")
    installed_names = [model.get("name", "") for model in installed_models]
    installed_names = [name for name in installed_names if name]
    
    # === SEÇÃO: MODELO DE IA (Sempre visível) ===
    with st.expander("🤖 **Modelo de IA**", expanded=True):
        # Seleção de modelo
        if installed_names:
            selected_model = st.selectbox(
                "Escolha o modelo",
                options=installed_names,
                index=0,
                help="Modelo de linguagem a ser usado",
                label_visibility="visible",
                key="model_selector"
            )
            st.session_state.model = selected_model
            st.caption(f"✅ {len(installed_names)} modelo(s) disponível(is)")
        else:
            st.info("📭 **Nenhum modelo instalado**")
            st.caption("Baixe um modelo na seção 'Gerenciador de Modelos' abaixo para começar.")
            st.session_state.model = None
        
        st.divider()
        
        # Subseção: Gerenciador de Modelos
        with st.expander("📦 **Gerenciador de Modelos**", expanded=False):
            # Tabs para organizar melhor
            tab1, tab2 = st.tabs(["📥 Disponíveis", "✅ Instalados"])
            
            with tab1:
                st.caption(f"{len(available_models_list)} modelos disponíveis")
                
                # Busca
                search_term = st.text_input(
                    "🔍 Buscar",
                    placeholder="Digite o nome do modelo...",
                    key="model_search",
                    label_visibility="collapsed"
                )
    
                # Filtra modelos
                filtered_models = available_models_list
                if search_term:
                    filtered_models = [
                        m for m in filtered_models
                        if search_term.lower() in m.get("name", "").lower() or 
                           search_term.lower() in m.get("id", "").lower()
                    ]
                
                if not filtered_models:
                    st.info("🔍 Nenhum modelo encontrado. Tente outra busca.")
                else:
                    # Mostra modelos em cards
                    for model_info in filtered_models[:15]:
                        model_id = model_info.get("id", "")
                        model_name = model_info.get("name", "")
                        model_size = model_info.get("size", "")
                        model_desc = model_info.get("description", "")
                        is_installed = model_info.get("installed", False)
                        
                        # Card do modelo
                        with st.container():
                            col1, col2 = st.columns([1, 0.3])
                            
                            with col1:
                                if is_installed:
                                    st.markdown(f"✅ **{model_name}**")
                                else:
                                    st.markdown(f"⬇️ **{model_name}**")
                                
                                st.caption(f"📦 {model_size}")
                                st.caption(f"💬 {model_desc}")
                            
                            with col2:
                                if is_installed:
                                    st.markdown("<br>", unsafe_allow_html=True)
                                    st.caption("✅ Instalado")
                                else:
                                    download_key = f"download_{model_id}"
                                    if download_key not in st.session_state:
                                        st.session_state[download_key] = False
                                    
                                    # Verifica se há download em andamento para este modelo
                                    download_in_progress = st.session_state.get(download_key, False)
                                    download_progress = st.session_state.get(f"download_progress_{model_id}", 0)
                                    download_status = st.session_state.get(f"download_status_{model_id}", "")
                                    
                                    if download_in_progress:
                                        # Mostra status do download
                                        if download_progress > 0 and download_progress < 1.0:
                                            st.caption(f"📥 **{download_progress*100:.1f}%**")
                                            if download_status:
                                                st.caption(download_status)
                                        elif download_status:
                                            st.caption(download_status)
                                        else:
                                            st.caption("🔄 Fazendo download, aguarde...")
                                        
                                        # Processa o download se ainda não foi iniciado
                                        if not st.session_state.get(f"download_started_{model_id}", False):
                                            st.session_state[f"download_started_{model_id}"] = True
                                            
                                            # Função para atualizar progresso
                                            def update_progress(data):
                                                status_update = data.get("status", "")
                                                
                                                if status_update == "downloading":
                                                    completed = data.get("completed", 0)
                                                    total = data.get("total", 0)
                                                    if total > 0:
                                                        progress = completed / total
                                                        st.session_state[f"download_progress_{model_id}"] = progress
                                                        size_mb_completed = completed / (1024 * 1024)
                                                        size_mb_total = total / (1024 * 1024)
                                                        size_gb_completed = size_mb_completed / 1024
                                                        size_gb_total = size_mb_total / 1024
                                                        
                                                        if size_gb_total >= 1:
                                                            size_str = f"{size_gb_completed:.2f} GB / {size_gb_total:.2f} GB"
                                                        else:
                                                            size_str = f"{size_mb_completed:.1f} MB / {size_mb_total:.1f} MB"
                                                        
                                                        st.session_state[f"download_status_{model_id}"] = f"{size_str}"
                                                elif status_update == "success":
                                                    st.session_state[f"download_progress_{model_id}"] = 1.0
                                                    st.session_state[f"download_status_{model_id}"] = "✅ Concluído!"
                                                    st.session_state[download_key] = False
                                                    st.session_state[f"download_started_{model_id}"] = False
                                                elif status_update == "error" or data.get("error"):
                                                    error_msg = data.get("error", "Erro desconhecido")
                                                    st.session_state[f"download_status_{model_id}"] = f"❌ Erro"
                                                    st.session_state[download_key] = False
                                                    st.session_state[f"download_started_{model_id}"] = False
                                                elif data.get("status") == "pulling":
                                                    st.session_state[f"download_status_{model_id}"] = "🔄 Preparando..."
                                            
                                            # Inicia o download
                                            try:
                                                success, message = pull_model(model_id, update_progress)
                                                if success:
                                                    st.session_state[f"download_progress_{model_id}"] = 1.0
                                                    st.session_state[f"download_status_{model_id}"] = "✅ Concluído!"
                                                    st.session_state[download_key] = False
                                                    st.session_state[f"download_started_{model_id}"] = False
                                                    st.rerun()
                                                else:
                                                    st.session_state[f"download_status_{model_id}"] = f"❌ {message}"
                                                    st.session_state[download_key] = False
                                                    st.session_state[f"download_started_{model_id}"] = False
                                            except Exception as e:
                                                st.session_state[f"download_status_{model_id}"] = f"❌ Erro: {str(e)}"
                                                st.session_state[download_key] = False
                                                st.session_state[f"download_started_{model_id}"] = False
                                        
                                        # Auto-refresh se ainda estiver em andamento
                                        if download_progress < 1.0 and "❌" not in download_status and "✅" not in download_status:
                                            import time
                                            time.sleep(1)
                                            st.rerun()
                                    else:
                                        if st.button("Baixar", key=f"pull_{model_id}", use_container_width=True, type="primary"):
                                            st.session_state[download_key] = True
                                            st.session_state[f"download_progress_{model_id}"] = 0
                                            st.session_state[f"download_status_{model_id}"] = ""
                                            st.session_state[f"download_started_{model_id}"] = False
                                            st.rerun()
                            
                            st.divider()
                    
                    if len(filtered_models) > 15:
                        st.caption(f"📄 Mostrando 15 de {len(filtered_models)} modelos. Use a busca para filtrar.")
            
            with tab2:
                if not installed_models:
                    st.info("📭 Nenhum modelo instalado ainda.")
                    st.caption("Vá para a aba 'Disponíveis' para baixar modelos.")
                else:
                    st.caption(f"{len(installed_models)} modelo(s) instalado(s)")
                    
                    for model in installed_models:
                        model_name = model.get("name", "Desconhecido")
                        model_size = model.get("size", 0)
                        size_gb = f"{model_size / (1024*1024*1024):.2f} GB" if model_size > 0 else "N/A"
                        
                        with st.container():
                            col1, col2 = st.columns([1, 0.3])
                            
                            with col1:
                                st.markdown(f"✅ **{model_name}**")
                                st.caption(f"📦 Tamanho: {size_gb}")
                            
                            with col2:
                                if st.button("🗑️", key=f"del_{model_name}", help="Remover modelo", use_container_width=True):
                                    success, msg = delete_model(model_name)
                                    if success:
                                        st.success(msg)
                                        st.rerun()
                                    else:
                                        st.error(msg)
                            
                            st.divider()
    
    # === SEÇÃO: PARÂMETROS DE RESPOSTA ===
    with st.expander("🎛️ **Parâmetros de Resposta**", expanded=False):
        st.session_state.temperature = st.slider(
            "Temperatura",
            min_value=0.0,
            max_value=2.0,
            value=st.session_state.temperature,
            step=0.1,
            help="Controla a criatividade das respostas (0.0 = mais determinístico, 2.0 = mais criativo)"
        )
        
        st.session_state.top_p = st.slider(
            "Top P",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.top_p,
            step=0.05,
            help="Controla a diversidade das respostas"
        )
        
        st.session_state.stream_enabled = st.checkbox(
            "Streaming de Respostas",
            value=st.session_state.stream_enabled,
            help="Mostra a resposta em tempo real (mais rápido)"
        )
    
    # === SEÇÃO: DOCUMENTOS ===
    with st.expander("📂 **Gerenciar Documentos**", expanded=False):
        # Upload de arquivo
        st.markdown("**📤 Upload de Arquivo**")
        uploaded_file = st.file_uploader(
            "Selecione um arquivo",
            type=["pdf", "txt", "md", "markdown", "docx"],
            help="Formatos suportados: PDF, TXT, Markdown, DOCX",
            label_visibility="collapsed"
        )
        
        if uploaded_file is not None:
            st.caption(f"📄 Arquivo: **{uploaded_file.name}**")
            
            custom_metadata = ""
            with st.expander("⚙️ Metadados (Opcional)", expanded=False):
                custom_metadata = st.text_area(
                    "Metadados em JSON",
                    height=100,
                    help='Exemplo: {"categoria": "devops", "versao": "1.0"}',
                    placeholder='{"categoria": "devops", "versao": "1.0"}',
                    label_visibility="collapsed"
                )
            
            if st.button("📤 Processar Arquivo", use_container_width=True, type="primary"):
                with st.spinner("⏳ Indexando documento..."):
                    files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
                    data = {}
                    if custom_metadata and custom_metadata.strip():
                        try:
                            json.loads(custom_metadata)
                            data["custom_metadata"] = custom_metadata
                        except:
                            st.warning("⚠️ JSON inválido, ignorando metadados...")
                    
                    try:
                        response = requests.post(f"{BACKEND_URL}/upload", files=files, data=data)
                        if response.status_code == 200:
                            result = response.json()
                            if result.get("duplicate"):
                                st.warning(f"⚠️ {result.get('message')}")
                            else:
                                st.success(f"✅ {result.get('message')} ({result.get('chunks')} chunks)")
                                st.rerun()
                        else:
                            st.error(f"❌ Erro: {response.text}")
                    except Exception as e:
                        st.error(f"❌ Erro: {e}")
        
        st.divider()
    
        # Lista de documentos
        st.markdown("**📚 Base de Conhecimento**")
        
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("🔄 Atualizar Lista", use_container_width=True):
                st.rerun()
        
        docs = get_indexed_documents()
        
        if docs:
            st.caption(f"📊 {len(docs)} documento(s) indexado(s)")
            for doc_info in docs:
                filename = doc_info.get("filename", "Desconhecido")
                chunks = doc_info.get("chunks", 0)
                
                with st.expander(f"📄 {filename}", expanded=False):
                    st.caption(f"**Chunks:** {chunks}")
                    if doc_info.get("file_hash"):
                        st.caption(f"**Hash:** `{doc_info['file_hash'][:16]}...`")
                    
                    if doc_info.get("custom_metadata"):
                        st.markdown("**Metadados:**")
                        st.json(doc_info["custom_metadata"])
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("ℹ️ Detalhes", key=f"info_{filename}", use_container_width=True):
                            try:
                                response = requests.get(f"{BACKEND_URL}/documents/{filename}")
                                if response.status_code == 200:
                                    details = response.json()
                                    st.json(details)
                            except Exception as e:
                                st.error(f"Erro: {e}")
                    
                    with col2:
                        if st.button("🗑️ Deletar", key=f"del_{filename}", use_container_width=True):
                            success, message = delete_document(filename)
                            if success:
                                st.success(message)
                                st.rerun()
                            else:
                                st.error(message)
        else:
            st.info("📭 Nenhum documento indexado ainda.")
            st.caption("Faça upload de arquivos acima para começar.")

# --- CHAT PRINCIPAL ---
# Mostra histórico de mensagens
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources_count"):
            st.caption(f"📚 Baseado em {message['sources_count']} documentos")
        if message.get("model"):
            st.caption(f"🤖 Modelo: {message['model']}")

# Input de chat
if prompt := st.chat_input("💬 Pergunte algo sobre a documentação..."):
    # Adiciona mensagem do usuário
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Prepara requisição
    payload = {
        "question": prompt,
        "model": st.session_state.model,
        "temperature": st.session_state.temperature,
        "top_p": st.session_state.top_p,
        "stream": st.session_state.stream_enabled
    }

    with st.chat_message("assistant"):
        if st.session_state.stream_enabled:
            # Modo streaming
            try:
                response = requests.post(
                    f"{BACKEND_URL}/chat",
                    json=payload,
                    stream=True,
                    timeout=300
                )
                
                if response.status_code == 200:
                    answer_placeholder = st.empty()
                    full_answer = ""
                    
                    for line in response.iter_lines():
                        if line:
                            line_str = line.decode('utf-8')
                            if line_str.startswith('data: '):
                                try:
                                    data = json.loads(line_str[6:])
                                    chunk = data.get('chunk', '')
                                    full_answer += chunk
                                    answer_placeholder.markdown(full_answer + "▌")
                                    
                                    if data.get('done'):
                                        answer_placeholder.markdown(full_answer)
                                        break
                                except:
                                    pass
                    
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": full_answer,
                        "model": payload.get("model"),
                        "temperature": payload.get("temperature")
                    })
                else:
                    error_msg = f"Erro no backend: {response.text}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })
            except Exception as e:
                error_msg = f"Erro de conexão: {e}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg
                })
        else:
            # Modo não-streaming
            with st.spinner("🤔 Consultando base de conhecimento..."):
                try:
                    response = requests.post(f"{BACKEND_URL}/chat", json=payload, timeout=300)
                    if response.status_code == 200:
                        result = response.json()
                        answer = result.get("answer", "Sem resposta.")
                        st.markdown(answer)
                        
                        message_data = {
                            "role": "assistant",
                            "content": answer
                        }
                        if result.get("sources_count"):
                            message_data["sources_count"] = result["sources_count"]
                            st.caption(f"📚 Baseado em {result['sources_count']} documentos")
                        if result.get("model"):
                            message_data["model"] = result["model"]
                            st.caption(f"🤖 Modelo: {result['model']}")
                        
                        st.session_state.messages.append(message_data)
                    else:
                        answer = f"Erro no backend: {response.text}"
                        st.error(answer)
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer
                        })
                except Exception as e:
                    answer = f"Erro de conexão: {e}"
                    st.error(answer)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer
                    })

# Botão para limpar conversa
if st.session_state.messages:
    if st.button("🗑️ Limpar Conversa", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
