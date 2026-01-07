# 🤖 DevOpsAI - RAG Local com PDFs

**DevOpsAI** é uma Prova de Conceito (POC) de uma solução de Inteligência Artificial generativa que roda 100% localmente. Ela utiliza a técnica **RAG (Retrieval-Augmented Generation)** para ler documentações técnicas em PDF e responder perguntas baseadas estritamente no conteúdo desses arquivos.

## 🚀 Stack Tecnológica

Esta solução foi desenhada para ser modular e conteinerizada:

* **LLM Engine:** [Ollama](https://ollama.com/) (executando Llama 3.2, DeepSeek, Mistral ou outros modelos locais).
* **Backend:** Python + FastAPI + LangChain (Orquestração e Ingestão).
* **Frontend:** Streamlit (Interface de Chat).
* **Vector Store:** ChromaDB (Persistência de vetores e metadados).
* **Infraestrutura:** Docker Compose.

---

## 📂 Estrutura do Projeto

```text
devopsai/
├── docker-compose.yml      # Orquestração dos containers e volumes
├── README.md               # Documentação
├── backend/                # API FastAPI
│   ├── Dockerfile
│   ├── requirements.txt
│   └── main.py             # Lógica RAG e Endpoints
└── frontend/               # Interface Streamlit
    ├── Dockerfile
    ├── requirements.txt
    └── app.py              # UI de Chat e Upload

```

---

## 🛠️ Como Rodar

### Pré-requisitos

* Docker e Docker Compose instalados.

### 1. Iniciar os Serviços

Na raiz do projeto, execute o comando para construir as imagens e subir os containers:

```bash
docker compose up --build -d

```

Isso iniciará três serviços:

* `devopsai_ollama` (Porta 11434)
* `devopsai_backend` (Porta 8000)
* `devopsai_frontend` (Porta 8501)

### 2. Baixar o Modelo de IA (Apenas na 1ª vez)

O container do Ollama inicia vazio. Precisamos baixar o modelo de linguagem. Execute o comando abaixo no seu terminal:

**Opção 1: Llama 3.2 (Padrão)**
```bash
docker exec -it devopsai_ollama ollama run llama3.2
```

*Isso fará o download do modelo Llama 3.2 (~2GB). Quando terminar e aparecer um prompt `>>>`, você pode digitar `/bye` ou pressionar `Ctrl+D` para sair.*

**Opção 2: DeepSeek R1 (Recomendado para velocidade)**

Para usar o DeepSeek (modelo rápido e eficiente), execute:

```bash
docker exec -it devopsai_ollama ollama pull deepseek-r1:1.5b
```

Ou para melhor qualidade (requer mais RAM):
```bash
docker exec -it devopsai_ollama ollama pull deepseek-r1:8b
```

Depois, atualize a variável `MODEL_NAME` em `backend/main.py` para `"deepseek-r1:1.5b"` ou `"deepseek-r1:8b"` e reinicie o backend:
```bash
docker compose restart backend
```

> **Nota:** Se desejar um modelo mais robusto e tiver hardware suficiente (8GB+ RAM), você pode usar `llama3`, `mistral` ou `deepseek-r1:8b`. Sempre atualize a variável `MODEL_NAME` em `backend/main.py` ao trocar de modelo.

---

## 📖 Como Usar

1. Acesse o frontend no navegador: **[http://localhost:8501](https://www.google.com/search?q=http://localhost:8501)**.
2. Na barra lateral (**Sidebar**):
* Clique em **"Browse files"** e selecione um PDF técnico.
* Clique em **"Processar PDF"**. Aguarde a mensagem de sucesso.
* Clique em **"🔄 Atualizar Lista"** para ver seu documento indexado na Base de Conhecimento.


3. No chat principal:
* Faça uma pergunta específica sobre o conteúdo do PDF.
* A IA irá processar e responder com base no contexto encontrado.



---

## 🚀 Usando DeepSeek (Modelo Rápido)

O **DeepSeek** é uma excelente alternativa ao Llama 3.2, oferecendo respostas mais rápidas mantendo boa qualidade. Está disponível localmente via Ollama.

### Opções de Modelos DeepSeek:

- **`deepseek-r1:1.5b`** (~1.1GB) - Mais rápido, ideal para respostas rápidas
- **`deepseek-r1:8b`** (~4.7GB) - Melhor qualidade, requer mais RAM
- **`deepseek`** - Versão padrão do DeepSeek Chat
- **`deepseek-coder`** - Especializado em código e programação

### Passos para habilitar DeepSeek:

1. **Baixar o modelo:**
   ```bash
   docker exec -it devopsai_ollama ollama pull deepseek-r1:1.5b
   ```

2. **Atualizar o backend:**
   Edite o arquivo `backend/main.py` e altere a linha:
   ```python
   MODEL_NAME = "deepseek-r1:1.5b"
   ```

3. **Reiniciar o backend:**
   ```bash
   docker compose restart backend
   ```

### Gerenciar Modelos Instalados:

**Listar modelos instalados:**
```bash
docker exec devopsai_ollama ollama list
```

**Deletar um modelo antigo (para liberar espaço):**
```bash
docker exec devopsai_ollama ollama rm llama3.2:latest
```

**Testar um modelo antes de usar:**
```bash
docker exec -it devopsai_ollama ollama run deepseek-r1:1.5b
```

---

## 💾 Persistência de Dados

O projeto utiliza **Docker Volumes** para garantir que os dados não sejam perdidos ao reiniciar os containers:

* **`ollama_storage`**: Mantém os modelos baixados (Llama, Mistral, etc) em `/root/.ollama`.
* **`chroma_storage`**: Mantém o índice vetorial dos seus PDFs em `/app/chroma_db`.

Para reiniciar a aplicação mantendo os dados:

```bash
docker compose restart

```

Para **apagar tudo** (resetar a IA e os documentos):

```bash
docker compose down -v

```

---

## 🔌 API Endpoints (Backend)

Se quiser interagir diretamente com a API (via Postman ou Curl):

* **`POST /upload`**: Envie um arquivo `multipart/form-data` (campo `file`) para indexação.
* **`GET /documents`**: Retorna uma lista JSON com os nomes dos arquivos já indexados.
* **`POST /chat`**: Envie um JSON `{"question": "Sua pergunta"}` para receber a resposta.

---

## ⚠️ Troubleshooting

**Erro: "Nenhum documento indexado"**

* Certifique-se de que fez o upload e clicou em "Processar PDF". Verifique a lista na sidebar.

**Erro de conexão com Ollama**

* Verifique se o container `devopsai_ollama` está rodando (`docker ps`).
* Verifique se você executou o passo 2 (download do modelo).

**Lentidão na resposta**

* Como é uma IA local, a velocidade depende 100% da sua CPU/GPU. Modelos menores como `deepseek-r1:1.5b` ou `llama3.2` são otimizados para velocidade. Para melhor performance, considere usar `deepseek-r1:1.5b` ou `phi3:mini`. Textos muito longos podem demorar alguns segundos mesmo assim.