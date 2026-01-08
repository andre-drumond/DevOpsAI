# 🤖 DevOpsAI - RAG Local com Múltiplos Formatos

**DevOpsAI** é uma Prova de Conceito (POC) de uma solução de Inteligência Artificial generativa que roda 100% localmente. Ela utiliza a técnica **RAG (Retrieval-Augmented Generation)** para ler documentações técnicas em múltiplos formatos (PDF, TXT, Markdown, DOCX) e responder perguntas baseadas estritamente no conteúdo desses arquivos.

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

O container do Ollama inicia vazio. Você pode baixar modelos de duas formas:

**Opção 1: Pela Interface Web (Recomendado) 🎉**

1. Acesse o frontend em [http://localhost:8501](http://localhost:8501)
2. Na barra lateral, expanda a seção **"🤖 Modelo de IA"** e depois **"📦 Gerenciador de Modelos"**
3. Vá para a aba **"📥 Disponíveis"**
4. Use a busca para encontrar o modelo desejado (ex: `deepseek-r1:1.5b`)
5. Clique em **"Baixar"** e acompanhe o progresso (percentual em texto)
6. Aguarde a conclusão do download

**Opção 2: Via Terminal (Alternativa)**

Se preferir usar o terminal, execute:

```bash
# Baixar Llama 3.2
docker exec -it devopsai_ollama ollama pull llama3.2

# Ou baixar DeepSeek R1 (recomendado para velocidade)
docker exec -it devopsai_ollama ollama pull deepseek-r1:1.5b

# Para melhor qualidade (requer mais RAM)
docker exec -it devopsai_ollama ollama pull deepseek-r1:8b
```

> **Nota:** Com as novas funcionalidades, você pode escolher o modelo diretamente na interface do frontend, sem precisar editar código! Basta selecionar o modelo desejado na barra lateral.

## 🤖 Gerenciamento de Modelos

A aplicação permite gerenciar modelos de IA diretamente pela interface web!

### Funcionalidades

1. **Ver Modelos Instalados**: Lista todos os modelos já baixados no Ollama
2. **Baixar Novos Modelos**: Download de modelos diretamente pelo frontend com progresso em tempo real (percentual)
3. **Remover Modelos**: Delete modelos que não são mais necessários para liberar espaço
4. **Seleção de Modelo**: Escolha qual modelo usar para as conversas

### Como Usar

1. **Acesse a seção "🤖 Modelo de IA"** na barra lateral (expandida por padrão)
2. **Selecione um modelo**: Use o dropdown "Escolha o modelo" (mostra apenas modelos instalados)
3. **Gerenciar modelos**: Expanda a subseção **"📦 Gerenciador de Modelos"**
   - **Aba "📥 Disponíveis"**: Veja e baixe novos modelos
     - Use a busca para encontrar o modelo desejado
     - Clique em "Baixar" ao lado do modelo
     - Acompanhe o progresso do download (percentual em texto)
     - Aguarde a conclusão (pode levar alguns minutos dependendo do tamanho)
   - **Aba "✅ Instalados"**: Veja e gerencie modelos já baixados
     - Lista todos os modelos instalados com seus tamanhos
     - Clique no botão 🗑️ para remover um modelo

### Modelos Disponíveis

A aplicação suporta diversos modelos populares:

- **DeepSeek R1**: Modelos rápidos e eficientes (1.5B e 8B)
- **Llama**: Modelos da Meta (Llama 3.2, Llama 3, Llama 3.1)
- **Mistral**: Modelo de alta qualidade
- **Phi-3**: Modelo leve da Microsoft
- **Gemma 2**: Modelo do Google
- **Qwen 2.5**: Modelo de alta qualidade

### API Endpoints para Modelos

```bash
# Listar modelos instalados
GET /models/installed

# Listar modelos disponíveis
GET /models/available

# Listar tudo (instalados + disponíveis)
GET /models

# Baixar um modelo (streaming)
POST /models/pull
{
  "model": "deepseek-r1:1.5b"
}

# Remover um modelo
DELETE /models/{model_name}

# Informações sobre um modelo
GET /models/{model_name}/info
```

---

## 📖 Como Usar

### Estrutura da Interface

A interface está organizada em duas áreas principais:

- **Barra Lateral (Esquerda)**: Configurações e gerenciamento
  - **🤖 Modelo de IA**: Seleção de modelo e gerenciador de modelos
  - **🎛️ Parâmetros de Resposta**: Configurações de temperatura, top_p e streaming
  - **📂 Gerenciar Documentos**: Upload e gerenciamento de documentos
  
- **Área Principal (Direita)**: Chat com a IA
  - Histórico de conversas
  - Campo de input para perguntas
  - Respostas da IA baseadas na documentação

1. Acesse o frontend no navegador: **[http://localhost:8501](http://localhost:8501)**.

2. **Configurar Modelo e Parâmetros** (na barra lateral):
   * **Seção "🤖 Modelo de IA"** (expandida por padrão):
     - Escolha o modelo de IA no dropdown (mostra apenas modelos instalados)
     - Se não houver modelos instalados, baixe um na subseção "📦 Gerenciador de Modelos"
   * **Seção "🎛️ Parâmetros de Resposta"**:
     - Ajuste a **Temperatura** (0.0-2.0): controla criatividade
     - Ajuste o **Top P** (0.0-1.0): controla diversidade
     - Ative **Streaming de Respostas** para respostas em tempo real

3. **Upload de Documentos** (na barra lateral - seção "📂 Gerenciar Documentos"):
   * Clique em **"Browse files"** e selecione um arquivo (PDF, TXT, MD, DOCX)
   * Opcionalmente, expanda "⚙️ Metadados (Opcional)" e adicione metadados customizados em JSON
   * Clique em **"📤 Processar Arquivo"** e aguarde a indexação
   * Clique em **"🔄 Atualizar Lista"** para ver documentos indexados

4. **Gerenciar Documentos**:
   * Veja detalhes de cada documento expandindo o card (chunks, hash, metadados)
   * Use os botões **"ℹ️ Detalhes"** e **"🗑️ Deletar"** para gerenciar documentos

5. **Chat com a IA** (área principal):
   * Faça perguntas sobre o conteúdo dos documentos no campo de chat
   * As respostas são baseadas exclusivamente na base de conhecimento
   * Com streaming ativado, veja a resposta sendo gerada em tempo real
   * Use o botão **"🗑️ Limpar Conversa"** para resetar o histórico



---

### Modelos Recomendados

- **`deepseek-r1:1.5b`** (~1.1GB) - Mais rápido, ideal para respostas rápidas ⚡
- **`deepseek-r1:8b`** (~4.7GB) - Melhor qualidade, requer mais RAM 🎯
- **`llama3.2`** (~2GB) - Modelo balanceado da Meta ⚖️
- **`mistral`** (~4.1GB) - Alta qualidade 🌟

**Nota:** Tudo pode ser feito pela interface web! Não é mais necessário editar código ou usar comandos do terminal para gerenciar modelos.

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

### Endpoints Principais

* **`GET /health`**: Healthcheck do backend
* **`GET /models`**: Lista modelos disponíveis
* **`POST /upload`**: Upload de arquivos (PDF, TXT, MD, DOCX)
  ```bash
  curl -X POST "http://localhost:8000/upload" \
    -F "file=@documento.pdf" \
    -F "custom_metadata={\"categoria\":\"devops\"}"
  ```

* **`GET /documents`**: Lista todos os documentos indexados com metadados
* **`GET /documents/{filename}`**: Detalhes de um documento específico
* **`DELETE /documents/{filename}`**: Remove um documento da base
* **`PUT /documents/{filename}/metadata`**: Atualiza metadados de um documento
  ```bash
  curl -X PUT "http://localhost:8000/documents/arquivo.pdf/metadata" \
    -H "Content-Type: application/json" \
    -d '{"custom_metadata": {"versao": "2.0"}}'
  ```

* **`POST /chat`**: Chat com a IA (suporta streaming)
  ```bash
  # Modo normal
  curl -X POST "http://localhost:8000/chat" \
    -H "Content-Type: application/json" \
    -d '{
      "question": "O que é Docker?",
      "model": "deepseek-r1:1.5b",
      "temperature": 0.7,
      "top_p": 0.9,
      "stream": false
    }'
  
  # Modo streaming (SSE)
  curl -X POST "http://localhost:8000/chat" \
    -H "Content-Type: application/json" \
    -d '{"question": "Explique RAG", "stream": true}' \
    --no-buffer
  ```

## ✨ Funcionalidades Avançadas

### 📄 Suporte a Múltiplos Formatos

A aplicação agora suporta:
- **PDF** (.pdf) - Documentos PDF padrão
- **Texto** (.txt) - Arquivos de texto simples
- **Markdown** (.md, .markdown) - Documentação em Markdown
- **Word** (.docx) - Documentos Microsoft Word

### 🗂️ Gerenciamento de Documentos

- **Listar documentos**: Veja todos os documentos indexados com informações detalhadas
- **Detalhes**: Visualize chunks, hash do arquivo e metadados
- **Deletar**: Remova documentos da base de conhecimento
- **Metadados customizados**: Adicione informações extras aos documentos (categoria, versão, tags, etc.)

### 🤖 Múltiplos Modelos e Configurações

- **Seleção de modelo**: Escolha entre diferentes modelos (DeepSeek, Llama, Mistral, etc.) diretamente na sidebar
- **Download de modelos**: Baixe modelos diretamente pela interface com progresso em tempo real (percentual em texto)
- **Gerenciamento de modelos**: Visualize modelos instalados, baixe novos e remova modelos desnecessários
- **Temperatura**: Ajuste a criatividade das respostas (0.0 = determinístico, 2.0 = criativo)
- **Top P**: Controle a diversidade das respostas (0.0-1.0)
- **Configuração por requisição**: Cada pergunta pode usar configurações diferentes

### ⚡ Streaming de Respostas

- **Respostas em tempo real**: Veja a resposta sendo gerada token por token
- **Melhor UX**: Feedback imediato ao usuário
- **SSE (Server-Sent Events)**: Implementação eficiente de streaming

### 🔒 Validações e Segurança

- **Validação de formato**: Apenas formatos suportados são aceitos
- **Limite de tamanho**: Arquivos limitados a 50MB
- **Detecção de duplicatas**: Hash MD5 para identificar arquivos duplicados
- **Validação de metadados**: JSON validado antes de processar

---

## ⚠️ Troubleshooting

**Erro: "Nenhum documento indexado"**

* Certifique-se de que fez o upload e clicou em "📤 Processar Arquivo". Verifique a lista na seção "📂 Gerenciar Documentos" na sidebar.

**Erro de conexão com Ollama**

* Verifique se o container `devopsai_ollama` está rodando (`docker ps`).
* Verifique se você executou o passo 2 (download do modelo).

**Lentidão na resposta**

* Como é uma IA local, a velocidade depende 100% da sua CPU/GPU. Modelos menores como `deepseek-r1:1.5b` ou `llama3.2` são otimizados para velocidade. Para melhor performance, considere usar `deepseek-r1:1.5b` ou `phi3:mini`. Textos muito longos podem demorar alguns segundos mesmo assim.