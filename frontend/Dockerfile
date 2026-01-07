FROM python:3.10-slim

# Cria usuário não-root
RUN useradd -m -u 1000 appuser

WORKDIR /app

# Ajusta permissões do diretório /app
RUN chown -R appuser:appuser /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Muda para usuário não-root
USER appuser

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]