FROM python:3.11-slim

# Define diretório de trabalho
WORKDIR /app

# Instala dependências do sistema necessárias
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copia requirements.txt primeiro para aproveitar cache do Docker
COPY requirements.txt .

# Instala dependências Python
RUN pip install --no-cache-dir -r requirements.txt

# Cria diretório para cache do modelo
RUN mkdir -p /root/.cache/huggingface/hub

# Copia o código da aplicação
COPY main.py .

# Pré-baixa o modelo durante o build para cachear na imagem
# Isso garante que o modelo não precise ser baixado a cada deploy
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')"

# Expõe a porta do FastAPI
EXPOSE 8000

# Comando para iniciar o servidor
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
