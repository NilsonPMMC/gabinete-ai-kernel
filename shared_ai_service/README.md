# Shared AI Service

Serviço FastAPI para embeddings, similaridade semântica e chat com LLM local.

## Modelo de embeddings atual

- `mixedbread-ai/mxbai-embed-large-v1`
- dimensão do vetor: `1024`

## Instalação

### Local

```bash
pip install -r requirements.txt
```

### Docker

```bash
docker build -t shared-ai-service .
docker run -p 8004:8000 shared-ai-service
```

### Docker Compose

```bash
docker compose up --build
```

## Uso

### Iniciar o servidor (local)

```bash
python main.py
# ou
uvicorn main:app --host 0.0.0.0 --port 8004
```

## Endpoints

### Health Check

```bash
curl http://localhost:8004/
```

### Embeddings

```bash
curl -X POST "http://localhost:8004/v1/embeddings" \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Olá mundo", "Hello world"]}'
```

### Similaridade

```bash
curl -X POST "http://localhost:8004/v1/similarity" \
  -H "Content-Type: application/json" \
  -d '{
    "target_text": "João da Silva",
    "candidates": ["Joao Silva", "Maria Oliveira", "Prefeitura Municipal"]
  }'
```

### Chat (Ollama)

```bash
curl -X POST "http://localhost:8004/v1/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "system_prompt": "Você é um assistente objetivo.",
    "user_prompt": "Resuma o objetivo do serviço."
  }'
```

## Estrutura

- `main.py`: API FastAPI com endpoints de embeddings, similaridade e chat
- `requirements.txt`: dependências Python
- `Dockerfile`: build da aplicação e pré-download do modelo
- `docker-compose.yml`: orquestração da API com Ollama
