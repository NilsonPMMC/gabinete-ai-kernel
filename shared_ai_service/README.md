# Shared AI Service

Serviço FastAPI para servir modelos de Deep Learning, especificamente o modelo de embeddings `paraphrase-multilingual-MiniLM-L12-v2`.

## Instalação

### Local

```bash
pip install -r requirements.txt
```

### Docker

```bash
docker build -t shared-ai-service .
docker run -p 8000:8000 shared-ai-service
```

## Uso

### Iniciar o servidor

```bash
python main.py
# ou
uvicorn main:app --host 0.0.0.0 --port 8000
```

### Endpoint de Embeddings

```bash
curl -X POST "http://localhost:8000/v1/embeddings" \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Olá mundo", "Hello world"]}'
```

### Health Check

```bash
curl http://localhost:8000/health
```

## Estrutura

- `main.py`: Aplicação FastAPI principal
- `requirements.txt`: Dependências do projeto
- `Dockerfile`: Configuração Docker otimizada para CPU
