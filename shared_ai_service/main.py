from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict
from sentence_transformers import SentenceTransformer, util

try:
    import ollama
except ImportError:
    ollama = None

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Configuração (Pydantic Settings)
# -----------------------------------------------------------------------------
class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    model_embedding: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    ollama_host: str | None = None  # None => usa padrão do ollama (localhost)
    ollama_model: str = "llama3.2:3b"


settings = Settings()

# -----------------------------------------------------------------------------
# Estado global (modelo de embeddings; cliente Ollama é opcional)
# -----------------------------------------------------------------------------
embedding_model: SentenceTransformer | None = None


def _encode_texts(texts: list[str], convert_to_tensor: bool = False) -> Any:
    """Chamada síncrona ao modelo (para ser executada em thread separada)."""
    if embedding_model is None:
        raise RuntimeError("Modelo de embeddings não carregado.")
    return embedding_model.encode(texts, convert_to_tensor=convert_to_tensor)


def _encode_single(text: str, convert_to_tensor: bool = False) -> Any:
    """Encode de um único texto (para uso em to_thread)."""
    if embedding_model is None:
        raise RuntimeError("Modelo de embeddings não carregado.")
    return embedding_model.encode(text, convert_to_tensor=convert_to_tensor)


# -----------------------------------------------------------------------------
# Lifespan
# -----------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global embedding_model
    logger.info("Iniciando Shared AI Service...")
    try:
        logger.info("Carregando modelo de embeddings: %s", settings.model_embedding)
        embedding_model = SentenceTransformer(settings.model_embedding, device="cpu")
        logger.info("Modelo de embeddings carregado com sucesso.")
    except Exception as e:
        logger.exception("Erro fatal ao carregar modelo de embeddings: %s", e)
        raise

    # Cliente Ollama assíncrono (opcional)
    ollama_client: ollama.AsyncClient | None = None
    if ollama is not None:
        try:
            ollama_client = ollama.AsyncClient(host=settings.ollama_host)
            app.state.ollama_client = ollama_client
            logger.info("Cliente Ollama configurado (host=%s, model=%s).", settings.ollama_host, settings.ollama_model)
        except Exception as e:
            logger.warning("Ollama não disponível: %s. Endpoint /v1/chat retornará 503.", e)
            app.state.ollama_client = None
    else:
        app.state.ollama_client = None

    yield

    if ollama_client is not None:
        await ollama_client.close()
        logger.info("Cliente Ollama encerrado.")
    embedding_model = None
    logger.info("Shutdown concluído.")


app = FastAPI(
    title="Shared AI Service",
    version="1.0",
    lifespan=lifespan,
)

# -----------------------------------------------------------------------------
# Schemas (request/response)
# -----------------------------------------------------------------------------
class EmbeddingRequest(BaseModel):
    texts: list[str]


class SimilarityRequest(BaseModel):
    target_text: str
    candidates: list[str]


class ChatRequest(BaseModel):
    system_prompt: str
    user_prompt: str


# -----------------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------------
@app.post("/v1/embeddings")
async def get_embeddings(request: EmbeddingRequest) -> dict[str, list[list[float]]]:
    """Gera vetores (embeddings) para uma lista de textos."""
    if embedding_model is None:
        raise HTTPException(status_code=503, detail="Modelo de embeddings não carregado.")
    try:
        embeddings = await asyncio.to_thread(
            _encode_texts,
            request.texts,
            False,
        )
        return {"embeddings": embeddings.tolist()}
    except Exception as e:
        logger.exception("Erro ao gerar embeddings: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/v1/similarity")
async def check_similarity(request: SimilarityRequest) -> dict[str, list[dict[str, Any]]]:
    """Compara um texto alvo com uma lista de candidatos (similaridade de cosseno)."""
    if embedding_model is None:
        raise HTTPException(status_code=503, detail="Modelo de embeddings não carregado.")
    try:

        def _compute_similarity() -> list[dict[str, Any]]:
            target_emb = _encode_single(request.target_text, convert_to_tensor=True)
            candidates_emb = _encode_texts(request.candidates, convert_to_tensor=True)
            scores = util.cos_sim(target_emb, candidates_emb)[0]
            results = [
                {"candidate": request.candidates[i], "score": float(scores[i]), "index": i}
                for i in range(len(request.candidates))
            ]
            results.sort(key=lambda x: x["score"], reverse=True)
            return results

        results = await asyncio.to_thread(_compute_similarity)
        return {"results": results}
    except Exception as e:
        logger.exception("Erro ao calcular similaridade: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/v1/chat")
async def chat(request: ChatRequest) -> str:
    """Envia system_prompt e user_prompt para o LLM (Ollama) e retorna apenas a string de resposta."""
    client: ollama.AsyncClient | None = getattr(app.state, "ollama_client", None)
    if client is None:
        if ollama is None:
            raise HTTPException(
                status_code=503,
                detail="Módulo 'ollama' não encontrado. Instale com: pip install ollama.",
            )
        raise HTTPException(
            status_code=503,
            detail="Ollama não está disponível. Inicie o Ollama (ex.: ollama serve) e tente novamente.",
        )

    messages = [
        {"role": "system", "content": request.system_prompt},
        {"role": "user", "content": request.user_prompt},
    ]
    try:
        response = await client.chat(model=settings.ollama_model, messages=messages)
        content = getattr(getattr(response, "message", None), "content", "") or ""
        return content
    except ConnectionError as e:
        logger.warning("Ollama inacessível (connection error): %s", e)
        raise HTTPException(
            status_code=503,
            detail="O Ollama não está rodando. Inicie o Ollama (ex.: ollama serve) e tente novamente.",
        ) from e
    except ollama.ResponseError as e:
        logger.error("Resposta de erro do Ollama: %s", e)
        raise HTTPException(status_code=502, detail=str(e)) from e
    except Exception as e:
        err_msg = str(e).lower()
        if "connection" in err_msg or "refused" in err_msg or "connect" in err_msg:
            logger.warning("Ollama inacessível: %s", e)
            raise HTTPException(
                status_code=503,
                detail="O Ollama não está rodando. Inicie o Ollama (ex.: ollama serve) e tente novamente.",
            ) from e
        logger.exception("Erro ao chamar Ollama: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/")
def health_check() -> dict[str, str]:
    return {
        "status": "online",
        "model": settings.model_embedding,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
