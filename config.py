from pydantic import BaseModel
from dotenv import load_dotenv
import os

load_dotenv()

class Settings(BaseModel):
    qdrant_url: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    embed_dim: int = int(os.getenv("EMBED_DIM", "384"))
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    mem0_collection_prefix: str = os.getenv("MEM0_COLLECTION_PREFIX", "ltm")

settings = Settings()
