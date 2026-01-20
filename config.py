from pydantic import BaseModel
from dotenv import load_dotenv
import os

load_dotenv()

class Settings(BaseModel):
    qdrant_url: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    embed_dim: int = int(os.getenv("EMBED_DIM", "384"))

settings = Settings()
