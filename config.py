from pydantic import BaseModel
from dotenv import load_dotenv
import os

load_dotenv()

class Settings(BaseModel):
    qdrant_url: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    embed_dim: int = int(os.getenv("EMBED_DIM", "384"))
    mem0_collection_prefix: str = os.getenv("MEM0_COLLECTION_PREFIX", "ltm")

    # --- Azure OpenAI (primary) ---
    azure_openai_endpoint: str = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    azure_openai_api_key: str = os.getenv("AZURE_OPENAI_API_KEY", "")
    azure_openai_api_version: str = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
    azure_openai_deployment_name: str = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "")
    azure_openai_embedding_deployment: str = os.getenv(
        "AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-small"
    )

    # --- Standard OpenAI (fallback) ---
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    @property
    def use_azure(self) -> bool:
        """True if Azure OpenAI credentials are configured."""
        return bool(self.azure_openai_endpoint and self.azure_openai_api_key)

    @property
    def llm_deployment_or_model(self) -> str:
        """Return the deployment name (Azure) or model name (OpenAI)."""
        if self.use_azure:
            return self.azure_openai_deployment_name
        return self.openai_model

settings = Settings()
