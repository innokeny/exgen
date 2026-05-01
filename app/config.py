from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Dict, List

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# Service metadata, surfaced via GET /api/v1/models so SAYIT can discover us.
SERVICE_VERSION = "1.0.0"


# Mapping of short model keys (used in API requests) → HF Hub repo IDs.
# Adapter directories are resolved as: ADAPTER_DIR / <model_key>.
SUPPORTED_MODELS: Dict[str, str] = {
    "qwen2.5-3b": "Qwen/Qwen2.5-3B-Instruct",
    "smollm3-3b": "HuggingFaceTB/SmolLM3-3B",
}


# Static evaluation metrics from the thesis — published via /api/v1/models so
# the SAYIT backend can decide which model to route to. Keep keys aligned with
# SUPPORTED_MODELS.
MODEL_METRICS: Dict[str, Dict[str, float]] = {
    "qwen2.5-3b": {
        "tr_composite": 0.938,
        "bertscore_f1": 0.9085,
        "perplexity": 1.22,
    },
}


SUPPORTED_TASK_TYPES: list[str] = [
    "grammar_choice",
    "vocabulary_fill",
    "matching",
    "transformation",
    "categories",
]


FALLBACK_METHODS: list[str] = ["fill_in_blanks", "reconstruction"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Default model used when client doesn't specify one.
    model_name: str = "qwen2.5-3b"

    # Root folder under which each model has its own adapter subfolder.
    adapter_dir: Path = Path("adapters")

    # HuggingFace cache (base weights live here after first download).
    hf_home: Path = Path("/app/models_cache")

    # Logging
    log_level: str = "INFO"

    # Server
    host: str = "0.0.0.0"
    port: int = 8000

    # Generation
    max_new_tokens: int = 2048
    generation_timeout_s: int = 120

    # CORS
    cors_origins: str = "*"

    # Force CPU even if CUDA is available — for local debugging only.
    force_cpu: bool = False

    # Optional HF token for gated/private models.
    hf_token: str | None = None

    # ---- SAYIT integration -------------------------------------------------
    # Service identity reported via /api/v1/models — used by SAYIT for
    # discovery / health gating before routing traffic away from Grok.
    service_name: str = "exercise-generator"

    # When true, the service expects to be reachable on the shared SAYIT
    # docker network and may opt into stricter response shapes / callbacks.
    sayit_integration: bool = False

    # Reserved for future server-initiated callbacks (e.g. async batch jobs).
    sayit_backend_url: str = "http://backend:8000"

    @field_validator("model_name")
    @classmethod
    def _validate_model_name(cls, v: str) -> str:
        if v not in SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported model '{v}'. Allowed: {list(SUPPORTED_MODELS)}"
            )
        return v

    @property
    def cors_origins_list(self) -> List[str]:
        raw = self.cors_origins.strip()
        if raw == "*" or raw == "":
            return ["*"]
        return [o.strip() for o in raw.split(",") if o.strip()]

    def adapter_path_for(self, model_key: str) -> Path:
        return self.adapter_dir / model_key

    def base_model_id(self, model_key: str) -> str:
        return SUPPORTED_MODELS[model_key]


@lru_cache
def get_settings() -> Settings:
    return Settings()
