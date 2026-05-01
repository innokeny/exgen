"""Loads base model + LoRA adapter once at startup and serves it.

Singleton — initialized in `app.main` lifespan, accessed via `get_manager()`.
Models are kept in VRAM for the lifetime of the process. Since the service is
synchronous and answers one request at a time, there is no concurrency hazard
around the underlying torch module.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import structlog
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from app.config import Settings, get_settings

log = structlog.get_logger(__name__)


@dataclass
class LoadedModel:
    key: str
    base_model_id: str
    adapter_dir: Path
    model: "torch.nn.Module"
    tokenizer: "AutoTokenizer"
    device: str


class ModelManager:
    """Lazy, thread-safe registry of loaded (base + LoRA) models."""

    def __init__(self, settings: Settings):
        self._settings = settings
        self._models: Dict[str, LoadedModel] = {}
        self._load_lock = threading.Lock()

    @property
    def device(self) -> str:
        if self._settings.force_cpu:
            return "cpu"
        return "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def loaded_keys(self) -> list[str]:
        return list(self._models.keys())

    def get(self, model_key: Optional[str] = None) -> LoadedModel:
        key = model_key or self._settings.model_name
        if key in self._models:
            return self._models[key]
        return self._load(key)

    def preload_default(self) -> None:
        """Force the default model into VRAM so the first request is fast."""
        self._load(self._settings.model_name)

    # ---- internals ----------------------------------------------------------

    def _load(self, model_key: str) -> LoadedModel:
        with self._load_lock:
            if model_key in self._models:
                return self._models[model_key]

            base_id = self._settings.base_model_id(model_key)
            adapter_dir = self._settings.adapter_path_for(model_key)
            if not adapter_dir.exists():
                raise FileNotFoundError(
                    f"Adapter directory not found for '{model_key}': {adapter_dir}"
                )

            log.info(
                "model.load.start",
                model_key=model_key,
                base_model=base_id,
                adapter_dir=str(adapter_dir),
                device=self.device,
            )

            tokenizer = AutoTokenizer.from_pretrained(
                str(adapter_dir),
                trust_remote_code=True,
                token=self._settings.hf_token,
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            dtype = torch.float16 if self.device == "cuda" else torch.float32
            base_model = AutoModelForCausalLM.from_pretrained(
                base_id,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True,
                torch_dtype=dtype,
                token=self._settings.hf_token,
            )
            if self.device == "cpu":
                base_model = base_model.to("cpu")

            model = PeftModel.from_pretrained(base_model, str(adapter_dir))
            model.eval()

            loaded = LoadedModel(
                key=model_key,
                base_model_id=base_id,
                adapter_dir=adapter_dir,
                model=model,
                tokenizer=tokenizer,
                device=self.device,
            )
            self._models[model_key] = loaded
            log.info("model.load.done", model_key=model_key)
            return loaded


_manager: Optional[ModelManager] = None


def init_manager() -> ModelManager:
    global _manager
    if _manager is None:
        _manager = ModelManager(get_settings())
    return _manager


def get_manager() -> ModelManager:
    if _manager is None:
        raise RuntimeError("ModelManager is not initialized. Call init_manager() first.")
    return _manager


def gpu_info() -> dict:
    """Snapshot of CUDA availability and VRAM. Safe to call without any model loaded."""
    if not torch.cuda.is_available():
        return {"available": False, "device_count": 0}

    idx = torch.cuda.current_device()
    free_b, total_b = torch.cuda.mem_get_info(idx)
    used_b = total_b - free_b
    return {
        "available": True,
        "device_count": torch.cuda.device_count(),
        "device_name": torch.cuda.get_device_name(idx),
        "vram_total_mb": int(total_b // (1024 * 1024)),
        "vram_used_mb": int(used_b // (1024 * 1024)),
        "vram_free_mb": int(free_b // (1024 * 1024)),
    }
