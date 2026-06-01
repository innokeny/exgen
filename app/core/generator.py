from __future__ import annotations

import os
import threading
import time
from typing import Optional

import structlog
import torch

from app.core.model_manager import LoadedModel, ModelManager
from app.prompts.templates import build_messages

log = structlog.get_logger(__name__)


class LLMGenerator:
    def __init__(self, manager: ModelManager, max_new_tokens: int = 2048):
        self._manager = manager
        self._max_new_tokens = max_new_tokens
        self._lock = threading.Lock()

    def generate_raw(
        self,
        *,
        user_id: str,
        message_content: str,
        grammar_error: str,
        explanation: str,
        llm_confidence: float,
        task_type: str,
        model_key: Optional[str] = None,
    ) -> tuple[str, str, int]:
        loaded = self._manager.get(model_key)
        messages = build_messages(
            user_id=user_id,
            message_content=message_content,
            grammar_error=grammar_error,
            explanation=explanation,
            llm_confidence=llm_confidence,
            task_type=task_type,
        )

        t0 = time.perf_counter()
        wait_t0 = time.perf_counter()
        with self._lock:
            wait_ms = int((time.perf_counter() - wait_t0) * 1000)
            try:
                text, n_in, n_out = self._run(loaded, messages)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                log.error("llm.generate.oom", user_id=user_id)
                raise
            except RuntimeError as e:
                log.error(
                    "llm.generate.cuda_error",
                    user_id=user_id,
                    error=repr(e),
                    note="CUDA context likely poisoned; service restart required",
                )
                raise
        elapsed_ms = int((time.perf_counter() - t0) * 1000)

        log.info(
            "llm.generate.done",
            model_key=loaded.key,
            elapsed_ms=elapsed_ms,
            wait_ms=wait_ms,
            input_tokens=n_in,
            output_tokens=n_out,
            output_chars=len(text),
        )
        return text, loaded.key, elapsed_ms

    def _run(
        self, loaded: LoadedModel, messages: list[dict]
    ) -> tuple[str, int, int]:
        tokenizer = loaded.tokenizer
        model = loaded.model

        pad_token_id = tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = tokenizer.eos_token_id
            if pad_token_id is None:
                raise RuntimeError(
                    "tokenizer has neither pad_token_id nor eos_token_id"
                )

        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=2048
        )
        n_in = int(inputs["input_ids"].shape[1])
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with self._lock, torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=self._max_new_tokens,
                do_sample=False,
                pad_token_id=pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        new_tokens = output_ids[0][n_in:]
        n_out = int(new_tokens.shape[0])
        text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        return text, n_in, n_out