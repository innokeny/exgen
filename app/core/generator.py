from __future__ import annotations

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
        """Run inference. Returns (raw_text, model_used, generation_time_ms)."""
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
        text = self._run(loaded, messages)
        elapsed_ms = int((time.perf_counter() - t0) * 1000)

        log.info(
            "llm.generate.done",
            model_key=loaded.key,
            elapsed_ms=elapsed_ms,
            output_chars=len(text),
        )
        return text, loaded.key, elapsed_ms

    def _run(self, loaded: LoadedModel, messages: list[dict]) -> str:
        tokenizer = loaded.tokenizer
        model = loaded.model

        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=2048
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=self._max_new_tokens,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.pad_token_id,
            )

        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        return tokenizer.decode(new_tokens, skip_special_tokens=True)
