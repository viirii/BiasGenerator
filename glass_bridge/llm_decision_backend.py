"""LLM-based decision backends for Glass Bridge tournament agents.

When an agent's strategy_profile has model_name not in (None, "none", ""),
the policy delegates to an LLM backend instead of the heuristic.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from typing import Any

_transformers_available: bool | None = None


def _check_transformers() -> bool:
    global _transformers_available
    if _transformers_available is not None:
        return _transformers_available
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
        _transformers_available = True
    except ImportError:
        _transformers_available = False
    return _transformers_available


class LLMDecisionBackend(ABC):
    @abstractmethod
    def select_action(
        self,
        observation: dict[str, Any],
        strategy_profile: dict[str, Any],
        legal_actions: list[Any],
        fallback_fn: Any,
    ) -> Any:
        ...


def _observation_to_prompt(observation: dict[str, Any]) -> str:
    """No reputation—model infers trust from game state."""
    parts = [
        f"Phase: {observation.get('phase')}",
        f"Round: {observation.get('round_idx')}",
        f"You are agent {observation.get('agent_name')}",
        f"Active agents: {observation.get('active_agents', [])}",
        f"Current order: {observation.get('current_order', [])}",
    ]
    round_history = observation.get("round_history", [])
    if round_history:
        parts.append("Past rounds (order, survivors, eliminated, progress, trade_summary):")
        for r in round_history:
            parts.append(f"  Round {r.get('round_idx')}: order={r.get('order')}, survivors={r.get('survivors')}, eliminated={r.get('eliminated')}, progress={r.get('progress')}, trades={r.get('trade_summary', {})}")
    if observation.get("phase", "").startswith("communication"):
        parts.append(f"Negotiable partners: {observation.get('negotiable_partners', [])}")
        parts.append(f"Your private known steps: {observation.get('private_known_steps', {})}")
        parts.append(f"Assignment by agent: {observation.get('assignment_by_agent', {})}")
        inc = observation.get("incoming_offers", [])
        if inc:
            inc_serial = [{"offer_id": o.get("offer_id"), "proposer": o.get("proposer"), "request_steps": o.get("request_steps", []), "claims": o.get("claims", [])} for o in inc]
            parts.append(f"Incoming offers: {inc_serial}")
    else:
        parts.append(f"Current step index: {observation.get('current_step_idx')}")
        parts.append(f"Verified public: {observation.get('verified_public', [])}")
        parts.append(f"Your private known steps: {observation.get('private_known_steps', {})}")
    parts.append(f"Legal actions: {observation.get('legal_actions', [])}")
    return "\n".join(parts)


def _parse_llm_action(raw: str, phase: str, legal_actions: list[Any]) -> Any | None:
    raw = raw.strip()
    json_match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", raw, re.DOTALL)
    if json_match:
        try:
            parsed = json.loads(json_match.group())
            action_type = str(parsed.get("type", "")).upper()
            if action_type == "OFFERS":
                offers = parsed.get("offers", [])
                if not isinstance(offers, list):
                    return None
                valid_offers = []
                for o in offers:
                    if not isinstance(o, dict):
                        continue
                    r = o.get("recipient")
                    g = o.get("give_steps", [])
                    req = o.get("request_steps", [])
                    mode = o.get("claim_mode", "truth")
                    if r and isinstance(g, list) and isinstance(req, list):
                        valid_offers.append({
                            "recipient": str(r),
                            "give_steps": [int(x) for x in g if isinstance(x, (int, float))],
                            "request_steps": [int(x) for x in req if isinstance(x, (int, float))],
                            "claim_mode": "truth" if str(mode).lower() == "truth" else "lie",
                        })
                if valid_offers:
                    return {"type": "OFFERS", "offers": valid_offers}
                return {"type": "NOOP"}
            if action_type == "RESPONSES":
                ids = parsed.get("accept_offer_ids", [])
                if isinstance(ids, list):
                    return {"type": "RESPONSES", "accept_offer_ids": [int(x) for x in ids if isinstance(x, (int, float))]}
                return {"type": "NOOP"}
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
    if "LEFT" in raw.upper() and "RIGHT" not in raw.upper().split("LEFT")[0]:
        return "LEFT"
    if "RIGHT" in raw.upper():
        return "RIGHT"
    return None


class QwenBackend(LLMDecisionBackend):
    def __init__(self, model_path: str = "unsloth/Qwen2.5-3B-Instruct", device: str | None = None):
        if not _check_transformers():
            raise ImportError("LLM backends require transformers and torch. Install with: pip install transformers torch")
        self._model_path = model_path
        self._device = device
        self._model = None
        self._tokenizer = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self._model_path, trust_remote_code=True)
        device = self._device if self._device else ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = AutoModelForCausalLM.from_pretrained(
            self._model_path,
            torch_dtype="auto",
            trust_remote_code=True,
        )
        self._model = self._model.to(device)
        self._model.eval()
        self._device = device

    def select_action(
        self,
        observation: dict[str, Any],
        strategy_profile: dict[str, Any],
        legal_actions: list[Any],
        fallback_fn: Any,
    ) -> Any:
        phase = observation.get("phase", "")
        if phase not in ("communication_offer", "communication_response", "movement"):
            return fallback_fn()

        self._ensure_loaded()
        import torch

        prompt = _observation_to_prompt(observation)
        if phase == "communication_offer":
            output_format = 'Respond with ONLY valid JSON. Example: {"type":"OFFERS","offers":[{"recipient":"p2","give_steps":[3],"request_steps":[5],"claim_mode":"truth"}]} or {"type":"NOOP"}.'
        elif phase == "communication_response":
            output_format = 'Respond with ONLY valid JSON. Example: {"type":"RESPONSES","accept_offer_ids":[0,1]} or {"type":"NOOP"}.'
        else:
            output_format = 'Respond with ONLY "LEFT" or "RIGHT".'

        user_content = f"{prompt}\n\n{output_format}"
        messages = [
            {"role": "system", "content": (
                "You are an agent in a glass bridge game. Maximize your survival; minimize others'. "
                "You decide how much to trust each agent and whether to negotiate or trade—infer from past rounds, "
                "who survived, who was eliminated, and what trades occurred. No reputation scores are given; use the game state. "
                "Output only the requested JSON or LEFT/RIGHT."
            )},
            {"role": "user", "content": user_content},
        ]
        text = self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self._tokenizer([text], return_tensors="pt").to(self._device)
        with torch.no_grad():
            out = self._model.generate(**inputs, max_new_tokens=256, do_sample=True, temperature=0.3, pad_token_id=self._tokenizer.eos_token_id)
        response = self._tokenizer.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        parsed = _parse_llm_action(response, phase, legal_actions)
        return parsed if parsed is not None else fallback_fn()


_LLM_BACKEND_REGISTRY: dict[str, tuple[type[LLMDecisionBackend], dict[str, Any]]] = {
    "qwen3.5": (QwenBackend, {"model_path": "unsloth/Qwen2.5-3B-Instruct"}),
    "qwen2.5": (QwenBackend, {"model_path": "unsloth/Qwen2.5-3B-Instruct"}),
    "qwen2.5-7b": (QwenBackend, {"model_path": "Qwen/Qwen2.5-7B-Instruct"}),
    "smollm2-1.7b": (QwenBackend, {"model_path": "HuggingFaceTB/SmolLM2-1.7B-Instruct"}),
    "smollm2-360m": (QwenBackend, {"model_path": "HuggingFaceTB/SmolLM2-360M-Instruct"}),
    "smollm2-135m": (QwenBackend, {"model_path": "HuggingFaceTB/SmolLM2-135M-Instruct"}),
}
_backend_cache: dict[str, LLMDecisionBackend] = {}


def get_llm_backend(model_name: str, model_path_override: str | None = None) -> LLMDecisionBackend | None:
    if not model_name or str(model_name).lower() in ("none", "null", ""):
        return None
    key = str(model_name).lower()
    if key not in _LLM_BACKEND_REGISTRY:
        return None
    cache_key = f"{key}:{model_path_override or ''}"
    if cache_key in _backend_cache:
        return _backend_cache[cache_key]
    cls, kwargs = _LLM_BACKEND_REGISTRY[key]
    if model_path_override:
        kwargs = {**kwargs, "model_path": model_path_override}
    try:
        backend = cls(**kwargs)
        _backend_cache[cache_key] = backend
        return backend
    except Exception:
        return None
