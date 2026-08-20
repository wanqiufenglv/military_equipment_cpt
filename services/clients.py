from __future__ import annotations

import base64
import io
import json
import mimetypes
import os
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse, urlunparse

from ..core.io_utils import maybe_json


class RequestsClient:
    def __init__(self) -> None:
        try:
            import requests  # type: ignore
        except ImportError as exc:
            raise RuntimeError("Please install requests to call MinerU or model services.") from exc
        self.requests = requests


class MinerUClient(RequestsClient):
    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__()
        self.cfg = cfg

    def _endpoint(self, route: str) -> str:
        parsed = urlparse(str(self.cfg["url"]))
        path = parsed.path.rstrip("/")
        for suffix in ("/file_parse", "/url_parse"):
            if path.endswith(suffix):
                path = path[: -len(suffix)]
        return urlunparse(parsed._replace(path=f"{path}/{route}".replace("//", "/"), params="", query="", fragment=""))

    def parse_pdf(self, pdf_path: Path) -> Any:
        data: list[tuple[str, str]] = [
            ("backend", str(self.cfg["backend"])),
            ("parse_method", str(self.cfg["parse_method"])),
            ("formula_enable", str(bool(self.cfg["formula_enable"])).lower()),
            ("table_enable", str(bool(self.cfg["table_enable"])).lower()),
            ("return_md", str(bool(self.cfg["return_md"])).lower()),
            ("return_middle_json", str(bool(self.cfg["return_middle_json"])).lower()),
            ("return_content_list", str(bool(self.cfg["return_content_list"])).lower()),
            ("return_images", str(bool(self.cfg["return_images"])).lower()),
            ("response_format_zip", str(bool(self.cfg["response_format_zip"])).lower()),
            ("return_original_file", str(bool(self.cfg["return_original_file"])).lower()),
        ]
        for lang in self.cfg["lang_list"]:
            data.append(("lang_list", str(lang)))
        with pdf_path.open("rb") as handle:
            files = [("files", (pdf_path.name, handle, "application/pdf"))]
            response = self.requests.post(self._endpoint("file_parse"), files=files, data=data, timeout=int(self.cfg["timeout"]))
        response.raise_for_status()
        return response.json()


@dataclass(frozen=True)
class ModelResponse:
    text: str
    provider_name: str
    model: str


class ModelClient(RequestsClient):
    def __init__(self, cfg: dict[str, Any], prompts: dict[str, str]) -> None:
        super().__init__()
        self.cfg = cfg
        self.prompts = prompts
        self.name = str(cfg.get("name") or cfg.get("model") or cfg.get("url") or "model")
        self._rate_lock = threading.Lock()
        self._last_request_at = 0.0

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        api_key_env = str(self.cfg.get("api_key_env") or "").strip()
        api_key = str(self.cfg.get("api_key") or os.getenv(api_key_env, "") or "").strip()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    def _compress_image(self, path: Path, image_bytes: bytes) -> bytes:
        try:
            from PIL import Image  # type: ignore
        except ImportError:
            return image_bytes
        max_side = int(self.cfg.get("max_image_side") or 1600)
        quality = int(self.cfg.get("image_jpeg_quality") or 85)
        with Image.open(path) as image:
            image = image.convert("RGB")
            image.thumbnail((max_side, max_side))
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=quality, optimize=True)
            return buffer.getvalue()

    def _image_to_data_url(self, path: Path) -> str:
        image_bytes = path.read_bytes()
        max_bytes = int(self.cfg.get("max_image_bytes") or 0)
        if max_bytes and len(image_bytes) > max_bytes:
            image_bytes = self._compress_image(path, image_bytes)
            mime = "image/jpeg"
        else:
            mime = mimetypes.guess_type(path.name)[0] or "image/png"
        return f"data:{mime};base64,{base64.b64encode(image_bytes).decode('ascii')}"

    def _apply_rate_limit(self) -> None:
        min_interval = float(self.cfg.get("min_interval_seconds") or 0.0)
        if min_interval <= 0:
            return
        with self._rate_lock:
            now = time.monotonic()
            delay = self._last_request_at + min_interval - now
            if delay > 0:
                time.sleep(delay)
            self._last_request_at = time.monotonic()

    def _extract_message(self, payload: Any) -> str:
        content = payload["choices"][0].get("message", {}).get("content", "")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            return "\n".join(str(item.get("text", "")) for item in content if isinstance(item, dict)).strip()
        return str(content).strip()

    def _extract_stream(self, response: Any) -> str:
        parts: list[str] = []
        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue
            text = str(line).strip()
            if text.startswith("data:"):
                text = text[5:].strip()
            if not text or text == "[DONE]":
                continue
            payload = maybe_json(text)
            if isinstance(payload, dict):
                choice = payload.get("choices", [{}])[0]
                delta = choice.get("delta", {}) if isinstance(choice, dict) else {}
                parts.append(str(delta.get("content", "")) if isinstance(delta, dict) else "")
        return "".join(parts).strip()

    def chat(self, prompt: str, images: list[Path] | None = None) -> str:
        content: str | list[dict[str, Any]]
        if images:
            content = [{"type": "text", "text": prompt}]
            for image in images:
                content.append({"type": "image_url", "image_url": {"url": self._image_to_data_url(image)}})
        else:
            content = prompt
        payload: dict[str, Any] = {
            "model": self.cfg["model"],
            "messages": [
                {"role": "system", "content": self.prompts["system"]},
                {"role": "user", "content": content},
            ],
            "temperature": self.cfg.get("temperature", 0.4),
            "stream": bool(self.cfg.get("stream", False)),
        }
        if self.cfg.get("max_tokens") is not None:
            payload["max_tokens"] = self.cfg["max_tokens"]
        if isinstance(self.cfg.get("chat_template_kwargs"), dict):
            payload["chat_template_kwargs"] = self.cfg["chat_template_kwargs"]
        self._apply_rate_limit()
        response = self.requests.post(
            self.cfg["url"],
            json=payload,
            headers=self._headers(),
            timeout=int(self.cfg["timeout"]),
            stream=bool(self.cfg.get("stream", False)),
        )
        response.raise_for_status()
        return self._extract_stream(response) if self.cfg.get("stream") else self._extract_message(response.json())


class _PooledModelClient:
    def __init__(self, client: ModelClient, cfg: dict[str, Any]) -> None:
        self.client = client
        self.cfg = cfg
        self.name = client.name
        self.model = str(cfg.get("model") or "")
        self.weight = max(1, int(cfg.get("weight") or 1))
        self.max_concurrency = max(1, int(cfg.get("max_concurrency") or 1))
        self.task_types = {str(item) for item in cfg.get("task_types", []) if str(item)}
        self.capabilities = {str(item) for item in cfg.get("capabilities", []) if str(item)}
        self._semaphore = threading.BoundedSemaphore(self.max_concurrency)
        self._active = 0
        self._cooldown_until = 0.0
        self._lock = threading.Lock()

    @property
    def active(self) -> int:
        with self._lock:
            return self._active

    def supports(self, task_type: str, has_images: bool) -> bool:
        if self.cfg.get("enabled") is False:
            return False
        if self.task_types and task_type not in self.task_types:
            return False
        if has_images and "image" not in self.capabilities:
            return False
        return True

    def is_cooling_down(self, now: float) -> bool:
        with self._lock:
            return self._cooldown_until > now

    def mark_failure(self, cooldown_seconds: float, now: float) -> None:
        if cooldown_seconds <= 0:
            return
        with self._lock:
            self._cooldown_until = max(self._cooldown_until, now + cooldown_seconds)

    def mark_success(self) -> None:
        with self._lock:
            self._cooldown_until = 0.0

    def chat(self, prompt: str, images: list[Path] | None = None) -> ModelResponse:
        self._semaphore.acquire()
        with self._lock:
            self._active += 1
        try:
            return ModelResponse(self.client.chat(prompt, images), self.name, self.model)
        finally:
            with self._lock:
                self._active -= 1
            self._semaphore.release()


class ModelPool:
    def __init__(self, clients: list[_PooledModelClient], fallback_enabled: bool = True, max_attempts: int = 2, cooldown_seconds: float = 300.0) -> None:
        if not clients:
            raise ValueError("Model pool requires at least one provider.")
        self.clients = clients
        self.fallback_enabled = fallback_enabled
        self.max_attempts = max(1, max_attempts)
        self.cooldown_seconds = max(0.0, cooldown_seconds)
        self._rr_index = 0
        self._lock = threading.Lock()

    @classmethod
    def from_config(cls, cfg: dict[str, Any], prompts: dict[str, str]) -> "ModelPool":
        pool_cfg = cfg.get("vlm_pool") if isinstance(cfg.get("vlm_pool"), dict) else {}
        provider_defaults = pool_cfg.get("provider_defaults") if isinstance(pool_cfg.get("provider_defaults"), dict) else {}
        providers = pool_cfg.get("providers") if isinstance(pool_cfg.get("providers"), list) else []
        clients: list[_PooledModelClient] = []
        for index, provider in enumerate(providers, start=1):
            if not isinstance(provider, dict) or provider.get("enabled") is False:
                continue
            provider_cfg = dict(provider_defaults)
            provider_cfg.update(provider)
            provider_cfg.setdefault("name", f"model_{index}")
            provider_cfg.setdefault("capabilities", ["text", "image"])
            clients.append(_PooledModelClient(ModelClient(provider_cfg, prompts), provider_cfg))
        fallback_cfg = pool_cfg.get("fallback") if isinstance(pool_cfg.get("fallback"), dict) else {}
        return cls(
            clients,
            fallback_enabled=bool(fallback_cfg.get("enabled", True)),
            max_attempts=int(fallback_cfg.get("max_attempts", 2)),
            cooldown_seconds=float(fallback_cfg.get("cooldown_seconds", 300.0)),
        )

    def _ranked_candidates(self, task_type: str, has_images: bool) -> list[_PooledModelClient]:
        now = time.monotonic()
        candidates = [client for client in self.clients if client.supports(task_type, has_images)]
        if not candidates:
            raise RuntimeError(f"No model provider is configured for task_type={task_type}.")
        available = [client for client in candidates if not client.is_cooling_down(now)]
        candidates = available or candidates
        with self._lock:
            start = self._rr_index
            self._rr_index = (self._rr_index + 1) % max(1, len(candidates))
        indexed = list(enumerate(candidates))
        indexed.sort(key=lambda item: (item[1].active / item[1].max_concurrency, ((item[0] - start) % len(candidates)) / item[1].weight))
        return [client for _, client in indexed]

    def chat(self, task_type: str, prompt: str, images: list[Path] | None = None) -> ModelResponse:
        candidates = self._ranked_candidates(task_type, bool(images))
        attempts = min(len(candidates), self.max_attempts) if self.fallback_enabled else 1
        last_error: Exception | None = None
        for client in candidates[:attempts]:
            try:
                response = client.chat(prompt, images)
                client.mark_success()
                return response
            except Exception as exc:
                last_error = exc
                client.mark_failure(self.cooldown_seconds, time.monotonic())
                if not self.fallback_enabled:
                    break
        if last_error:
            raise last_error
        raise RuntimeError(f"No model provider attempted for task_type={task_type}.")


def _strip_json_fence(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _extract_json(text: str) -> str:
    text = _strip_json_fence(text)
    starts = [idx for idx in (text.find("["), text.find("{")) if idx >= 0]
    if not starts:
        return text
    start = min(starts)
    opening = text[start]
    closing = "]" if opening == "[" else "}"
    in_string = False
    escaped = False
    depth = 0
    for index in range(start, len(text)):
        char = text[index]
        if escaped:
            escaped = False
            continue
        if char == "\\" and in_string:
            escaped = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == opening:
            depth += 1
        elif char == closing:
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return text[start:]


def parse_json_objects(text: str) -> list[dict[str, Any]]:
    raw = _extract_json(text)
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        fixed = re.sub(r",(\s*[\]}])", r"\1", raw)
        try:
            payload = json.loads(fixed)
        except json.JSONDecodeError:
            raise ValueError(f"Model response is not valid JSON: {exc}") from exc
    if isinstance(payload, dict):
        return [payload]
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    raise ValueError("Model response must be a JSON object or array.")

