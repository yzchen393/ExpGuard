from __future__ import annotations

import httpx


class LanguageModel:
    def __init__(self, endpoint: str, model: str, api_key: str | None, timeout: float = 120):
        self.endpoint = endpoint.rstrip("/") + "/chat/completions"
        self.model = model
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.client = httpx.Client(headers=headers, timeout=timeout)
        self.total_tokens = 0
        self.unreported_requests = 0

    def __enter__(self) -> LanguageModel:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.client.close()

    def complete(self, system: str, user: str, *, temperature: float, seed: int | None = None) -> str:
        request = {
            "model": self.model, "temperature": temperature,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        }
        if seed is not None:
            request["seed"] = seed
        response = self.client.post(self.endpoint, json=request)
        response.raise_for_status()
        result = response.json()
        tokens = (result.get("usage") or {}).get("total_tokens")
        if type(tokens) is int and tokens >= 0:
            self.total_tokens += tokens
        else:
            self.unreported_requests += 1
        content = result["choices"][0]["message"]["content"]
        if not isinstance(content, str) or not content.strip():
            raise ValueError("The model returned no text")
        return content
