"""Thin async HTTP client for the vendored BrowseComp-Plus search server.

Ported from the SUPO paper reference impl's `AsyncSearchClient`. Kept alongside
the rest of the SUPO example so slime does not need to import any external
package to talk to the retrieval server (see `search_server.py` in this dir).
"""

from __future__ import annotations

import asyncio

import httpx


class AsyncSearchClient:
    def __init__(self, base_url: str, timeout: float = 300.0, retries: int = 3, backoff: float = 0.5):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.retries = retries
        self.backoff = backoff
        self._client = httpx.AsyncClient(base_url=self.base_url)

    async def close(self) -> None:
        await self._client.aclose()

    async def _post(self, path: str, payload: dict):
        last_exc: Exception | None = None
        for attempt in range(1, self.retries + 1):
            try:
                r = await self._client.post(path, json=payload, timeout=self.timeout)
                r.raise_for_status()
                data = r.json()
                return data.get("results", data)
            except httpx.HTTPError as e:
                last_exc = e
                if attempt == self.retries:
                    raise
                await asyncio.sleep(self.backoff * attempt)
        raise last_exc

    async def search(self, query: str, k: int = 10):
        return await self._post("/search", {"query": query, "k": k})

    async def open(self, url: str | None = None, docid: str | None = None):
        return await self._post("/open", {"url": url, "docid": docid})
