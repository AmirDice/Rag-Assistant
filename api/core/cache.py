"""Cache — Tier 1 (memory), Tier 2 (Redis exact), Tier 3 (Redis semantic, WP16 §16.4)."""

from __future__ import annotations

import hashlib
import json
import logging
import math
from typing import Optional

from api.core.settings import get_settings

logger = logging.getLogger(__name__)


def _cosine_sim(a: list[float], b: list[float]) -> float:
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return dot / (na * nb)


def _cache_key(question: str, tenant_id: str) -> str:
    raw = f"{question.strip().lower()}:{tenant_id}"
    return hashlib.sha256(raw.encode()).hexdigest()


class InMemoryCache:
    """Tier 1 — session-scoped, resets on API restart."""

    def __init__(self) -> None:
        self._store: dict[str, str] = {}

    async def get(self, question: str, tenant_id: str) -> Optional[dict]:
        key = _cache_key(question, tenant_id)
        raw = self._store.get(key)
        return json.loads(raw) if raw else None

    async def set(self, question: str, tenant_id: str, value: dict) -> None:
        key = _cache_key(question, tenant_id)
        self._store[key] = json.dumps(value, ensure_ascii=False)

    async def clear(self) -> None:
        self._store.clear()

    async def close(self) -> None:
        pass


class RedisCache:
    """Tier 2 — survives API restarts. TTL-based expiry."""

    def __init__(self) -> None:
        self._redis = None
        settings = get_settings()
        self._url = settings.redis_url
        cfg = settings.models_config()
        self._ttl = cfg.get("cache", {}).get("ttl_seconds", 86400)

    async def _get_redis(self):
        if self._redis is None:
            import redis.asyncio as aioredis
            self._redis = aioredis.from_url(self._url, decode_responses=True)
        return self._redis

    async def get(self, question: str, tenant_id: str) -> Optional[dict]:
        try:
            r = await self._get_redis()
            key = f"ragapp:cache:{_cache_key(question, tenant_id)}"
            raw = await r.get(key)
            return json.loads(raw) if raw else None
        except Exception as e:
            logger.warning("Redis cache GET failed: %s", e)
            return None

    async def set(self, question: str, tenant_id: str, value: dict) -> None:
        try:
            r = await self._get_redis()
            key = f"ragapp:cache:{_cache_key(question, tenant_id)}"
            await r.setex(key, self._ttl, json.dumps(value, ensure_ascii=False))
        except Exception as e:
            logger.warning("Redis cache SET failed: %s", e)

    async def clear(self) -> None:
        try:
            r = await self._get_redis()
            keys = []
            async for key in r.scan_iter("ragapp:cache:*"):
                keys.append(key)
            if keys:
                await r.delete(*keys)
        except Exception as e:
            logger.warning("Redis cache CLEAR failed: %s", e)

    async def close(self) -> None:
        if self._redis:
            await self._redis.close()


class SemanticRedisCache:
    """Tier 3 — Redis list of {vec, data}; match if cosine(question, stored) >= threshold."""

    def __init__(self) -> None:
        settings = get_settings()
        self._url = settings.redis_url
        cfg = settings.models_config().get("cache", {})
        self._ttl = int(cfg.get("ttl_seconds", 86400))
        self._max_entries = int(cfg.get("semantic_max_entries", 256))
        self._threshold = float(cfg.get("semantic_min_similarity", 0.97))
        self._redis = None

    async def _get_redis(self):
        if self._redis is None:
            import redis.asyncio as aioredis
            self._redis = aioredis.from_url(self._url, decode_responses=True)
        return self._redis

    async def get(self, question: str, tenant_id: str) -> Optional[dict]:
        try:
            from api.core.embedder import get_embedder

            vec = await get_embedder().embed_query(question)
            r = await self._get_redis()
            key = f"ragapp:sem:{tenant_id}"
            raw_entries = await r.lrange(key, 0, -1)
            for raw in raw_entries:
                entry = json.loads(raw)
                if _cosine_sim(vec, entry.get("vec", [])) >= self._threshold:
                    return entry.get("data")
        except Exception as e:
            logger.warning("Semantic cache GET failed: %s", e)
        return None

    async def set(self, question: str, tenant_id: str, value: dict) -> None:
        try:
            from api.core.embedder import get_embedder

            vec = await get_embedder().embed_query(question)
            r = await self._get_redis()
            key = f"ragapp:sem:{tenant_id}"
            blob = json.dumps({"vec": vec, "data": value}, ensure_ascii=False)
            await r.lpush(key, blob)
            await r.ltrim(key, 0, self._max_entries - 1)
            await r.expire(key, self._ttl)
        except Exception as e:
            logger.warning("Semantic cache SET failed: %s", e)

    async def clear(self) -> None:
        try:
            r = await self._get_redis()
            keys = []
            async for key in r.scan_iter("ragapp:sem:*"):
                keys.append(key)
            if keys:
                await r.delete(*keys)
        except Exception as e:
            logger.warning("Semantic cache CLEAR failed: %s", e)

    async def close(self) -> None:
        if self._redis:
            await self._redis.close()


_cache_instance = None


async def get_cache() -> InMemoryCache | RedisCache | SemanticRedisCache:
    global _cache_instance
    if _cache_instance is not None:
        return _cache_instance

    settings = get_settings()
    cfg = settings.models_config()
    mode = cfg.get("cache", {}).get("mode", "memory")

    if mode == "semantic":
        try:
            cache = SemanticRedisCache()
            r = await cache._get_redis()
            await r.ping()
            _cache_instance = cache
            logger.info("Cache: Tier 3 (semantic / Redis)")
            return _cache_instance
        except Exception as e:
            logger.warning("Semantic cache needs Redis (%s) — trying exact Redis", e)
            mode = "redis"

    if mode == "redis":
        try:
            cache = RedisCache()
            r = await cache._get_redis()
            await r.ping()
            _cache_instance = cache
            logger.info("Cache: Tier 2 (Redis)")
            return _cache_instance
        except Exception as e:
            logger.warning("Redis not available (%s) — falling back to in-memory", e)

    _cache_instance = InMemoryCache()
    logger.info("Cache: Tier 1 (in-memory)")
    return _cache_instance
