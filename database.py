"""
MongoDB persistence layer for saved filters and generation sessions.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import List, Optional

import config  # noqa: F401
from pymongo import ASCENDING, DESCENDING, MongoClient
from pymongo.collection import Collection
from pymongo.database import Database


DEFAULT_MONGODB_URI = os.environ.get("MONGODB_URI", "mongodb://localhost:27017")
DEFAULT_DATABASE_NAME = os.environ.get("MONGODB_DATABASE", "artcam")


class FilterDatabase:
    """MongoDB wrapper for filter storage and generation audit records."""

    def __init__(
        self,
        mongodb_uri: str = DEFAULT_MONGODB_URI,
        database_name: str = DEFAULT_DATABASE_NAME,
    ) -> None:
        self._client = MongoClient(
            mongodb_uri,
            serverSelectionTimeoutMS=5000,
            connectTimeoutMS=5000,
            socketTimeoutMS=5000,
            retryWrites=True,
        )
        self._db: Database = self._client[database_name]
        self._filters: Collection = self._db["filters"]
        self._generations: Collection = self._db["generation_sessions"]
        self._initialized = False

    def save_filter(self, name: str, code: str) -> None:
        self._ensure_ready()
        now = self._utc_now()
        self._filters.update_one(
            {"name": name},
            {
                "$set": {
                    "name": name,
                    "code": code,
                    "updated_at": now,
                },
                "$setOnInsert": {
                    "created_at": now,
                },
            },
            upsert=True,
        )

    def delete_filter(self, name: str) -> None:
        self._ensure_ready()
        self._filters.delete_one({"name": name})

    def get_filter_code(self, name: str) -> Optional[str]:
        self._ensure_ready()
        document = self._filters.find_one({"name": name}, {"_id": 0, "code": 1})
        return None if document is None else document["code"]

    def list_filters(self) -> List[str]:
        self._ensure_ready()
        cursor = self._filters.find({}, {"_id": 0, "name": 1}).sort("name", ASCENDING)
        return [document["name"] for document in cursor]

    def get_all_filters(self) -> List[dict]:
        self._ensure_ready()
        cursor = self._filters.find(
            {},
            {"_id": 0, "name": 1, "code": 1, "created_at": 1, "updated_at": 1},
        ).sort("name", ASCENDING)
        return [self._serialize_document(document) for document in cursor]

    def save_generation_session(self, result: dict) -> None:
        self._ensure_ready()
        document = dict(result)
        document["updated_at"] = self._utc_now()
        document.setdefault("created_at", document["updated_at"])
        self._generations.update_one(
            {"session_id": document["session_id"]},
            {"$set": document},
            upsert=True,
        )

    def list_generation_sessions(self, limit: int = 25) -> List[dict]:
        self._ensure_ready()
        cursor = self._generations.find({}, {"_id": 0}).sort("created_at", DESCENDING).limit(limit)
        return [self._serialize_document(document) for document in cursor]

    def ping(self) -> bool:
        self._client.admin.command("ping")
        return True

    def _ensure_ready(self) -> None:
        if self._initialized:
            return

        self.ping()
        self._filters.create_index([("name", ASCENDING)], unique=True)
        self._generations.create_index([("session_id", ASCENDING)], unique=True)
        self._generations.create_index([("created_at", DESCENDING)])
        self._initialized = True

    @staticmethod
    def _utc_now() -> datetime:
        return datetime.now(timezone.utc)

    def _serialize_document(self, document: dict) -> dict:
        payload = dict(document)
        for key in ("created_at", "updated_at"):
            value = payload.get(key)
            if isinstance(value, datetime):
                payload[key] = value.isoformat()
        return payload
