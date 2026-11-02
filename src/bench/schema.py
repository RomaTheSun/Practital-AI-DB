"""Строга схема відповіді задачі вибору оперативного сховища (крок 04)."""
from typing import Literal

from pydantic import BaseModel, ConfigDict


class StorageLabel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: Literal["postgresql", "mongodb", "redis", "neo4j"]
