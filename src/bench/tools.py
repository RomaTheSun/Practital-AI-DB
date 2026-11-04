"""Інструменти для циклу (крок 05): Pydantic-аргументи й строгі JSON Schema."""
import json
import re
from dataclasses import dataclass
from typing import Callable

from pydantic import BaseModel, ConfigDict, ValidationError

from bench.clients import to_strict_schema


WORKLOADS = {
    "W-101": {
        "dominant_operation": "атомарно змінювати замовлення, оплату і залишок",
        "signals": ["ACID", "foreign_keys", "JOIN", "audit"],
    },
    "W-102": {
        "dominant_operation": "читати цілі товарні картки з різними наборами полів",
        "signals": ["variable_schema", "whole_document"],
    },
    "W-103": {
        "dominant_operation": "інкрементувати хвилинні лічильники за ключем",
        "signals": ["TTL", "atomic_counter", "low_latency"],
    },
    "W-104": {
        "dominant_operation": "шукати найкоротші шляхи і сусідів кількох рівнів",
        "signals": ["multi_hop", "path_search"],
    },
    "W-105": {
        "dominant_operation": "читати цілі конфігурації з мінливою схемою",
        "signals": ["variable_schema", "whole_document"],
    },
    "W-106": {
        "dominant_operation": "читати короткоживучі стани за ключем",
        "signals": ["TTL", "key_value", "low_latency"],
    },
}

REQUIREMENTS = {
    "R-205": {
        "mandatory": "пов'язані зміни мають бути однією ACID-транзакцією з аудитом",
        "signals": ["ACID", "audit", "foreign_keys"],
    },
    "R-206": {
        "mandatory": "потрібні рекурсивні обходи залежностей і пошук шляхів",
        "signals": ["multi_hop", "path_search", "dependency_graph"],
    },
}


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    args_model: type[BaseModel]
    fn: Callable[[BaseModel], str]


class LookupWorkloadArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    code: str


class LookupRequirementArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    code: str


def lookup_workload(args: LookupWorkloadArgs) -> str:
    """Повертає синтетичний профіль навантаження за кодом W-123."""
    if not re.fullmatch(r"W-\d{3}", args.code):
        raise ValueError("код навантаження має формат W-123")
    profile = WORKLOADS.get(args.code)
    return json.dumps(
        {"code": args.code, "found": profile is not None, "profile": profile},
        ensure_ascii=False,
    )


def lookup_requirement(args: LookupRequirementArgs) -> str:
    """Повертає обов'язкову вимогу за кодом R-123."""
    if not re.fullmatch(r"R-\d{3}", args.code):
        raise ValueError("код вимоги має формат R-123")
    requirement = REQUIREMENTS.get(args.code)
    return json.dumps(
        {"code": args.code, "found": requirement is not None, "requirement": requirement},
        ensure_ascii=False,
    )


TOOLS: dict[str, Tool] = {
    tool.name: tool
    for tool in (
        Tool(
            "lookup_workload",
            "Синтетичний профіль навантаження за кодом формату W-123.",
            LookupWorkloadArgs,
            lookup_workload,
        ),
        Tool(
            "lookup_requirement",
            "Обов'язкова вимога до сховища за кодом формату R-123.",
            LookupRequirementArgs,
            lookup_requirement,
        ),
    )
}


def tool_specs(names: list[str]) -> list[dict]:
    """Специфікації у форматі OpenAI function tools."""
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": TOOLS[name].description,
                "parameters": to_strict_schema(TOOLS[name].args_model),
                "strict": True,
            },
        }
        for name in names
    ]


def execute(name: str, arguments_raw: str) -> tuple[str, str | None]:
    """Виконує інструмент; помилку повертає текстом для моделі, а не винятком."""
    tool = TOOLS.get(name)
    if tool is None:
        return f"unknown tool: {name}", "unknown_tool"
    try:
        args = tool.args_model.model_validate_json(arguments_raw or "{}")
    except ValidationError as error:
        return f"bad arguments: {error.errors(include_url=False)}", "bad_args"
    try:
        return tool.fn(args), None
    except Exception as error:  # noqa: BLE001
        return f"tool error: {error}", "tool_error"
