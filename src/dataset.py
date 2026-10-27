"""
Dataset loading, schema models, and validation logic.
Supports loading tool definitions and benchmark evaluation tasks.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from src.config import DATA_DIR


class ToolParameter(BaseModel):
    """Schema for a single tool parameter or parameter object."""
    type: str
    description: Optional[str] = None
    enum: Optional[List[str]] = None
    items: Optional[Dict[str, Any]] = None


class ToolParameters(BaseModel):
    """Schema for tool parameters dictionary."""
    type: str = "OBJECT"
    properties: Dict[str, Any] = Field(default_factory=dict)
    required: List[str] = Field(default_factory=list)


class ToolDefinition(BaseModel):
    """Standardized representation of a harmless tool in the toolkit."""
    name: str
    description: str
    parameters: ToolParameters
    is_distractor: bool = False
    target_counterpart: Optional[str] = None
    domain: str = "general"

    def to_gemini_schema(self) -> Dict[str, Any]:
        """Convert to Gemini API function_declaration format."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters.model_dump(exclude_none=True)
        }


class Task(BaseModel):
    """Single benchmark tool-calling task."""
    task_id: str
    original_query: str
    paraphrased_query: str
    expected_tool: Optional[str] = None
    expected_tool_name: Optional[str] = None
    expected_arguments: Dict[str, Any] = Field(default_factory=dict)
    tool_definitions: List[str] = Field(default_factory=list)
    task_category: Optional[str] = None
    domain: Optional[str] = "general"
    difficulty_category: Optional[str] = "medium"
    data_source: Optional[str] = "custom_curated_pilot"
    source: Optional[str] = "custom_curated_pilot"
    license_or_provenance: str = "CC BY 4.0"
    validation_status: str = "human_verified"

    def model_post_init(self, __context: Any) -> None:
        # Reconcile expected_tool and expected_tool_name
        if not self.expected_tool_name and self.expected_tool:
            self.expected_tool_name = self.expected_tool
        elif not self.expected_tool and self.expected_tool_name:
            self.expected_tool = self.expected_tool_name
        # Reconcile domain and task_category
        if not self.domain and self.task_category:
            self.domain = self.task_category
        elif not self.task_category and self.domain:
            self.task_category = self.domain
        # Reconcile source and data_source
        if not self.data_source and self.source:
            self.data_source = self.source
        elif not self.source and self.data_source:
            self.source = self.data_source


def load_tools(path: Optional[Path] = None) -> List[ToolDefinition]:
    """Load tool definitions from JSON file."""
    file_path = path or (DATA_DIR / "tool_definitions.json")
    if not file_path.exists():
        raise FileNotFoundError(f"Tool definitions not found at: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return [ToolDefinition(**item) for item in data]


load_tool_definitions = load_tools


def load_tasks(path: Optional[Path] = None, limit: Optional[int] = None) -> List[Task]:
    """Load benchmark tasks from JSON file."""
    file_path = path or (DATA_DIR / "tasks.json")
    if not file_path.exists():
        raise FileNotFoundError(f"Tasks file not found at: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    tasks = [Task(**item) for item in data]
    if limit and limit > 0:
        tasks = tasks[:limit]
    return tasks


def validate_dataset(
    tools: Optional[List[ToolDefinition]] = None,
    tasks: Optional[List[Task]] = None
) -> Dict[str, Any]:
    """
    Validate that all tasks reference existing tools and valid arguments.
    Returns validation report dict.
    """
    tools = tools or load_tools()
    tasks = tasks or load_tasks()

    tool_map = {t.name: t for t in tools}
    errors = []
    warnings = []

    for task in tasks:
        # Check tool existence
        if task.expected_tool_name not in tool_map:
            errors.append(
                f"Task {task.task_id}: expected tool '{task.expected_tool_name}' not in toolkit."
            )
            continue

        tool = tool_map[task.expected_tool_name]

        # Check required arguments
        props = tool.parameters.properties
        req = tool.parameters.required
        for r in req:
            if r not in task.expected_arguments:
                errors.append(
                    f"Task {task.task_id}: missing required argument '{r}' in expected_arguments."
                )

        # Check argument types
        for arg_key, arg_val in task.expected_arguments.items():
            if arg_key not in props:
                warnings.append(
                    f"Task {task.task_id}: argument '{arg_key}' not in tool properties schema."
                )

    return {
        "valid": len(errors) == 0,
        "tasks_count": len(tasks),
        "tools_count": len(tools),
        "target_tools_count": sum(1 for t in tools if not t.is_distractor),
        "distractor_tools_count": sum(1 for t in tools if t.is_distractor),
        "errors": errors,
        "warnings": warnings,
    }


# Safe local mock tool implementations for functional execution and testing
MOCK_TOOL_REGISTRY = {
    "get_weather": lambda city, units="celsius": {
        "city": city,
        "temperature": 18.5,
        "units": units,
        "condition": "Partly Cloudy"
    },
    "calculate": lambda expression: {
        "expression": expression,
        "result": eval(expression, {"__builtins__": None}, {})  # safe simple arithmetic
    },
    "search_documents": lambda query, max_results=5: {
        "query": query,
        "results": [f"Document matching '{query}' - Section 1", f"Doc {query} - Reference 2"][:max_results]
    },
    "get_stock_price": lambda symbol: {
        "symbol": symbol.upper(),
        "price": 182.40,
        "currency": "USD"
    },
    "convert_currency": lambda amount, from_currency, to_currency: {
        "converted_amount": round(float(amount) * 0.92, 2),
        "from": from_currency,
        "to": to_currency
    },
    "get_calendar_events": lambda date: {
        "date": date,
        "events": ["AI Research Sync at 10:00 AM", "Lab Meeting at 2:00 PM"]
    },
    "calculate_shipping": lambda weight_kg, distance_km: {
        "weight_kg": weight_kg,
        "distance_km": distance_km,
        "cost_usd": round(10.0 + (float(weight_kg) * 1.5) + (float(distance_km) * 0.05), 2)
    },
    "find_restaurant": lambda cuisine, location, price_range="$$": {
        "cuisine": cuisine,
        "location": location,
        "price_range": price_range,
        "recommendation": f"Trattoria Bella in {location}"
    },
    "get_order_status": lambda order_id: {
        "order_id": order_id,
        "status": "In Transit",
        "estimated_delivery": "Tomorrow by 5 PM"
    },
    "search_products": lambda query, category="all": {
        "query": query,
        "category": category,
        "matches": [f"{query} Premium Edition", f"{query} Standard"]
    }
}
