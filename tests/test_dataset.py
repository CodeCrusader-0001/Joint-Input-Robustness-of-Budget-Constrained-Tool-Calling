"""
Unit tests for dataset loading, schema verification, and mock execution.
"""

import pytest
from src.dataset import (
    MOCK_TOOL_REGISTRY,
    Task,
    ToolDefinition,
    load_tasks,
    load_tools,
    validate_dataset,
)


def test_load_tools():
    tools = load_tools()
    assert len(tools) >= 20
    assert sum(1 for t in tools if not t.is_distractor) >= 10
    assert sum(1 for t in tools if t.is_distractor) >= 10

    # Ensure all target tools have unique names
    names = [t.name for t in tools]
    assert len(names) == len(set(names))


def test_load_tasks():
    tasks = load_tasks()
    assert len(tasks) == 30

    for task in tasks:
        assert task.task_id.startswith("task_")
        assert len(task.original_query) > 10
        assert len(task.paraphrased_query) > 10
        assert task.expected_tool_name != ""
        assert isinstance(task.expected_arguments, dict)
        assert task.task_category is not None


def test_dataset_validation():
    report = validate_dataset()
    assert report["valid"] is True
    assert len(report["errors"]) == 0
    assert report["tasks_count"] == 30
    assert report["tools_count"] >= 20


def test_corrupted_dataset_detection():
    # Intentionally corrupt a task to verify validator catches errors
    corrupted_task = Task(
        task_id="task_corrupted",
        domain="weather",
        original_query="What is the weather?",
        paraphrased_query="Check weather please",
        expected_tool_name="non_existent_tool_xyz",
        expected_arguments={}
    )
    report = validate_dataset(tasks=[corrupted_task])
    assert report["valid"] is False
    assert any("non_existent_tool_xyz" in err for err in report["errors"])


def test_mock_tool_registry():
    assert "get_weather" in MOCK_TOOL_REGISTRY
    weather_res = MOCK_TOOL_REGISTRY["get_weather"](city="Paris")
    assert weather_res["city"] == "Paris"
    assert "temperature" in weather_res

    calc_res = MOCK_TOOL_REGISTRY["calculate"](expression="10 + 20")
    assert calc_res["result"] == 30
