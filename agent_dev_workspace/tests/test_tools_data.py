import os
import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_data import make_tools


def test_tokenize_tool():
    # Initialize session and load dataset using premade tools
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_result = load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    assert load_result["n_documents"] == 8

    # Instantiate tokenize_tool from tools_data
    data_tools = make_tools(session)
    tokenize_tool = next(t for t in data_tools if t.name == "tokenize_tool")

    # Execute tokenize_tool
    res = tokenize_tool.invoke({})
    assert res["status"] == "success"
    assert res["n_documents"] == 8

    # Assert against known answer #10 in TEST_FIXTURE.md
    unigrams = session.dataframe["unigrams"].tolist()

    expected_unigrams = [
        ["always", "alpha", "alpha", "alpha", "beta"],
        ["always", "alpha", "alpha", "beta"],
        ["always", "alpha", "beta"],
        ["always", "gamma", "gamma", "gamma", "delta"],
        ["always", "gamma", "gamma", "delta"],
        ["always", "gamma", "delta"],
        ["always", "gamma", "delta"],
        []  # row 7 is empty text
    ]

    assert unigrams == expected_unigrams


def test_list_files_tool_success():
    session = SessionState()
    tools = make_tools(session)
    list_files_tool = next(t for t in tools if t.name == "list_files_tool")

    # Test listing existing directory
    res = list_files_tool.invoke({"directory": "."})
    assert res.get("status") == "success"
    assert "entries" in res
    assert res.get("total_entries") > 0


def test_list_files_tool_non_existent():
    session = SessionState()
    tools = make_tools(session)
    list_files_tool = next(t for t in tools if t.name == "list_files_tool")

    # Test non-existent directory error
    res = list_files_tool.invoke({"directory": "non_existent_folder_xyz"})
    assert "error" in res
    assert "does not exist" in res["error"]


def test_list_files_tool_empty_directory(tmp_path):
    session = SessionState()
    tools = make_tools(session)
    list_files_tool = next(t for t in tools if t.name == "list_files_tool")

    # Test empty directory error
    empty_dir = str(tmp_path / "empty_dir")
    os.makedirs(empty_dir)

    res = list_files_tool.invoke({"directory": empty_dir})
    assert "error" in res
    assert "No files or folders found" in res["error"]


def test_check_missing_tool():
    # Create session and load sample fixture dataset
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # Instantiate check_missing_tool from tools_data
    tools = make_tools(session)
    check_missing_tool = next(t for t in tools if t.name == "check_missing_tool")

    # Run check_missing_tool
    result = check_missing_tool.invoke({})

    # Assert against known answer #2 in TEST_FIXTURE.md
    assert result["status"] == "success"
    assert result["n_missing"] == 1
    assert result["missing_indices"] == [7]


def test_check_duplicates_tool():
    # Single session reused across setup and test
    session = SessionState()

    # Load premade load_dataset_tool
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    # Load data tools under test
    tools = make_tools(session)
    check_duplicates_tool = next(t for t in tools if t.name == "check_duplicates_tool")

    # 1. Load sample fixture dataset
    load_res = load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    assert load_res["n_documents"] == 8

    # 2. Check duplicates without dropping (Known Answer #3: exactly 1 duplicate row pair, 2 indices)
    res_check = check_duplicates_tool.invoke({"drop": False})
    assert res_check["status"] == "success"
    assert res_check["n_duplicates"] == 2
    assert res_check["dropped"] is False
    assert len(session.dataframe) == 8

    # 3. Drop duplicates with keep=False (Known Answer #3: both copies dropped, leaving 6 rows)
    res_drop = check_duplicates_tool.invoke({"drop": True})
    assert res_drop["status"] == "success"
    assert res_drop["dropped"] is True
    assert res_drop["n_remaining"] == 6
    assert len(session.dataframe) == 6
