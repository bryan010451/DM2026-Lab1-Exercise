import re
import os
from typing import Dict, Any, Optional
import pandas as pd
from langchain_core.tools import tool


def make_tools(session):

    @tool
    def tokenize_tool() -> Dict[str, Any]:
        """Tokenize text in session.dataframe into individual word tokens (unigrams).

        Reads document text from session.dataframe["text"], extracts all word
        tokens (alphanumeric sequences), lowercases them, and stores the resulting
        list of word lists in session.dataframe["unigrams"] and session.artifacts["unigrams"].

        Each document's text is converted into a Python list of string elements, where
        each element is an individual word token.

        Returns:
            Dict containing status, total documents tokenized, total unigram count,
            and the assigned result_id.
        """
        if session.dataframe is None:
            return {"error": "No dataset loaded in session. Call load_dataset_tool first."}

        if "text" not in session.dataframe.columns:
            return {"error": "DataFrame missing required 'text' column."}

        unigrams_list = []
        total_tokens = 0

        for text in session.dataframe["text"]:
            text_str = str(text) if text is not None else ""
            # Extract individual words as elements of a Python list
            tokens = re.findall(r"\b\w+\b", text_str.lower())
            unigrams_list.append(tokens)
            total_tokens += len(tokens)

        # Store list of lists in dataframe and artifacts
        session.dataframe["unigrams"] = unigrams_list
        session.artifacts["unigrams"] = unigrams_list

        result_id = session.next_result_id("tokenize")
        summary = {
            "result_id": result_id,
            "status": "success",
            "n_documents": len(unigrams_list),
            "total_unigrams": total_tokens,
            "sample_tokens": unigrams_list[0][:10] if unigrams_list else [],
        }

        session.store_result("tokenize_tool", {}, summary)
        return summary

    @tool
    def list_files_tool(directory: str = ".") -> Dict[str, Any]:
        """List files and directories within a target directory in the repository.

        Enables exploring available datasets and pipeline files before loading.
        Checks if the specified directory exists and whether it contains any files or folders.
        Returns an error response if the directory does not exist or if no files/folders are found.

        Args:
            directory: Directory path to inspect (defaults to current directory '.').

        Returns:
            Dict containing status, directory path, list of entries, total count,
            and assigned result_id, or an error dict if empty/invalid.
        """
        if not os.path.exists(directory):
            return {"error": f"Directory '{directory}' does not exist."}

        if not os.path.isdir(directory):
            return {"error": f"Path '{directory}' is not a directory."}

        try:
            entries = os.listdir(directory)
        except Exception as e:
            return {"error": f"Failed to list directory '{directory}': {str(e)}"}

        if not entries:
            return {"error": f"No files or folders found in directory '{directory}'."}

        result_id = session.next_result_id("list_files")
        summary = {
            "result_id": result_id,
            "status": "success",
            "directory": directory,
            "entries": sorted(entries),
            "total_entries": len(entries),
        }

        session.store_result("list_files_tool", {"directory": directory}, summary)
        return summary

    @tool
    def count_rows_tool(file_path: str, skip_header: bool = True) -> Dict[str, Any]:
        """Count the number of data rows in a file, optionally excluding the header row.

        Args:
            file_path: Path to the target file.
            skip_header: Whether to exclude the first row (header/attribute row) from the count (default True).

        Returns:
            Dict containing result_id, file_path, total_lines, data_rows_count, and skip_header setting.
        """
        if not os.path.exists(file_path):
            return {"error": f"File '{file_path}' does not exist."}

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
        except Exception as e:
            return {"error": f"Failed to read file '{file_path}': {str(e)}"}

        total_lines = len(lines)
        data_rows_count = max(0, total_lines - 1) if skip_header and total_lines > 0 else total_lines

        result_id = session.next_result_id("count_rows")
        summary = {
            "result_id": result_id,
            "status": "success",
            "file_path": file_path,
            "total_lines": total_lines,
            "data_rows_count": data_rows_count,
            "skip_header": skip_header,
        }

        session.store_result("count_rows_tool", {"file_path": file_path, "skip_header": skip_header}, summary)
        return summary

    @tool
    def check_missing_tool() -> Dict[str, Any]:
        """Check for missing or empty text elements in session.dataframe.

        Identifies rows where text is null, NaN, or contains only whitespace,
        and records the specific row indices (directions/locations) where these occur.

        Returns:
            Dict containing status, total missing count, list of missing row indices,
            and assigned result_id.
        """
        if session.dataframe is None:
            return {"error": "No dataset loaded in session. Call load_dataset_tool first."}

        if "text" not in session.dataframe.columns:
            return {"error": "DataFrame missing required 'text' column."}

        missing_indices = []
        for idx, text in enumerate(session.dataframe["text"]):
            if text is None or (isinstance(text, float) and pd.isna(text)):
                missing_indices.append(idx)
            elif isinstance(text, str) and not text.strip():
                missing_indices.append(idx)

        result_id = session.next_result_id("check_missing")
        summary = {
            "result_id": result_id,
            "status": "success",
            "n_missing": len(missing_indices),
            "missing_indices": missing_indices,
            "has_missing": len(missing_indices) > 0,
        }

        session.store_result("check_missing_tool", {}, summary)
        return summary

    return [tokenize_tool, list_files_tool, count_rows_tool, check_missing_tool]
