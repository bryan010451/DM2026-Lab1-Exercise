import re
from typing import Dict, Any
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

    return [tokenize_tool]
