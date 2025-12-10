# -*- coding: utf-8 -*-
"""
PandasAI is a wrapper around a LLM to make dataframes conversational
"""

from io import BytesIO
from typing import Hashable, Union

import pandas as pd

from pandasai.config import APIKeyManager, ConfigManager

from .agent import Agent
from .dataframe import DataFrame
from .helpers.path import get_table_name_from_path
from .helpers.sql_sanitizer import (
    sanitize_sql_table_name,
    sanitize_sql_table_name_lowercase,
)
from .smart_dataframe import SmartDataframe
from .smart_datalake import SmartDatalake


# Global variable to store the current agent
_current_agent = None

config = ConfigManager()

api_key = APIKeyManager()


def chat(query: str, *dataframes: DataFrame):
    """
    Start a new chat interaction with the assistant on Dataframe(s).

    Args:
        query (str): The query to run against the dataframes.
        *dataframes: Variable number of dataframes to query.

    Returns:
        The result of the query.
    """
    global _current_agent
    if not dataframes:
        raise ValueError("At least one dataframe must be provided.")

    _current_agent = Agent(list(dataframes))
    return _current_agent.chat(query)


def follow_up(query: str):
    """
    Continue the existing chat interaction with the assistant on Dataframe(s).

    Args:
        query (str): The follow-up query to run.

    Returns:
        The result of the query.
    """
    global _current_agent

    if _current_agent is None:
        raise ValueError(
            "No existing conversation. Please use chat() to start a new conversation."
        )

    return _current_agent.follow_up(query)


def read_csv(filepath: Union[str, BytesIO]) -> DataFrame:
    data = pd.read_csv(filepath)
    table = get_table_name_from_path(filepath)
    return DataFrame(data, _table_name=table)


def read_excel(
    filepath: Union[str, BytesIO],
    sheet_name: Union[str, int, list[Union[str, int]], None] = 0,
) -> dict[Hashable, DataFrame] | DataFrame:
    data = pd.read_excel(filepath, sheet_name=sheet_name)

    if isinstance(data, pd.DataFrame):
        table = get_table_name_from_path(filepath)
        return DataFrame(data, _table_name=table)

    return {
        k: DataFrame(
            v,
            _table_name=sanitize_sql_table_name_lowercase(
                f"{get_table_name_from_path(filepath)}_{k}"
            ),
        )
        for k, v in data.items()
    }


__all__ = [
    "Agent",
    "DataFrame",
    "pandas",
    "chat",
    "follow_up",
    # Deprecated
    "SmartDataframe",
    "SmartDatalake",
]
