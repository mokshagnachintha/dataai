from __future__ import annotations

import hashlib
import os
from io import BytesIO
from typing import TYPE_CHECKING, Optional, Union
from zipfile import ZipFile

import pandas as pd
from pandas._typing import Axes, Dtype

import dataai as pai
from dataai.helpers.path import get_validated_dataset_path
from dataai.config import Config, ConfigManager
from dataai.constants import LOCAL_SOURCE_TYPES
from dataai.core.response import BaseResponse
from dataai.exceptions import DatasetNotFound, PandasAIApiKeyError
from dataai.helpers.dataframe_serializer import DataframeSerializer
from dataai.helpers.session import get_PandasAI_session

if TYPE_CHECKING:
    from dataai.agent.base import Agent


class SimpleSchema:
    """Minimal schema object for DataFrames without semantic layer."""
    def __init__(self, name: str, source_type: str = "csv", description: str = None):
        self.name = name
        self.description = description
        self.source = type('Source', (), {'type': source_type})()
        self.columns = None


class DataFrame(pd.DataFrame):
    """
    PandasAI DataFrame that extends pandas DataFrame with natural language capabilities.

    Attributes:
        name (Optional[str]): Name of the dataframe
        description (Optional[str]): Description of the dataframe
        schema (Optional[SemanticLayerSchema]): Schema definition for the dataframe
        config (Config): Configuration settings
    """

    _metadata = [
        "_agent",
        "_column_hash",
        "_table_name",
        "config",
        "path",
    ]

    def __init__(
        self,
        data=None,
        index: Axes | None = None,
        columns: Axes | None = None,
        dtype: Dtype | None = None,
        copy: bool | None = None,
        **kwargs,
    ) -> None:
        _schema: Optional[SemanticLayerSchema] = kwargs.pop("schema", None)
        _path: Optional[str] = kwargs.pop("path", None)
        _table_name: Optional[str] = kwargs.pop("_table_name", None)

        super().__init__(
            data=data, index=index, columns=columns, dtype=dtype, copy=copy
        )

        if _table_name:
            self._table_name = _table_name

        self._column_hash = self._calculate_column_hash()
        self.schema = _schema or DataFrame.get_default_schema(self)
        self.path = _path
        self._agent: Optional[Agent] = None

    def __repr__(self) -> str:
        """Return a string representation of the DataFrame."""
        name_str = f"name='{self.schema.name}'"
        desc_str = (
            f"description='{self.schema.description}'"
            if self.schema.description
            else ""
        )
        metadata = ", ".join(filter(None, [name_str, desc_str]))

        return f"PandasAI DataFrame({metadata})\n{super().__repr__()}"

    def _calculate_column_hash(self):
        column_string = ",".join(self.columns)
        return hashlib.md5(column_string.encode()).hexdigest()

    @property
    def column_hash(self):
        return self._column_hash

    @property
    def type(self) -> str:
        return "pd.DataFrame"

    def chat(self, prompt: str) -> BaseResponse:
        """
        Interact with the DataFrame using natural language.

        Args:
            prompt (str): The natural language query or instruction.

        Returns:
            str: The response to the prompt.
        """
        if self._agent is None:
            from dataai.agent import (
                Agent,
            )

            self._agent = Agent([self])

        return self._agent.chat(prompt)

    def follow_up(self, query: str, output_type: Optional[str] = None):
        if self._agent is None:
            raise ValueError(
                "No existing conversation. Please use chat() to start a new conversation."
            )
        return self._agent.follow_up(query, output_type)

    @property
    def rows_count(self) -> int:
        return len(self)

    @property
    def columns_count(self) -> int:
        return len(self.columns)

    def get_dialect(self):
        source = self.schema.source or None
        if source:
            dialect = "duckdb" if source.type in LOCAL_SOURCE_TYPES else source.type
        else:
            dialect = "postgres"

        return dialect

    def serialize_dataframe(self) -> str:
        """
        Serialize DataFrame to string representation.

        Returns:
            str: Serialized string representation of the DataFrame
        """
        dialect = self.get_dialect()
        return DataframeSerializer.serialize(self, dialect)

    def get_head(self):
        return self.head()

    @classmethod
    def get_default_schema(cls, dataframe: DataFrame):
        """Create a simple schema for the DataFrame."""
        table_name = getattr(
            dataframe, "_table_name", f"table_{dataframe._column_hash}"
        )
        return SimpleSchema(name=table_name, source_type="csv")
