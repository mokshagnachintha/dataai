import traceback
import warnings
from typing import Any, List, Optional, Union

import pandas as pd

from pandasai.core.code_execution.code_executor import CodeExecutor
from pandasai.core.code_generation.base import CodeGenerator
from pandasai.core.prompts import (
    get_chat_prompt_for_sql,
    get_correct_error_prompt_for_sql,
    get_correct_output_type_error_prompt,
)
from pandasai.core.response.error import ErrorResponse
from pandasai.core.response.parser import ResponseParser
from pandasai.core.user_query import UserQuery
from pandasai.dataframe.base import DataFrame
from pandasai.exceptions import (
    CodeExecutionError,
    InvalidLLMOutputType,
)

from ..config import Config
from ..data_loader.duck_db_connection_manager import DuckDBConnectionManager
from ..query_builders.sql_parser import SQLParser
from .state import AgentState


class Agent:
    """
    Base Agent class to improve the conversational experience in PandasAI
    """

    def __init__(
        self,
        dfs: Union[DataFrame, List[DataFrame]],
        config: Optional[Union[Config, dict]] = None,
        memory_size: Optional[int] = 10,
        description: str = None,
    ):
        """
        Args:
            dfs (Union[Union[DataFrame, VirtualDataFrame], List[Union[DataFrame, VirtualDataFrame]]]): The dataframe(s) to be used for the conversation.
            config (Optional[Union[Config, dict]]): The configuration for the agent.
            memory_size (Optional[int]): The size of the memory.
            vectorstore (Optional[VectorStore]): The vectorstore to be used for the conversation.
            description (str): The description of the agent.
        """

        # Deprecation warnings
        if config is not None:
            warnings.warn(
                "The 'config' parameter is deprecated and will be removed in a future version. "
                "Please use the global configuration instead.",
                DeprecationWarning,
                stacklevel=2,
            )

        self.description = description
        self._state = AgentState()
        self._state.initialize(dfs, config, memory_size, description)

        self._code_generator = CodeGenerator(self._state)
        self._response_parser = ResponseParser()

    def chat(self, query: str, output_type: Optional[str] = None):
        """
        Start a new chat interaction with the assistant on Dataframe.
        """
        if self._state.config.llm is None:
            raise ValueError(
                "PandasAI API key does not include LLM credits. Please configure an OpenAI or LiteLLM key. "
                "Learn more at: https://docs.pandas-ai.com/v3/large-language-models#how-to-set-up-any-llm%3F"
            )

        self.start_new_conversation()
        return self._process_query(query, output_type)

    def follow_up(self, query: str, output_type: Optional[str] = None):
        """
        Continue the existing chat interaction with the assistant on Dataframe.
        """
        return self._process_query(query, output_type)

    def generate_code(self, query: Union[UserQuery, str]) -> str:
        """Generate code using the LLM."""

        self._state.memory.add(str(query), is_user=True)

        self._state.logger.log("Generating new code...")
        prompt = get_chat_prompt_for_sql(self._state)

        code = self._code_generator.generate_code(prompt)
        self._state.last_prompt_used = prompt
        return code

    def execute_code(self, code: str) -> dict:
        """Execute the generated code."""
        self._state.logger.log(f"Executing code: {code}")

        code_executor = CodeExecutor(self._state.config)
        code_executor.add_to_env("execute_sql_query", self._execute_sql_query)

        return code_executor.execute_and_return_result(code)

    def _execute_sql_query(self, query: str) -> pd.DataFrame:
        """
        Executes an SQL query on registered DataFrames.

        Args:
            query (str): The SQL query to execute.

        Returns:
            pd.DataFrame: The result of the SQL query as a pandas DataFrame.
        """
        if not self._state.dfs:
            raise ValueError("No DataFrames available to register for query execution.")

        db_manager = DuckDBConnectionManager()

        for df in self._state.dfs:
            # Register DataFrames with DuckDB using their table name
            table_name = df._table_name if hasattr(df, '_table_name') and df._table_name else 'df'
            db_manager.register(table_name, df)

        return db_manager.sql(query).df()

    def generate_code_with_retries(self, query: str) -> Any:
        """Execute the code with retry logic."""
        max_retries = self._state.config.max_retries
        attempts = 0
        try:
            return self.generate_code(query)
        except Exception as e:
            exception = e
            while attempts <= max_retries:
                try:
                    return self._regenerate_code_after_error(
                        self._state.last_code_generated, exception
                    )
                except Exception as e:
                    exception = e
                    attempts += 1
                    if attempts > max_retries:
                        self._state.logger.log(
                            f"Maximum retry attempts exceeded. Last error: {e}"
                        )
                        raise
                    self._state.logger.log(
                        f"Retrying Code Generation ({attempts}/{max_retries})..."
                    )
            return None

    def execute_with_retries(self, code: str) -> Any:
        """Execute the code with retry logic."""
        max_retries = self._state.config.max_retries
        attempts = 0

        while attempts <= max_retries:
            try:
                result = self.execute_code(code)
                return self._response_parser.parse(result, code)
            except Exception as e:
                attempts += 1
                if attempts > max_retries:
                    self._state.logger.log(f"Max retries reached. Error: {e}")
                    raise
                self._state.logger.log(
                    f"Retrying execution ({attempts}/{max_retries})..."
                )
                code = self._regenerate_code_after_error(code, e)

        return None

    def train(
        self,
        queries: Optional[List[str]] = None,
        codes: Optional[List[str]] = None,
        docs: Optional[List[str]] = None,
    ) -> None:
        """
        Trains the context to be passed to model
        Args:
            queries (Optional[str], optional): user user
            codes (Optional[str], optional): generated code
            docs (Optional[List[str]], optional): additional docs
        Raises:
            ImportError: if default vector db lib is not installed it raises an error
        """
        if self._state.vectorstore is None:
            raise MissingVectorStoreError(
                "No vector store provided. Please provide a vector store to train the agent."
            )

        if (queries and not codes) or (not queries and codes):
            raise ValueError(
                "If either queries or codes are provided, both must be provided."
            )

        if docs is not None:
            self._state.vectorstore.add_docs(docs)

        if queries and codes:
            self._state.vectorstore.add_question_answer(queries, codes)

        self._state.logger.log("Agent successfully trained on the data")

    def clear_memory(self):
        """
        Clears the memory
        """
        self._state.memory.clear()

    def add_message(self, message, is_user=False):
        """
        Add message to the memory. This is useful when you want to add a message
        to the memory without calling the chat function (for example, when you
        need to add a message from the agent).
        """
        self._state.memory.add(message, is_user=is_user)

    def start_new_conversation(self):
        """
        Clears the previous conversation
        """
        self.clear_memory()

    def _process_query(self, query: str, output_type: Optional[str] = None):
        """Process a user query and return the result."""
        query = UserQuery(query)
        self._state.logger.log(f"Question: {query}")
        self._state.logger.log(
            f"Running PandasAI with {self._state.config.llm.type} LLM..."
        )

        self._state.output_type = output_type
        try:
            self._state.assign_prompt_id()

            # Generate code
            code = self.generate_code_with_retries(query)

            # Execute code with retries
            result = self.execute_with_retries(code)

            self._state.logger.log("Response generated successfully.")
            # Generate and return the final response
            return result

        except CodeExecutionError:
            return self._handle_exception(code)

    def _regenerate_code_after_error(self, code: str, error: Exception) -> str:
        """Generate a new code snippet based on the error."""
        error_trace = traceback.format_exc()
        self._state.logger.log(f"Execution failed with error: {error_trace}")

        if isinstance(error, InvalidLLMOutputType):
            prompt = get_correct_output_type_error_prompt(
                self._state, code, error_trace
            )
        else:
            prompt = get_correct_error_prompt_for_sql(self._state, code, error_trace)

        return self._code_generator.generate_code(prompt)

    def _handle_exception(self, code: str) -> str:
        """Handle exceptions and return an error message."""
        error_message = traceback.format_exc()
        self._state.logger.log(f"Processing failed with error: {error_message}")

        return ErrorResponse(last_code_executed=code, error=error_message)

    @property
    def last_generated_code(self):
        return self._state.last_code_generated

    @property
    def last_code_executed(self):
        return self._state.last_code_generated

    @property
    def last_prompt_used(self):
        return self._state.last_prompt_used
