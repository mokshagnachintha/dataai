import traceback

from dataai.agent.state import AgentState
from dataai.core.prompts.base import BasePrompt

from .code_cleaning import CodeCleaner
from .code_validation import CodeRequirementValidator


class CodeGenerator:
    def __init__(self, context: AgentState):
        self._context = context
        self._code_cleaner = CodeCleaner(self._context)
        self._code_validator = CodeRequirementValidator(self._context)

    def generate_code(self, prompt: BasePrompt) -> str:
        """
        Generates code using a given LLM and performs validation and cleaning steps.

        Args:
            context (PipelineContext): The pipeline context containing dataframes.
            prompt (BasePrompt): The prompt to guide code generation.

        Returns:
            str: The final cleaned and validated code.

        Raises:
            Exception: If any step fails during the process.
        """
        try:
            # Generate the code
            code = self._context.config.llm.generate_code(prompt, self._context)
            self._context.last_code_generated = code

            return self.validate_and_clean_code(code)

        except Exception as e:
            error_message = f"An error occurred during code generation: {e}"
            stack_trace = traceback.format_exc()

            raise e

    def validate_and_clean_code(self, code: str) -> str:
        # Validate code requirements
        if not self._code_validator.validate(code):
            raise ValueError("Code validation failed due to unmet requirements.")

        # Clean the code
        return self._code_cleaner.clean_code(code)
