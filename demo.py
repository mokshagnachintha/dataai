import os
import pandas as pd
import dataai as dai
from dataai.llm.base import LLM
from dataai.core.prompts.base import BasePrompt

# Get API configuration
api_key = os.environ.get("OPENAI_API_KEY")
api_base = os.environ.get("OPENAI_API_BASE")

print(f"API Key: {api_key[:20] if api_key else 'NOT SET'}...")
print(f"API Base: {api_base if api_base else 'NOT SET'}")

# Create OpenRouter LLM
class OpenRouterLLM(LLM):
    """OpenRouter LLM via OpenAI-compatible API"""
    
    def __init__(self, api_key: str, api_base: str = None):
        super().__init__(api_key=api_key)
        self.api_base = api_base or "https://openrouter.ai/api/v1"
        self._type = "openrouter"
    
    @property
    def type(self) -> str:
        return self._type
    
    def call(self, instruction: BasePrompt, context=None) -> str:
        """Call the LLM API"""
        from openai import OpenAI
        
        client = OpenAI(api_key=self.api_key, base_url=self.api_base)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": instruction.to_string()}],
            temperature=0,
        )
        print("✓ API CALL MADE to OpenRouter")
        return response.choices[0].message.content

llm = OpenRouterLLM(api_key=api_key, api_base=api_base)
print(f"LLM Type: {type(llm).__name__}\n")

dai.config.set({"llm": llm})

# Create DataFrame
df = pd.DataFrame({
    "country": ["United States", "United Kingdom", "France", "Germany", "Italy"],
    "revenue": [5000, 3200, 2900, 4100, 2300]
})

smart_df = dai.DataFrame(df)

print("=" * 60)
print("DataFrame created successfully!")
print("=" * 60)
print(f"DataFrame shape: {smart_df.shape}")
print(f"DataFrame columns: {list(smart_df.columns)}")
print("\nData:")
print(smart_df)

# Ask the question
print("\n" + "=" * 60)
print("Query: Which country has the highest revenue?")
print("=" * 60)

answer = smart_df.chat("Which country has the highest revenue?")
print(f"\nLLM Response:\n{answer}")