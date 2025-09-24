import json
from typing import Any, Dict

from langchain.schema import HumanMessage
from langchain_openai import AzureChatOpenAI

from core.prompt_manager import PromptManager


class Stage1Agent:
    def __init__(
        self, prompt_manager: PromptManager, deployment_name: str = "lunar-gpt-4o"
    ):
        self.prompt_manager = prompt_manager
        self.llm = AzureChatOpenAI(
            deployment_name=deployment_name,
            api_version="2024-12-01-preview",
            temperature=0,  # deterministic
            max_tokens=500,
            model_kwargs={"response_format": {"type": "json_object"}},
        )

    def run(self, question: str, schema: str) -> Dict[str, Any]:
        """
        Select relevant tables and attributes given NL question + schema.
        Returns dict: {"tables": [...], "attributes": [...]}
        """
        prompt = self.prompt_manager.prompts["stage1"].format(
            question=question, schema=schema
        )

        # Call Azure OpenAI
        response = self.llm.invoke([HumanMessage(content=prompt)])
        text = response.content.strip()

        # Clean common markdown fences
        if text.startswith("```"):
            text = text.strip("`")  # remove backticks
            if text.lower().startswith("json"):
                text = text[4:].strip()  # remove leading 'json'

        try:
            result = json.loads(text)
        except json.JSONDecodeError:
            result = {"tables": [], "attributes": [], "raw_output": response.content}

        print("STAGE 1: ", result)
        return result
