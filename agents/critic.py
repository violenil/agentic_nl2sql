import json
from typing import Any, Dict

from langchain.schema import HumanMessage
from langchain_openai import AzureChatOpenAI

from core.prompt_manager import PromptManager


class CriticAgent:
    """
    LLM-based critic: reads Analyzer report + pipeline logs and outputs a structured critique.
    """

    def __init__(
        self, prompt_manager: PromptManager, deployment_name: str = "lunar-gpt-4o"
    ):
        self.prompt_manager = prompt_manager
        self.llm = AzureChatOpenAI(
            deployment_name=deployment_name,
            api_version="2024-05-01-preview",
            temperature=0,
            max_tokens=500,
            model_kwargs={
                "response_format": {"type": "json_object"}
            },  # force JSON output
        )

    def run(self, logs: Dict[str, Any], analysis: Dict[str, Any]) -> Dict[str, Any]:
        prompt = self.prompt_manager.prompts["critic"].format(
            question=logs.get("question", ""),
            schema=logs.get("schema", ""),
            sql=logs.get("sql", ""),
            stage1=logs.get("stage1", ""),
            stage2=logs.get("stage2", ""),
            analysis=json.dumps(analysis, indent=2),
        )

        response = self.llm.invoke([HumanMessage(content=prompt)])
        text = response.content.strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {"likely_stage": None, "issues": ["Parse error"], "notes": [text]}
