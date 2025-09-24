import json
from string import Template
from typing import Any, Dict

from langchain.schema import HumanMessage
from langchain_openai import AzureChatOpenAI

from core.prompt_manager import PromptManager


class Stage2Agent:
    def __init__(
        self, prompt_manager: PromptManager, deployment_name: str = "lunar-gpt-4o"
    ):
        self.prompt_manager = prompt_manager
        self.llm = AzureChatOpenAI(
            deployment_name=deployment_name,
            api_version="2024-12-01-preview",
            temperature=0,
            max_tokens=500,
            model_kwargs={"response_format": {"type": "json_object"}},
        )

    def run(self, question: str, stage1_output: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract literal values (constants, strings, numbers, dates, etc.)
        and map them to candidate columns.
        Returns dict: {"predicates": [{"column": ..., "operator": ..., "value": ...}, ...]}
        """

        raw_prompt = self.prompt_manager.prompts["stage2"]
        prompt = Template(raw_prompt).substitute(
            question=question, stage1=json.dumps(stage1_output)
        )

        response = self.llm.invoke([HumanMessage(content=prompt)])
        text = response.content.strip()

        # Clean markdown fences
        if text.startswith("```"):
            text = text.strip("`")
            if text.lower().startswith("json"):
                text = text[4:].strip()  # remove "json" tag

        try:
            result = json.loads(text)
        except json.JSONDecodeError:
            result = {"predicates": [], "raw_output": response.content}

        print("STAGE 2: ", result)
        return result
