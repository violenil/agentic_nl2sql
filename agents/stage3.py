from typing import Dict, Any

from core.prompt_manager import PromptManager
from langchain_openai import AzureChatOpenAI
from langchain.schema import HumanMessage


class Stage3Agent:
    def __init__(self, prompt_manager: PromptManager, deployment_name: str = "lunar-gpt-4o"):
        self.prompt_manager = prompt_manager
        self.llm = AzureChatOpenAI(
            deployment_name=deployment_name,
            api_version="2024-05-01-preview",  # 👈 adjust if different
            temperature=0,
            max_tokens=500
        )

    def run(self, question: str, stage1_output: Dict[str, Any], stage2_output: Dict[str, Any]) -> str:
        """
        Generate the final SQL query as plain text.
        """
        prompt = self.prompt_manager.prompts["stage3"].format(
            question=question,
            stage1=stage1_output,
            stage2=stage2_output
        )

        response = self.llm.invoke([HumanMessage(content=prompt)])
        sql = response.content.strip()

        # Ensure it ends with semicolon
        if not sql.endswith(";"):
            sql += ";"

        print("STAGE 3: ", sql)
        return sql
