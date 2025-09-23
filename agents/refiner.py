import json
from datetime import datetime
import os
from langchain_openai import AzureChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from core.prompt_manager import PromptManager
from typing import Any, Dict

def sanitize_prompt(prompt: str) -> str:
    """Escape stray braces while preserving {question}, {stage1}, {stage2} placeholders."""
    allowed = ["{question}", "{stage1}", "{stage2}"]
    # Escape all braces
    prompt = prompt.replace("{", "{{").replace("}", "}}")
    # Restore allowed placeholders
    for ph in allowed:
        prompt = prompt.replace("{{" + ph.strip("{}") + "}}", ph)
    return prompt

class RefinerAgent:
    def __init__(self, prompt_manager: PromptManager, deployment_name: str = "lunar-gpt-4o",
                 history_file: str = None):
        self.prompt_manager = prompt_manager

        if history_file is None:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            self.history_file = f"history/prompt_evolution_{timestamp}.log"
        else:
            self.history_file = history_file

        os.makedirs(os.path.dirname(self.history_file), exist_ok=True)

        self.llm = AzureChatOpenAI(
            deployment_name=deployment_name,
            api_version="2024-12-01-preview",
            temperature=0,
            max_tokens=800,
            model_kwargs={"response_format": {"type": "json_object"}}
        )

    def run(self, critique: Dict[str, Any]) -> Dict[str, Any]:
        stage = critique.get("likely_stage")
        if not stage or stage not in self.prompt_manager.prompts:
            return {"status": "no_refinement", "reason": "No actionable stage identified"}

        original_prompt = self.prompt_manager.prompts[stage]

        # Load refiner template
        template = self.prompt_manager.prompts["refiner"]
        system_msg = template["system"]
        user_msg = template["user"].format(
            stage=stage,
            original_prompt=original_prompt,
            critique_json=json.dumps(critique, indent=2)
        )

        response = self.llm.invoke([
            SystemMessage(content=system_msg),
            HumanMessage(content=user_msg)
        ])
        text = response.content.strip()

        try:
            result = json.loads(text)
        except json.JSONDecodeError:
            return {"status": "failed", "reason": "Invalid JSON from LLM", "raw": text}

        new_prompt = result.get("new_prompt", original_prompt)
        new_prompt = sanitize_prompt(new_prompt)
        explanation = result.get("explanation", "")

        # Update prompt
        self.prompt_manager.prompts[stage] = new_prompt

        # Log refinement
        with open(self.history_file, "a") as f:
            f.write(f"\n[{datetime.utcnow().isoformat()}] Stage: {stage}\n")
            f.write(f"Issues: {critique.get('issues')}\n")
            f.write(f"Notes: {critique.get('notes')}\n")
            f.write(f"Explanation: {explanation}\n")
            f.write("---- BEFORE ----\n")
            f.write(json.dumps(original_prompt, indent=2) + "\n")
            f.write("---- AFTER ----\n")
            f.write(json.dumps(new_prompt, indent=2) + "\n")
            f.write("="*60 + "\n")

        return {"status": "refined", "stage": stage, "explanation": explanation}