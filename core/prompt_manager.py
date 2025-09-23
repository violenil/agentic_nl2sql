import yaml, datetime, os
from typing import Dict

class PromptManager:
    def __init__(self, prompt_dir="prompts", history_file="history/prompt_evolution.log"):
        self.prompt_dir = prompt_dir
        self.history_file = history_file
        self.prompts = self._load_prompts()

    def _load_prompts(self) -> Dict[str, str]:
        prompts = {}
        for fname in os.listdir(self.prompt_dir):
            if fname.endswith(".yaml"):
                stage = fname.replace(".yaml", "")
                with open(os.path.join(self.prompt_dir, fname), "r") as f:
                    data = yaml.safe_load(f)
                    if "prompt" in data:
                        prompts[stage] = data["prompt"]
                    elif "system" in data and "user" in data:
                        prompts[stage] = {
                            "system": data["system"],
                            "user": data["user"]
                        }
                    else:
                        raise ValueError(f"Prompt file {f} missing prompt or system/user keys")
        return prompts

    def save_refinement(self, stage, old_prompt, new_prompt, critique):
        os.makedirs(os.path.dirname(self.history_file), exist_ok=True)
        with open(self.history_file, "a") as f:
            f.write(yaml.dump([{
                "time": datetime.datetime.now().isoformat(),
                "stage": stage,
                "critique": critique,
                "old_prompt": old_prompt.strip(),
                "new_prompt": new_prompt.strip()
            }], sort_keys=False))
            f.write("\n---\n")

    def update_prompt(self, stage, new_prompt, critique):
        old_prompt = self.prompts[stage]
        self.prompts[stage] = new_prompt
        self.save_refinement(stage, old_prompt, new_prompt, critique)