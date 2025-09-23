from typing import TypedDict, Dict, Any

class NL2SQLState(TypedDict):
    question: str
    schema: str
    stage1: str
    stage2: str
    sql: str
    analysis: Dict[str, Any]
    critique: Dict[str, Any]
    prompts: Dict[str, str]