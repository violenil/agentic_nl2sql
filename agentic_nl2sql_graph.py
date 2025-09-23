import os
from datetime import datetime

from langgraph.graph import StateGraph, END
from typing import TypedDict, Any, Dict

from agents.analyzer import ExperimentAnalyzer
from agents.critic import CriticAgent
from agents.refiner import RefinerAgent
from agents.stage1 import Stage1Agent
from agents.stage2 import Stage2Agent
from agents.stage3 import Stage3Agent

from core.prompt_manager import PromptManager

MAX_REFINEMENTS = int(os.getenv("MAX_REFINEMENTS", "3"))

# Define shared state
class NL2SQLState(TypedDict):
    question: str
    schema: str
    stage1: Any
    stage2: Any
    sql: str
    analysis: Dict[str, Any]
    previous_analysis: Dict[str, Any]
    critique: Dict[str, Any]
    refinement: Dict[str, Any]
    prompts: Dict[str, str]
    prompt_manager: Any
    refinement_count: int


def stage1_node(state):
    # expects state to have {"prompt_manager": PromptManager}
    agent = Stage1Agent(state["prompt_manager"])
    state["stage1"] = agent.run(state["question"], state["schema"])
    return state

def stage2_node(state):
    agent = Stage2Agent(state["prompt_manager"])
    state["stage2"] = agent.run(state["question"], state["stage1"])
    return state

def stage3_node(state):
    agent = Stage3Agent(state["prompt_manager"])
    state["sql"] = agent.run(state["question"], state["stage1"], state["stage2"])
    return state

def analyzer_node(state):
    """
    Expects in state:
      - question: str
      - sql: str
      - optional: db_path in state["config"]["db_path"] (else env var will be used)
    Outputs:
      - state["analysis"]: dict
    """
    db_path = None
    if isinstance(state.get("config"), dict):
        db_path = state["config"].get("db_path")
    analyzer = ExperimentAnalyzer(db_path=db_path)
    state["analysis"] = analyzer.analyze(state["question"], state["sql"])
    return state

def critique_node(state):
    agent = CriticAgent(state["prompt_manager"])
    state["critique"] = agent.run(state, state["analysis"])
    return state

def refiner_node(state):
    history_file = None
    if isinstance(state.get("config"), dict):
        history_file = state["config"].get("history_file")

    agent = RefinerAgent(state["prompt_manager"], history_file=history_file)
    refinement_report = agent.run(state["critique"])
    state["refinement"] = refinement_report

    state["previous_analysis"] = state.get("analysis", {}).copy()
    state["refinement_count"] = state.get("refinement_count", 0) + 1

    return state

# Decision node
def accept_refinement_node(state: NL2SQLState):
    """Decide whether to accept refinement based on old vs new analysis."""
    old = state.get("previous_analysis", {})
    new = state.get("analysis", {})

    # Default: reject if nothing to compare
    if not old:
        return "reject"

    # If new syntax fails, reject
    if not new.get("syntax_ok", False):
        return "reject"

    # Improvement heuristic: more rows returned than before
    if new.get("row_count_sample", 0) > old.get("row_count_sample", 0):
        return "accept"

    # Optional: could add semantic-hint based checks here
    return "reject"

def refinement_router(state: NL2SQLState):
    """Decide whether to keep refining or stop."""
    if state.get("refinement_count", 0) >= MAX_REFINEMENTS:
        return "stop"
    old = state.get("previous_analysis", {})
    new = state.get("analysis", {})

    # If improved (syntax ok and more rows), stop early
    if new.get("syntax_ok", False) and new.get("row_count_sample", 0) > old.get("row_count_sample", 0):
        return "stop"

    # Otherwise, try another refinement
    return "refine_more"

# Init state with PromptManager
pm = PromptManager(prompt_dir="prompts")


# Build graph
builder = StateGraph(NL2SQLState)

# main pipeline
builder.add_node("stage1", stage1_node)
builder.add_node("stage2", stage2_node)
builder.add_node("stage3", stage3_node)
builder.add_node("analyzer", analyzer_node)
builder.add_node("critic", critique_node)
builder.add_node("refiner", refiner_node)

# rerun pipeline after refinement
builder.add_node("stage1_rerun", stage1_node)
builder.add_node("stage2_rerun", stage2_node)
builder.add_node("stage3_rerun", stage3_node)
builder.add_node("analyzer_rerun", analyzer_node)
builder.add_node("accept_refinement", accept_refinement_node)

# entry + main edges
builder.set_entry_point("stage1")
builder.add_edge("stage1", "stage2")
builder.add_edge("stage2", "stage3")
builder.add_edge("stage3", "analyzer")
builder.add_edge("analyzer", "critic")
builder.add_edge("critic", "refiner")

# refinement loop
builder.add_edge("refiner", "stage1_rerun")
builder.add_edge("stage1_rerun", "stage2_rerun")
builder.add_edge("stage2_rerun", "stage3_rerun")
builder.add_edge("stage3_rerun", "analyzer_rerun")

# conditional decision after rerun
builder.add_conditional_edges(
    "analyzer_rerun",
    refinement_router,
    {"stop": END, "refine_more": "critic"}
)

graph = builder.compile()

import dotenv

dotenv.load_dotenv(".env")

def get_schema_string(db_path: str) -> str:
    import sqlite3
    conn = sqlite3.connect(db_path)
    tables = [row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table';")]
    schema_lines = []
    for t in tables:
        cols = [col[1] for col in conn.execute(f"PRAGMA table_info({t});")]
        schema_lines.append(f"{t}({', '.join(cols)})")
    return "\n".join(schema_lines)

if __name__ == "__main__":
    pm = PromptManager(prompt_dir="prompts")

    # Example NL question
    #question = "In Formula 1 seasons since 2001, considering only drivers who scored points in a season, which five constructors have had the most seasons where their drivers scored the fewest total points among all point-scoring drivers in that season?"
    question = "return me the homepage of PVLDB ."
    schema = get_schema_string(os.getenv("SQLITE_DB_FILE", None))

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    history_file = f"history/prompt_evolution_{timestamp}.log"
    # Initial state
    state = {
        "prompt_manager": pm,
        "question": question,
        "schema": schema,
        "config": {
            "db_path": os.getenv("SQLITE_DB_FILE", None),
            "history_file": history_file,
        },
        "refinement_count": 0,
    }
    # Run graph
    final_state = graph.invoke(state)

    print("\n--- Final State ---")
    print("SQL:", final_state.get("sql"))
    print("Analysis:", final_state.get("analysis"))
    print("Critique:", final_state.get("critique"))
    print("Refinement:", final_state.get("refinement"))