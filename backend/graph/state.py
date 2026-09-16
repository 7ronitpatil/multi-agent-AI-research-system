from typing import TypedDict


class ResearchState(TypedDict):
    query: str

    # Planner
    research_plan: list[str]

    # Research agents
    web_results: list[dict]
    paper_results: list[dict]

    # Evidence pipeline
    evidence: list[dict]
    verified_evidence: list[dict]

    # Final output
    report: str
    citations: list[dict]
    final_report: str