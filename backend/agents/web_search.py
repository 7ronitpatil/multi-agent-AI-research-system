import os
import asyncio

from dotenv import load_dotenv
from tavily import TavilyClient

from backend.graph.state import ResearchState
from backend.graph.utils import select_evenly


load_dotenv()

tavily_client = TavilyClient(
    api_key=os.getenv("TAVILY_API_KEY")
)


async def search_web(task: str) -> dict:

    try:

        result = await asyncio.to_thread(
            tavily_client.search,
            query=task,
            search_depth="advanced",
            max_results=5,
        )

        return {
            "task": task,
            "results": result.get("results", []),
        }

    except Exception as error:

        print(
            f"\n[Web Agent] Search failed for task: {task}"
        )

        print(
            f"[Web Agent] Error: {error}"
        )

        return {
            "task": task,
            "results": [],
        }


async def web_research_agent(
    state: ResearchState
) -> dict:

    if not state["use_web"]:
        return {
            "web_results": []
        }

    research_plan = state["research_plan"]

    search_count = min(
        state["web_searches"],
        len(research_plan)
    )

    # Spread the search budget across the whole plan instead
    # of only ever searching the first N tasks.
    selected_tasks = select_evenly(
        research_plan,
        search_count
    )

    tasks = [
        search_web(task)
        for task in selected_tasks
    ]

    results = await asyncio.gather(*tasks)

    return {
        "web_results": results
    }