import asyncio
import arxiv

from backend.graph.state import ResearchState
from backend.graph.utils import select_evenly


ARXIV_SEMAPHORE = asyncio.Semaphore(1)


async def search_papers(task: str) -> dict:
    """
    Search arXiv for one research task.
    """

    async with ARXIV_SEMAPHORE:

        def perform_search():

            client = arxiv.Client(
                page_size=5,
                delay_seconds=5,
                num_retries=1,
            )

            search = arxiv.Search(
                query=task,
                max_results=5,
                sort_by=arxiv.SortCriterion.Relevance,
            )

            results = []

            for paper in client.results(search):

                results.append(
                    {
                        "title": paper.title,
                        "authors": [
                            author.name
                            for author in paper.authors
                        ],
                        "summary": paper.summary,
                        "url": paper.entry_id,
                        "published": str(paper.published),
                    }
                )

            return results

        try:

            results = await asyncio.to_thread(
                perform_search
            )

            return {
                "task": task,
                "results": results,
            }

        except arxiv.HTTPError as error:

            print(
                f"\n[Paper Agent] arXiv request failed "
                f"for task: {task}"
            )

            print(
                f"[Paper Agent] Error: {error}"
            )

            return {
                "task": task,
                "results": [],
            }

        except Exception as error:

            print(
                f"\n[Paper Agent] Unexpected error "
                f"for task: {task}"
            )

            print(
                f"[Paper Agent] Error: {error}"
            )

            return {
                "task": task,
                "results": [],
            }


async def paper_research_agent(
    state: ResearchState
) -> dict:

    if not state["use_papers"]:
        return {
            "paper_results": []
        }

    research_plan = state["research_plan"]

    search_count = min(
        state["paper_searches"],
        len(research_plan)
    )

    selected_tasks = select_evenly(
        research_plan,
        search_count
    )

    tasks = [
        search_papers(task)
        for task in selected_tasks
    ]

    results = await asyncio.gather(
        *tasks
    )

    return {
        "paper_results": results
    }