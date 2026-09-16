import asyncio
import time

from backend.graph.graph import chatbot


async def run_evaluation(query: str):
    initial_state = {
        "query": query,
        "research_plan": [],
        "web_results": [],
        "paper_results": [],
        "evidence": [],
        "verified_evidence": [],
        "report": "",
        "citations": [],
        "final_report": "",
    }

    config = {
        "configurable": {
            "thread_id": "evaluation-run-1"
        }
    }

    start_time = time.perf_counter()

    result = await chatbot.ainvoke(
        initial_state,
        config=config,
    )

    end_time = time.perf_counter()

    latency = end_time - start_time

    return {
        "query": query,
        "latency_seconds": latency,
        "report": result["final_report"],
        "citations": result["citations"],
        "evidence_count": len(result["evidence"]),
        "verified_evidence_count": len(
            result["verified_evidence"]
        ),
    }


async def main():

    query = (
        "What are the major techniques used to "
        "reduce hallucinations in large language models?"
    )

    result = await run_evaluation(query)

    print("\nEvaluation Results")
    print("=" * 60)

    print(
        f"Latency: "
        f"{result['latency_seconds']:.2f} seconds"
    )

    print(
        f"Evidence collected: "
        f"{result['evidence_count']}"
    )

    print(
        f"Evidence verified: "
        f"{result['verified_evidence_count']}"
    )

    print(
        f"Citations generated: "
        f"{len(result['citations'])}"
    )


if __name__ == "__main__":
    asyncio.run(main())