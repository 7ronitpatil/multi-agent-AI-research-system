from backend.graph.state import ResearchState


async def evidence_collector(
    state: ResearchState
) -> dict:

    evidence = []


    # --------------------------------------------------
    # Debug information
    # --------------------------------------------------

    print(
        f"\n[Evidence Collector] "
        f"Web result groups: {len(state['web_results'])}"
    )

    print(
        f"[Evidence Collector] "
        f"Paper result groups: {len(state['paper_results'])}"
    )


    # --------------------------------------------------
    # Collect Web Evidence
    # --------------------------------------------------

    for task_result in state["web_results"]:

        task = task_result["task"]

        results = task_result["results"]

        print(
            f"[Evidence Collector] "
            f"Web task '{task}' returned "
            f"{len(results)} results"
        )

        for result in results:

            evidence.append(
                {
                    "source_type": "web",
                    "task": task,
                    "title": result.get(
                        "title",
                        ""
                    ),
                    "content": result.get(
                        "content",
                        ""
                    ),
                    "url": result.get(
                        "url",
                        ""
                    ),
                }
            )


    # --------------------------------------------------
    # Collect Paper Evidence
    # --------------------------------------------------

    for task_result in state["paper_results"]:

        task = task_result["task"]

        results = task_result["results"]

        print(
            f"[Evidence Collector] "
            f"Paper task '{task}' returned "
            f"{len(results)} results"
        )

        for result in results:

            evidence.append(
                {
                    "source_type": "paper",
                    "task": task,
                    "title": result.get(
                        "title",
                        ""
                    ),
                    "content": result.get(
                        "summary",
                        ""
                    ),
                    "url": result.get(
                        "url",
                        ""
                    ),
                }
            )


    # --------------------------------------------------
    # Final evidence count
    # --------------------------------------------------

    print(
        f"[Evidence Collector] "
        f"Total evidence collected: {len(evidence)}"
    )


    return {
        "evidence": evidence
    }