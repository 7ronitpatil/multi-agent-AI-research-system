import asyncio


if __name__ == "__main__":
    asyncio.set_event_loop_policy(
        asyncio.WindowsSelectorEventLoopPolicy()
    )


from backend.graph.graph import builder, get_checkpointer


async def run_research(query: str):

    initial_state = {

        # User query
        "query": query,

        # Planner
        "research_plan": [],
        "use_web": False,
        "use_papers": False,
        "web_searches": 0,
        "paper_searches": 0,

        # Research agents
        "web_results": [],
        "paper_results": [],

        # Evidence pipeline
        "evidence": [],
        "verified_evidence": [],

        # Final output
        "report": "",
        "citations": [],
        "final_report": "",
    }


    config = {
        "configurable": {
            "thread_id": "research-session-1"
        }
    }


    async with await get_checkpointer() as checkpointer:

        await checkpointer.setup()

        chatbot = builder.compile(
            checkpointer=checkpointer
        )

        final_state = await chatbot.ainvoke(
            initial_state,
            config=config,
        )


    return final_state


async def main():

    query = input(
        "Enter your research question: "
    )


    result = await run_research(query)


    print("\n" + "=" * 70)
    print("FINAL RESEARCH REPORT")
    print("=" * 70)

    print(
        result["final_report"]
    )


    print("\n" + "=" * 70)
    print("CITATIONS")
    print("=" * 70)


    for citation in result["citations"]:

        print(citation)


if __name__ == "__main__":
    asyncio.run(main())