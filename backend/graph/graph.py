import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from backend.graph.state import ResearchState

from backend.agents.planner import planner_agent
from backend.agents.web_search import web_research_agent
from backend.agents.paper_agent import paper_research_agent
from backend.agents.evidence_collector import evidence_collector
from backend.agents.fact_checker import fact_checker_agent
from backend.agents.report_generator import report_generator
from backend.agents.citation_builder import citation_builder


load_dotenv()


DB_URI = os.getenv("DATABASE_URL")

if not DB_URI:
    raise RuntimeError(
        "DATABASE_URL environment variable is not set. "
        "Add it to your .env file."
    )


# --------------------------------------------------
# Research routing
# --------------------------------------------------

def route_research(state: ResearchState):
    """
    Decide which research branches to fan out to based on
    the planner's decision. Falls back to evidence_collector
    directly if neither web nor paper research is needed.
    """

    use_web = state.get("use_web", False)
    use_papers = state.get("use_papers", False)

    routes = []

    if use_web:
        routes.append("web_research")

    if use_papers:
        routes.append("paper_research")

    if not routes:
        routes.append("evidence_collector")

    return routes


# --------------------------------------------------
# Build graph (nodes + edges only, not compiled yet)
# --------------------------------------------------

def build_graph_definition() -> StateGraph:

    builder = StateGraph(ResearchState)

    # ----------------------------------------------
    # Nodes
    # ----------------------------------------------

    builder.add_node(
        "planner",
        planner_agent
    )

    builder.add_node(
        "web_research",
        web_research_agent
    )

    builder.add_node(
        "paper_research",
        paper_research_agent
    )

    builder.add_node(
        "evidence_collector",
        evidence_collector
    )

    builder.add_node(
        "fact_checker",
        fact_checker_agent
    )

    builder.add_node(
        "report_generator",
        report_generator
    )

    builder.add_node(
        "citation_builder",
        citation_builder
    )

    # ----------------------------------------------
    # Edges
    # ----------------------------------------------

    builder.add_edge(
        START,
        "planner"
    )

    # Planner decides which research agents are required.
    builder.add_conditional_edges(
        "planner",
        route_research,
        {
            "web_research": "web_research",
            "paper_research": "paper_research",
            "evidence_collector": "evidence_collector",
        },
    )

    # Both research agents lead to evidence collection.
    #
    # If both were selected by the Planner, LangGraph
    # waits for both branches to complete before moving
    # forward.
    builder.add_edge(
        "web_research",
        "evidence_collector"
    )

    builder.add_edge(
        "paper_research",
        "evidence_collector"
    )

    # ----------------------------------------------
    # Evidence -> Fact Checker -> Report -> Citations
    # ----------------------------------------------

    builder.add_edge(
        "evidence_collector",
        "fact_checker"
    )

    builder.add_edge(
        "fact_checker",
        "report_generator"
    )

    builder.add_edge(
        "report_generator",
        "citation_builder"
    )

    builder.add_edge(
        "citation_builder",
        END
    )

    return builder


# --------------------------------------------------
# Compiled graph, with a properly scoped Postgres
# checkpointer connection.
# --------------------------------------------------

@asynccontextmanager
async def get_graph():
    """
    Async context manager that yields a compiled graph backed
    by an AsyncPostgresSaver checkpointer.

    Usage:

        async with get_graph() as graph:
            result = await graph.ainvoke(initial_state, config=config)

    The checkpointer's underlying connection is only valid for
    the lifetime of this `async with` block, so all graph
    invocations must happen inside it.
    """

    builder = build_graph_definition()

    async with AsyncPostgresSaver.from_conn_string(DB_URI) as checkpointer:

        # Creates the checkpoint tables on first run; safe to
        # call on every startup, it's a no-op if they exist.
        await checkpointer.setup()

        graph = builder.compile(checkpointer=checkpointer)

        yield graph
        
import uuid

async def run_research(query: str) -> dict:

    initial_state = {
        "query": query,
        "web_results": [],
        "paper_results": [],
        "evidence": [],
        "verified_evidence": [],
    }

    config = {
        "configurable": {
            "thread_id": str(uuid.uuid4())
        }
    }

    async with get_graph() as graph:
        result = await graph.ainvoke(initial_state, config=config)

    return result