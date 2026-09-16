import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq

from backend.graph.state import ResearchState


load_dotenv()


report_llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
    max_tokens=1500,
    api_key=os.getenv("GROQ_API_KEY"),
)


async def report_generator(
    state: ResearchState
) -> dict:

    query = state["query"]
    verified_evidence = state["verified_evidence"]


    # --------------------------------------------------
    # No evidence fallback
    # --------------------------------------------------

    if not verified_evidence:

        return {
            "report": (
                "I could not generate a reliable research report, "
                "because no verified evidence was "
                "available for this question."
            )
        }


    # --------------------------------------------------
    # Prepare evidence
    # --------------------------------------------------

    evidence_text = "\n\n".join(
        [
            f"""
Source Type: {item["source_type"]}

Title:
{item["title"]}

URL:
{item["url"]}

Verification:
{item["verification"]}

Content:
{item["content"][:2500]}
"""
            for item in verified_evidence
        ]
    )


    # --------------------------------------------------
    # Generate report
    # --------------------------------------------------

    prompt = f"""
You are the report generation agent in a
multi-agent AI research system.

User research question:

{query}

Below is evidence collected from web sources
and academic papers and reviewed by a
fact-checking agent.

{evidence_text}

Write a clear and well-structured research report.

Requirements:

1. Directly answer the user's research question.
2. Use only the provided evidence.
3. Do not invent facts, sources, or statistics.
4. Clearly distinguish evidence from interpretation.
5. Mention uncertainty where evidence is weak
   or conflicting.
6. Organize the report with useful headings.
7. Do not create citations yourself.
8. Keep the report concise but informative.
9. If the evidence is insufficient to answer
   part of the question, explicitly say so.

Return only the research report.
"""


    response = await report_llm.ainvoke(
        prompt
    )


    return {
        "report": response.content
    }