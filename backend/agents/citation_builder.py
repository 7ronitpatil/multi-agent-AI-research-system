import os
import json
import re

from dotenv import load_dotenv
from langchain_groq import ChatGroq

from backend.graph.state import ResearchState


load_dotenv()


citation_llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
    max_tokens=1000,
    api_key=os.getenv("GROQ_API_KEY"),
)


def apply_citation_marker(
    report: str,
    claim: str,
    markers: str
) -> str:
    """
    Insert citation markers after a claim. Tries an exact match
    first, then falls back to a whitespace-tolerant match, since
    the LLM won't always reproduce the claim byte-for-byte.
    """

    if claim in report:
        return report.replace(claim, f"{claim} {markers}", 1)

    # Whitespace-tolerant fallback: collapse runs of whitespace
    # on both sides before searching.
    pattern = re.escape(claim)
    pattern = re.sub(r"\\\s+", r"\\s+", pattern)

    match = re.search(pattern, report)

    if match:
        start, end = match.span()
        return (
            report[:end]
            + f" {markers}"
            + report[end:]
        )

    # No match found — leave the report unchanged rather than
    # silently pretending it worked.
    return report


async def citation_builder(
    state: ResearchState
) -> dict:

    report = state["report"]
    verified_evidence = state["verified_evidence"]

    sources = [
        {
            "source_id": index + 1,
            "title": item["title"],
            "url": item["url"],
            "source_type": item["source_type"],
            "verification": item["verification"],
        }
        for index, item in enumerate(
            verified_evidence
        )
    ]

    prompt = f"""
You are a citation-building agent.

Research report:

{report}

Available verified sources:

{json.dumps(sources, indent=2)}

For each important factual claim in the report,
identify the source or sources that support it.

Rules:

1. Only use the provided sources.
2. Do not invent URLs or sources.
3. Do not cite a source if it does not support the claim.
4. A claim may have multiple supporting sources.
5. If no source supports a claim, use an empty source_ids list.
6. Return valid JSON only.

Use this structure:

{{
    "citations": [
        {{
            "claim": "The factual claim",
            "source_ids": [1, 3]
        }}
    ]
}}
"""

    response = await citation_llm.ainvoke(
        prompt
    )

    try:

        content = response.content.strip()

        if content.startswith("```"):
            content = content.replace(
                "```json", ""
            ).replace(
                "```", ""
            ).strip()

        citation_data = json.loads(content)

    except (json.JSONDecodeError, TypeError):

        citation_data = {
            "citations": []
        }

    citations = citation_data.get(
        "citations",
        []
    )

    final_report = report

    for citation in citations:

        claim = citation.get(
            "claim",
            ""
        )

        source_ids = citation.get(
            "source_ids",
            []
        )

        if not claim or not source_ids:
            continue

        markers = " ".join(
            f"[{source_id}]"
            for source_id in source_ids
        )

        final_report = apply_citation_marker(
            final_report,
            claim,
            markers
        )

    final_report += "\n\n## Sources\n\n"

    for index, source in enumerate(
        verified_evidence,
        start=1
    ):

        final_report += (
            f"[{index}] "
            f"{source['title']} — "
            f"{source['url']}\n"
        )

    return {
        "citations": citations,
        "final_report": final_report,
    }