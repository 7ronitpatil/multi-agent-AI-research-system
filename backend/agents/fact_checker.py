import os
import asyncio

from dotenv import load_dotenv
from langchain_groq import ChatGroq

from backend.graph.state import ResearchState


load_dotenv()


fact_checker_llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
    max_tokens=300,
    api_key=os.getenv("GROQ_API_KEY"),
)


FACT_CHECK_SEMAPHORE = asyncio.Semaphore(2)

MAX_EVIDENCE_TO_CHECK = 8


def select_representative_evidence(
    evidence: list,
    limit: int
) -> list:
    """
    Round-robin evidence by task so that every task (and both
    web + paper sources) gets a fair shot at being fact-checked,
    instead of just taking the first `limit` items in order.
    """

    if len(evidence) <= limit:
        return evidence

    groups: dict[str, list] = {}

    for item in evidence:
        groups.setdefault(item["task"], []).append(item)

    selected = []
    group_lists = list(groups.values())
    index = 0

    while len(selected) < limit and any(group_lists):

        group = group_lists[index % len(group_lists)]

        if group:
            selected.append(group.pop(0))

        index += 1

        # Drop empty groups to avoid an infinite loop.
        group_lists = [g for g in group_lists if g]

    return selected


async def verify_evidence(item: dict) -> dict:
    """
    Verify one piece of research evidence.
    """

    prompt = f"""
You are a fact-checking agent.

Evaluate the following research evidence.

Research task:
{item["task"]}

Source title:
{item["title"]}

Source content:
{item["content"][:2000]}

Determine:

1. Does the evidence directly support the research task?
2. Does the source content appear reliable?
3. Should this evidence be used in the final report?

Return exactly:

VERDICT: SUPPORTED / WEAK / UNSUPPORTED

REASON:
A short explanation.
"""

    for attempt in range(3):

        try:

            async with FACT_CHECK_SEMAPHORE:

                response = await fact_checker_llm.ainvoke(
                    prompt
                )

            return {
                **item,
                "verification": response.content,
            }

        except Exception as error:

            error_message = str(error).lower()
            is_rate_limit = "429" in error_message

            if attempt == 2 or not is_rate_limit:

                reason = (
                    "LLM rate limit was exceeded."
                    if is_rate_limit
                    else f"Unexpected error: {error}"
                )

                print(
                    f"\n[Fact Checker] Failed to verify "
                    f"'{item['title']}': {error}"
                )

                # Fail soft: never let one bad item take down
                # the whole asyncio.gather() batch.
                return {
                    **item,
                    "verification": (
                        f"VERDICT: WEAK\n"
                        f"REASON: Fact checking failed. {reason}"
                    ),
                }

            wait_time = 5 * (attempt + 1)

            print(
                f"\n[Fact Checker] Rate limit reached. "
                f"Retrying in {wait_time} seconds..."
            )

            await asyncio.sleep(wait_time)


async def fact_checker_agent(
    state: ResearchState
) -> dict:
    """
    Verify a limited number of evidence items in parallel,
    sampled evenly across tasks/source types.
    """

    evidence = state["evidence"]
    print(f"\n[Fact Checker] Evidence received: {len(evidence)}")

    evidence_to_check = select_representative_evidence(
        evidence,
        MAX_EVIDENCE_TO_CHECK
    )

    verification_tasks = [
        verify_evidence(item)
        for item in evidence_to_check
    ]

    verified_evidence = await asyncio.gather(
        *verification_tasks
    )

    print(
        f"[Fact Checker] Evidence verified: "
        f"{len(verified_evidence)}"
    )

    return {
        "verified_evidence": verified_evidence
    }