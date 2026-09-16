import os
import asyncio
import json
import re
from typing import List

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field, ValidationError

from backend.graph.state import ResearchState


load_dotenv()


class ResearchPlan(BaseModel):

    tasks: List[str] = Field(
        description="Independent research tasks."
    )

    use_web: bool

    use_papers: bool

    web_searches: int = Field(
        ge=0,
        le=3
    )

    paper_searches: int = Field(
        ge=0,
        le=3
    )


planner_llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
    max_tokens=800,
    api_key=os.getenv("GROQ_API_KEY"),
)


def build_prompt(query: str, error_feedback: str = "") -> str:

    feedback_block = (
        f"\nYour previous response was invalid: {error_feedback}\n"
        "Fix this and return ONLY valid JSON this time.\n"
        if error_feedback else ""
    )

    return f"""
You are the planning agent in a multi-agent
AI research system.

Analyze this research question:

{query}
{feedback_block}
Create an efficient research strategy.

Rules:

1. Create 2 to 5 independent research tasks.
2. Avoid duplicate tasks.
3. Decide whether web research is required.
4. Decide whether academic paper research is required.
5. Use the minimum number of searches necessary.
6. web_searches must be between 0 and 3.
7. paper_searches must be between 0 and 3.
8. If use_web is false, web_searches must be 0.
9. If use_papers is false, paper_searches must be 0.

Return ONLY valid JSON.

Use exactly this structure:

{{
    "tasks": [
        "task 1",
        "task 2"
    ],
    "use_web": true,
    "use_papers": false,
    "web_searches": 2,
    "paper_searches": 0
}}
"""


async def planner_agent(
    state: ResearchState
) -> dict:

    query = state["query"]

    error_feedback = ""

    for attempt in range(3):

        prompt = build_prompt(query, error_feedback)

        try:

            response = await planner_llm.ainvoke(
                prompt
            )

            content = response.content.strip()

            content = re.sub(
                r"^```(?:json)?\s*",
                "",
                content
            )

            content = re.sub(
                r"\s*```$",
                "",
                content
            )

            data = json.loads(content)

            plan = ResearchPlan.model_validate(data)

            return {
                "research_plan": plan.tasks,
                "use_web": plan.use_web,
                "use_papers": plan.use_papers,
                "web_searches": plan.web_searches,
                "paper_searches": plan.paper_searches,
            }

        except (json.JSONDecodeError, ValidationError) as error:

            if attempt == 2:
                raise

            error_feedback = str(error)[:300]

            print(
                f"\n[Planner] Invalid plan output. "
                f"Retrying (attempt {attempt + 1}/3)..."
            )

        except Exception as error:

            error_message = str(error).lower()

            if "429" not in error_message:
                raise

            if attempt == 2:
                raise

            wait_time = 5 * (attempt + 1)

            print(
                f"\n[Planner] Rate limit reached. "
                f"Retrying in {wait_time} seconds..."
            )

            await asyncio.sleep(wait_time)