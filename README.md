# Multi-Agent AI Research System

An asynchronous multi-agent research system built with **Python, LangGraph, Groq, Tavily, arXiv, PostgreSQL, Docker, and LangSmith**.

The system takes a research question, creates a research strategy, performs web and academic-paper research in parallel when required, collects evidence, verifies it with an LLM-based fact checker, generates a research report, and attaches source citations.

---

## Architecture

```text
                         ┌──────────────────┐
                         │    User Query    │
                         └────────┬─────────┘
                                  │
                                  ▼
                         ┌──────────────────┐
                         │  Planner Agent   │
                         │   GPT-OSS-120B   │
                         └────────┬─────────┘
                                  │
                       Research Strategy
                                  │
                    ┌─────────────┴─────────────┐
                    │                           │
                    ▼                           ▼
             ┌──────────────┐            ┌──────────────┐
             │  Web Agent   │            │ Paper Agent  │
             │   Tavily     │            │    arXiv     │
             └──────┬───────┘            └──────┬───────┘
                    │                           │
                    └─────────────┬─────────────┘
                                  │
                                  ▼
                       ┌─────────────────────┐
                       │ Evidence Collector  │
                       └──────────┬──────────┘
                                  │
                                  ▼
                       ┌─────────────────────┐
                       │    Fact Checker     │
                       │   GPT-OSS-120B      │
                       └──────────┬──────────┘
                                  │
                                  ▼
                       ┌─────────────────────┐
                       │  Report Generator   │
                       │   GPT-OSS-120B      │
                       └──────────┬──────────┘
                                  │
                                  ▼
                       ┌─────────────────────┐
                       │  Citation Builder   │
                       └──────────┬──────────┘
                                  │
                                  ▼
                         ┌──────────────────┐
                         │  Final Report    │
                         │ + Source Links   │
                         └──────────────────┘


       ┌──────────────────────────────────────────┐
       │              Infrastructure              │
       │                                          │
       │  PostgreSQL ── LangGraph Checkpoints    │
       │  Docker ─────── Containerization         │
       │  LangSmith ──── LLM Observability        │
       └──────────────────────────────────────────┘
```

---

## How It Works

### 1. Planner Agent

The Planner analyzes the user's research question and creates a structured research strategy.

It determines:

* Research tasks
* Whether web research is required
* Whether academic papers are required
* Number of web searches
* Number of paper searches

The Planner uses structured output to make these decisions.

---

### 2. Web Research Agent

The Web Agent uses **Tavily** to search the internet.

It can perform multiple searches asynchronously:

```text
Research Task 1 ──┐
Research Task 2 ──┼──→ Tavily
Research Task 3 ──┘
```

Searches are executed concurrently using Python's `asyncio`.

---

### 3. Paper Research Agent

The Paper Agent searches **arXiv** for academic research relevant to the research tasks.

It collects:

* Paper title
* Authors
* Abstract
* Publication date
* Paper URL

arXiv requests are controlled with an asynchronous semaphore and retry configuration to avoid excessive requests.

---

### 4. Parallel Research

When both sources are required, Web and Paper research are executed as separate LangGraph branches.

```text
                  Planner
                     │
             ┌───────┴───────┐
             ▼               ▼
         Web Agent       Paper Agent
             │               │
             └───────┬───────┘
                     ▼
              Evidence Collector
```

This allows independent research work to happen concurrently.

---

### 5. Evidence Collector

The Evidence Collector converts results from different sources into a common evidence format.

Each evidence item contains information such as:

```text
source_type
task
title
content
url
```

This gives downstream agents a consistent structure.

---

### 6. Fact Checker

The Fact Checker evaluates collected evidence.

For each selected evidence item, it determines:

```text
SUPPORTED
WEAK
UNSUPPORTED
```

It also provides a short explanation.

The system limits the number of evidence items sent to the fact checker to control LLM API usage.

---

### 7. Report Generator

The Report Generator uses the verified evidence to create the research report.

It is instructed to:

* Answer the original research question
* Use only the provided evidence
* Avoid inventing sources or statistics
* Mention uncertainty
* Separate evidence from interpretation
* Produce a structured report

---

### 8. Citation Builder

The Citation Builder maps important factual claims in the report to the available verified sources.

Example:

```text
RAG combines retrieval with text generation [1].

## Sources

[1] Retrieval-Augmented Generation for
Knowledge-Intensive NLP Tasks — <source URL>
```

The system only allows citations from sources already collected during research.

---

## Key Features

* **Multi-agent architecture**
* **LangGraph-based orchestration**
* **Asynchronous Python execution**
* **Parallel Web + Paper research**
* **Dynamic research planning**
* **Evidence collection**
* **LLM-based fact checking**
* **Evidence-based report generation**
* **Automatic citation mapping**
* **PostgreSQL-backed LangGraph persistence**
* **Dockerized PostgreSQL**
* **LangSmith observability**
* **Groq-hosted open-weight LLMs**
* **Rate-limit protection and retries**

---

## Technology Stack

| Technology   | Purpose                                     |
| ------------ | ------------------------------------------- |
| Python       | Core application                            |
| LangGraph    | Agent orchestration                         |
| LangChain    | LLM integration                             |
| Groq         | LLM inference                               |
| GPT-OSS-120B | Planning, fact checking & report generation |
| Tavily       | Web research                                |
| arXiv        | Academic paper research                     |
| PostgreSQL   | Persistent LangGraph checkpoints            |
| Docker       | PostgreSQL containerization                 |
| LangSmith    | LLM tracing & observability                 |
| Pydantic     | Structured planner output                   |
| asyncio      | Asynchronous & concurrent execution         |

---

## Project Structure

```text
multi-agent-AI-research-system/
│
├── backend/
│   ├── agents/
│   │   ├── planner.py
│   │   ├── web_search.py
│   │   ├── paper_agent.py
│   │   ├── evidence_collector.py
│   │   ├── fact_checker.py
│   │   ├── report_generator.py
│   │   └── citation_builder.py
│   │
│   ├── database/
│   │   └── setup_db.py
│   │
│   ├── graph/
│   │   ├── graph.py
│   │   ├── state.py
│   │   └── utils.py
│   │
│   └── main.py
│
├── evaluation/
│   └── evaluate.py
│
├── tests/
│
├── .dockerignore
├── .env.example
├── .gitignore
├── DockerFile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/7ronitpatil/multi-agent-AI-research-system.git

cd multi-agent-AI-research-system
```

### 2. Create a virtual environment

Python 3.11 is recommended.

```bash
python -m venv myenv
```

Activate it on Windows:

```powershell
.\myenv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Environment Variables

Create a `.env` file in the project root.

Use `.env.example` as the template:

```text
GROQ_API_KEY=your_groq_api_key
TAVILY_API_KEY=your_tavily_api_key

LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your_langsmith_api_key
LANGSMITH_PROJECT=multi-agent-research-system

DATABASE_URL=postgresql://postgres:postgres@localhost:5432/research_system
```

**Never commit `.env` to GitHub.**

The repository already includes `.gitignore` to prevent this.

---

## PostgreSQL Setup

Start PostgreSQL using Docker:

```bash
docker compose up -d postgres
```

Verify that the container is running:

```bash
docker ps
```

Initialize the LangGraph checkpoint tables:

```bash
python -m backend.database.setup_db
```

You should see:

```text
Setting up PostgreSQL checkpoint database...
Database setup completed.
```

---

## Running the System

Start the research system with:

```bash
python -m backend.main
```

Enter a research question:

```text
Enter your research question:
What are different methods to reduce hallucinations in Large Language Models?
```

The system then executes:

```text
User Query
    ↓
Planner
    ↓
Research Strategy
    ↓
Web / Paper Research
    ↓
Evidence Collection
    ↓
Fact Checking
    ↓
Report Generation
    ↓
Citation Building
    ↓
Final Report
```

---

## Asynchronous Execution

The system uses Python `asyncio` to allow independent operations to execute concurrently.

For example, multiple web searches can run concurrently:

```python
tasks = [
    search_web(task)
    for task in research_tasks
]

results = await asyncio.gather(*tasks)
```

Similarly, Web and Paper research are represented as separate LangGraph branches.

This architecture is designed so that independent research operations do not need to execute strictly one after another.

---

## Persistence

LangGraph uses PostgreSQL as its checkpoint database.

```text
Research Graph
      │
      ▼
PostgreSQL
      │
      ├── Graph checkpoints
      ├── Thread state
      └── Persistent execution state
```

This provides a persistent backend for graph execution rather than relying only on in-memory state.

---

## Observability

The project uses **LangSmith** for LLM observability.

This allows development and debugging of:

* Agent execution
* LLM calls
* Latency
* Token usage
* Errors
* Graph execution traces

Enable tracing through:

```text
LANGSMITH_TRACING=true
```

and provide the required LangSmith API key in `.env`.

---

## Rate Limit Handling

LLM APIs can impose request and token limits.

The system therefore uses several safeguards:

* Limited number of research searches
* Limited evidence sent to the Fact Checker
* Limited LLM output tokens
* Concurrent request limits
* Retry and backoff handling for rate-limit errors

These controls are intended to make development runs more predictable under API limits.

---

## Evaluation

The `evaluation/` directory is intended for evaluating the research system.

Potential evaluation dimensions include:

### Answer Quality

Does the final report correctly answer the research question?

### Citation Correctness

Do citations actually support the claims they are attached to?

### Evidence Support

Are important claims backed by collected evidence?

### Hallucination Rate

How frequently does the system produce claims that are unsupported by the collected evidence?

### Latency

Compare sequential and parallel execution times.

### Example

```text
Sequential execution
        ↓
Measure latency

Parallel execution
        ↓
Measure latency

Compare results
```

The goal is to measure these properties rather than assume that parallel execution or fact checking automatically improves them.

---

## Design Principles

### Evidence First

The report generator receives research evidence instead of relying only on the model's internal knowledge.

### Separation of Responsibilities

Each agent has a specific responsibility:

```text
Planner          → Decide research strategy
Web Agent        → Search web
Paper Agent      → Search papers
Evidence         → Normalize sources
Fact Checker     → Verify evidence
Report Generator → Write report
Citation Builder → Map claims to sources
```

### Controlled LLM Usage

The system limits unnecessary LLM requests and generated output to make API usage manageable.

### Persistent State

PostgreSQL provides persistent LangGraph checkpoints.

### Observability

LLM calls and graph execution can be inspected through LangSmith.

---

## Future Improvements

* Better source ranking
* Deduplication of research results
* More robust citation placement
* Claim-level evidence verification
* Improved 429 retry handling
* Streaming final reports
* REST API interface
* Automated evaluation dataset
* RAGAS-based evaluation
* Human feedback collection
* More advanced research planning
* Additional academic databases

---

## Disclaimer

Research results depend on the quality and availability of external sources. The system is designed to provide evidence-backed research, but generated reports should still be reviewed when accuracy is important.

---

## Author

**Ronit Patil**

GitHub: `7ronitpatil`

Project: **Multi-Agent AI Research System**
