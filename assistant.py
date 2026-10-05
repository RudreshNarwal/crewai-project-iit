"""A Manager agent routes each question to PDF / Web / Weather specialists,
then combines their answers.

Flow of one question:
  1. route()  - the Manager returns a Plan: which specialists, and why.
  2. answer() - a Crew runs one Task per chosen specialist.
  3. If more than one specialist ran, the Manager writes the combined answer.
"""
import os
from typing import Literal

os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")  # no first-run "view traces?" prompt

import httpx
from crewai import LLM, Agent, Crew, Task
from crewai.tools import tool
from ddgs import DDGS
from dotenv import load_dotenv
from pydantic import BaseModel

import rag

load_dotenv()

MODEL = "openrouter/openai/gpt-4o-mini"  # reads OPENROUTER_API_KEY


class Step(BaseModel):
    agent: Literal["pdf", "web", "weather"]
    question: str


class Plan(BaseModel):
    reason: str
    steps: list[Step]


ROUTING_PROMPT = """Decide which specialists should handle the user's question.

- pdf: questions about the user's uploaded document. {pdf_status}
- weather: current weather in a named city. Put the city in the sub-question.
- web: everything else, including general knowledge and current events.

Use more than one step only when the question has clearly separate parts.
Give each step a self-contained sub-question, and give one short reason for
your choice.

Question: {question}"""


def make_llm() -> LLM:
    return LLM(model=MODEL, temperature=0.2, max_tokens=1200)


def make_manager() -> Agent:
    return Agent(
        role="Manager",
        goal="Send each question to the right specialist and combine their answers",
        backstory="You lead a small research team and know exactly what each member is good at.",
        llm=make_llm(),
    )


def route(question: str, has_pdf: bool) -> Plan:
    """Ask the Manager which specialists should answer."""
    pdf_status = (
        "A PDF is uploaded."
        if has_pdf
        else "No PDF is uploaded yet; still choose pdf if they ask about their document."
    )
    prompt = ROUTING_PROMPT.format(pdf_status=pdf_status, question=question)
    plan = make_manager().kickoff(prompt, response_format=Plan).pydantic
    # .pydantic is None (no exception) when the model's reply can't be parsed.
    if not plan or not plan.steps:
        return Plan(
            reason="The Manager gave no usable plan, so this fell back to web search.",
            steps=[Step(agent="web", question=question)],
        )
    return plan


def make_tools(collection, sources: list[str]) -> dict:
    """One tool per specialist. Each records what it used in `sources`, so the
    citations shown to the user come from the tools and not the LLM's memory."""

    @tool("Search PDF")
    def search_pdf(query: str) -> str:
        """Search the uploaded PDF. Returns the most relevant passages with page numbers."""
        if collection is None:
            return "No PDF has been uploaded. Ask the user to upload one in the sidebar."
        hits = rag.search(collection, query)
        sources.extend(f"PDF page {page}" for page, _ in hits)
        return "\n\n".join(f"[page {page}] {text}" for page, text in hits)

    @tool("Web Search")
    def web_search(query: str) -> str:
        """Search the web. Returns titles, snippets and links."""
        try:
            results = DDGS().text(query, max_results=5)
        except Exception as error:  # ddgs raises on rate limits and on zero results
            return f"Web search failed: {error}"
        sources.extend(r["href"] for r in results)
        return "\n\n".join(f"{r['title']}\n{r['body']}\n{r['href']}" for r in results)

    @tool("Get Weather")
    def get_weather(city: str) -> str:
        """Current weather for a city: temperature, humidity, wind speed and condition."""
        response = httpx.get(
            "https://api.openweathermap.org/data/2.5/weather",
            params={"q": city, "appid": os.getenv("OPENWEATHER_API_KEY"), "units": "metric"},
            timeout=10,
        )
        data = response.json()
        if response.status_code != 200:
            return f"Weather lookup failed for {city}: {data.get('message', response.status_code)}"
        sources.append(f"OpenWeather: {data['name']}")
        return (
            f"{data['name']}: temperature {data['main']['temp']} °C, "
            f"humidity {data['main']['humidity']}%, "
            f"wind speed {data['wind']['speed']} m/s, "
            f"condition {data['weather'][0]['description']}"
        )

    return {"pdf": search_pdf, "web": web_search, "weather": get_weather}


def make_specialists(tools: dict) -> dict:
    llm = make_llm()
    return {
        "pdf": Agent(
            role="PDF Analyst",
            goal="Answer questions using only the uploaded PDF, citing page numbers",
            backstory="You read documents carefully and never state what the document does not say.",
            tools=[tools["pdf"]],
            llm=llm,
        ),
        "web": Agent(
            role="Web Researcher",
            goal="Answer general and current questions from web search results, with source links",
            backstory="You verify claims against search results and always show where they came from.",
            tools=[tools["web"]],
            llm=llm,
        ),
        "weather": Agent(
            role="Weather Reporter",
            goal="Report the current temperature, humidity, wind speed and condition for a city",
            backstory="You report live weather readings exactly as the weather service returns them.",
            tools=[tools["weather"]],
            llm=llm,
        ),
    }


def answer(question: str, collection=None) -> dict:
    """Route the question, run the chosen specialists, return answer + routing + sources."""
    plan = route(question, has_pdf=collection is not None)
    sources: list[str] = []
    specialists = make_specialists(make_tools(collection, sources))

    tasks = [
        Task(
            description=step.question,
            expected_output="A direct answer based on your tool's results, with page numbers or links where you have them.",
            agent=specialists[step.agent],
        )
        for step in plan.steps
    ]
    agents = [task.agent for task in tasks]
    if len(tasks) > 1:
        manager = make_manager()
        agents.append(manager)
        tasks.append(
            Task(
                description=f"Combine the specialists' findings into one answer to: {question}",
                expected_output="One clear answer covering every part of the question, keeping page numbers and links.",
                agent=manager,
                context=list(tasks),
            )
        )

    result = Crew(agents=agents, tasks=tasks, verbose=True).kickoff()
    return {
        "routes": [step.agent for step in plan.steps],
        "reason": plan.reason,
        "answer": result.raw,
        "sources": list(dict.fromkeys(sources)),  # de-duplicated, order kept
    }
