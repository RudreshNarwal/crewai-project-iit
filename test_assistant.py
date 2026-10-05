"""Checks. Run: python test_assistant.py

The first two run offline. The routing check calls the LLM, so it only runs
when OPENROUTER_API_KEY is set.
"""
import os

import assistant
import rag

ROUTING_CASES = [
    ("What's the weather in Delhi right now?", {"weather"}),
    ("Is it raining in London?", {"weather"}),
    ("Who won the most recent FIFA World Cup?", {"web"}),
    ("What is the latest version of Python?", {"web"}),
    ("Summarise the uploaded document.", {"pdf"}),
    ("What does my PDF say about the methodology?", {"pdf"}),
    ("Summarise my PDF's conclusion and tell me the weather in Mumbai.", {"pdf", "weather"}),
    ("What's the weather in Paris, and what is the latest news about the Louvre?", {"weather", "web"}),
]


def test_chunk():
    letters = "".join(chr(97 + i % 26) for i in range(1000))
    chunks = rag.chunk([letters, "   ", "short page"])
    assert [page for page, _ in chunks] == [1, 1, 3], "blank page skipped, page numbers kept"
    assert chunks[0][1][-100:] == chunks[1][1][:100], "consecutive chunks overlap by 100 characters"
    assert "".join([chunks[0][1], chunks[1][1][100:]]) == letters, "no text lost"


def test_search():
    collection = rag.build_index([
        "The Eiffel Tower is a wrought-iron tower in Paris, completed in 1889.",
        "Photosynthesis is how plants turn sunlight, water and carbon dioxide into sugar.",
    ])
    page, _ = rag.search(collection, "How do plants make food?", k=1)[0]
    assert page == 2
    assert len(rag.search(collection, "tower", k=10)) == 2, "k larger than the index is fine"


def test_routing():
    for question, expected in ROUTING_CASES:
        got = {step.agent for step in assistant.route(question, has_pdf=True).steps}
        assert got == expected, f"{question!r}: expected {expected}, got {got}"


if __name__ == "__main__":
    test_chunk()
    test_search()
    print("chunking and retrieval: ok")
    if os.getenv("OPENROUTER_API_KEY"):
        test_routing()
        print(f"routing: {len(ROUTING_CASES)}/{len(ROUTING_CASES)} ok")
    else:
        print("routing: skipped (set OPENROUTER_API_KEY to run it)")
