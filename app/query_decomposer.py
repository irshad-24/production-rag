
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY") or os.getenv("groq_api_key"),
)


def decompose_query(query):
    # Simple questions do not need an LLM call.
    words = query.strip().rstrip("?").split()

    if len(words) <= 8 and not any(
        word.lower() in {
            "compare", "difference", "differences",
            "relationship", "relationships",
            "explain both", "respectively",
        }
        for word in words
    ):
        return [query]

    prompt = f"""
Break the question into independent, standalone sub-questions
only when it requires multiple information needs.

Otherwise, return the original question unchanged.
Return one question per line, without numbering.
Do not answer the questions.

Question: {query}
"""

    response = llm.invoke(prompt)
    result = response.content.strip()

    sub_questions = [
        line.strip().lstrip("0123456789.-) ")
        for line in result.splitlines()
        if line.strip()
    ]

    return sub_questions or [query]
