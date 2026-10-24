import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()


llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
    api_key=os.getenv("groq_api_key")
)


def generate_answer(query, retrieved_documents):

    context = "\n\n".join(
        document.page_content
        for document in retrieved_documents
    )

    prompt = f"""
You are a helpful assistant answering questions about a Business Communication document.

Answer the question using ONLY the information provided in the context.

If the answer cannot be found in the context, say:
"I could not find the answer in the provided document."

Do not make up information.

Context:
{context}

Question:
{query}

Answer:
"""

    response = llm.invoke(prompt)

    return response.content