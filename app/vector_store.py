import json
from pathlib import Path

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings


COLLECTION_NAME = "business_communication"

UNIFIED_STORE_PATH = Path(
    "data/unified_chunks.json"
)

QDRANT_PATH = "qdrant_data"


def get_embeddings():

    return HuggingFaceEmbeddings(
        model_name=
        "sentence-transformers/all-MiniLM-L6-v2"
    )


def load_unified_chunks():

    with open(
        UNIFIED_STORE_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


def create_vector_store():

    unified_chunks = load_unified_chunks()

    documents = []

    for chunk in unified_chunks:

        document = Document(
            page_content=
                chunk["page_content"],

            metadata=
                chunk.get(
                    "metadata",
                    {}
                )
        )

        documents.append(document)

    embeddings = get_embeddings()

    vector_store = QdrantVectorStore.from_documents(
        documents=documents,
        embedding=embeddings,
        path=QDRANT_PATH,
        collection_name=COLLECTION_NAME
    )

    print(
        f"Added {len(documents)} documents "
        f"to Qdrant."
    )

    return vector_store


def load_vector_store():

    embeddings = get_embeddings()

    vector_store = (
        QdrantVectorStore
        .from_existing_collection(
            embedding=embeddings,
            path=QDRANT_PATH,
            collection_name=COLLECTION_NAME
        )
    )

    return vector_store


if __name__ == "__main__":

    print(
        "Creating Qdrant vector store "
        "from unified chunks..."
    )

    create_vector_store()

    print(
        "Qdrant vector store created successfully!"
    )