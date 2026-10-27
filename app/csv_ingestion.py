
import csv
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_and_split_csv(file_path):
    path = Path(file_path)

    if not path.is_file():
        raise FileNotFoundError(f"CSV not found: {path}")

    documents = []

    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)

        if not reader.fieldnames:
            raise ValueError("CSV must contain a header row.")

        for row_number, row in enumerate(reader, start=2):
            content = "\n".join(
                f"{key}: {value or ''}"
                for key, value in row.items()
                if key
            )

            if content.strip():
                documents.append(
                    Document(
                        page_content=content,
                        metadata={
                            "source_type": "csv",
                            "source": str(path),
                            "row": row_number,
                        },
                    )
                )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
    )

    chunks = splitter.split_documents(documents)

    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = f"csv_{index}"

    return chunks
