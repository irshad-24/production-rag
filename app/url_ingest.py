
import json
from pathlib import Path

from app.url_ingestion import load_and_split_url

OUTPUT_PATH = Path("data/url_chunks.json")


def ingest_url(url):
    chunks = load_and_split_url(url)

    data = [
        {
            "page_content": chunk.page_content,
            "metadata": chunk.metadata,
        }
        for chunk in chunks
    ]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Saved {len(data)} URL chunks to {OUTPUT_PATH}")


if __name__ == "__main__":
    url = input("Enter public webpage URL: ").strip()
    ingest_url(url)
