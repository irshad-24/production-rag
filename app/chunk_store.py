import json
from pathlib import Path


CHUNK_STORE_PATH = Path(
    "data/chunks.json"
)

VISUAL_STORE_PATH = Path(
    "data/visual_chunks.json"
)

UNIFIED_STORE_PATH = Path(
    "data/unified_chunks.json"
)


def save_chunks(chunks):

    data = []

    for chunk in chunks:

        data.append({
            "page_content": chunk.page_content,
            "metadata": {
                **chunk.metadata,
                "source_type": "text"
            }
        })

    with open(
        CHUNK_STORE_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            data,
            file,
            indent=2,
            ensure_ascii=False
        )


def load_chunks():

    with open(
        CHUNK_STORE_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


def load_visual_chunks():

    with open(
        VISUAL_STORE_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


def load_unified_chunks():

    with open(
        UNIFIED_STORE_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


def create_unified_chunks():

    text_chunks = load_chunks()

    visual_chunks = load_visual_chunks()

    unified_chunks = []

    # -------------------------
    # TEXT CHUNKS
    # -------------------------

    for chunk in text_chunks:

        unified_chunks.append({
            "page_content":
                chunk["page_content"],

            "metadata": {
                **chunk.get(
                    "metadata",
                    {}
                ),

                "source_type": "text"
            }
        })

    # -------------------------
    # VISUAL CHUNKS
    # -------------------------

    for index, chunk in enumerate(
        visual_chunks
    ):

        metadata = {
            **chunk.get(
                "metadata",
                {}
            ),

            "source_type": "visual",

            # Give every visual a unique
            # chunk ID compatible with
            # the retrieval pipeline.
            "chunk_id":
                f"visual_{index}",

            # Visual chunks don't have
            # normal text-chunk page metadata.
            # Use PDF page as their page.
            "page":
                chunk.get(
                    "metadata",
                    {}
                ).get(
                    "pdf_page"
                )
        }

        unified_chunks.append({
            "page_content":
                chunk["page_content"],

            "metadata": metadata
        })

    # -------------------------
    # SAVE
    # -------------------------

    UNIFIED_STORE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        UNIFIED_STORE_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            unified_chunks,
            file,
            indent=2,
            ensure_ascii=False
        )

    print(
        f"Text chunks: {len(text_chunks)}"
    )

    print(
        f"Visual chunks: "
        f"{len(visual_chunks)}"
    )

    print(
        f"Unified chunks: "
        f"{len(unified_chunks)}"
    )

    print(
        f"Saved to: "
        f"{UNIFIED_STORE_PATH}"
    )


if __name__ == "__main__":

    create_unified_chunks()