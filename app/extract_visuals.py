import json
from pathlib import Path

from visual_detector import detect_visual_pages
from vision_extractor import analyze_page
from visual_store import save_visual_chunks


PDF_PATH = "data/business_communication.pdf"

RENDERED_PAGES_DIR = Path(
    "data/rendered_pages"
)


def extract_visuals():

    # Detect candidate pages automatically
    candidate_pages = detect_visual_pages(
        PDF_PATH
    )

    print("\n==============================")
    print("CANDIDATE PAGES")
    print("==============================")

    print(candidate_pages)

    visual_results = []

    for pdf_page in candidate_pages:

        image_path = (
            RENDERED_PAGES_DIR
            / f"page_{pdf_page}.png"
        )

        print("\n==============================")
        print(
            f"Analyzing PDF page {pdf_page}"
        )
        print("==============================")

        if not image_path.exists():

            print(
                f"Image not found: {image_path}"
            )

            continue

        result = analyze_page(
            image_path=str(image_path),
            page_number=pdf_page
        )

        # Store PDF page number
        result["pdf_page"] = pdf_page

        # At this stage we don't know the
        # printed page number automatically.
        # Keep it empty for now.
        result["printed_page"] = None

        print(
            json.dumps(
                result,
                indent=2,
                ensure_ascii=False
            )
        )

        # Only keep actual visual content.
        if result.get("type") != "none":

            visual_results.append(
                result
            )

        else:

            print(
                f"Page {pdf_page} does not contain "
                f"a meaningful visual."
            )

    return visual_results


if __name__ == "__main__":

    print(
        "Starting automatic visual extraction..."
    )

    visual_results = extract_visuals()

    print("\n==============================")
    print("VISUAL EXTRACTION COMPLETE")
    print("==============================")

    print(
        f"Total visuals extracted: "
        f"{len(visual_results)}"
    )

    save_visual_chunks(
        visual_results
    )