import pymupdf


def detect_visual_pages(pdf_path):
    pdf = pymupdf.open(pdf_path)

    visual_pages = []

    for page_index in range(len(pdf)):

        page_number = page_index + 1
        page = pdf[page_index]

        image_count = len(
            page.get_images(full=True)
        )

        text_blocks = page.get_text(
            "blocks"
        )

        page_area = page.rect.width * page.rect.height

        text_area = 0

        for block in text_blocks:

            x0, y0, x1, y1 = block[:4]

            block_area = (
                x1 - x0
            ) * (
                y1 - y0
            )

            text_area += block_area

        text_ratio = (
            text_area / page_area
            if page_area > 0
            else 0
        )

        # First-stage candidate detection.
        #
        # Images are a strong signal.
        # Low text coverage can indicate a visual-heavy page.
        #
        # We intentionally do NOT use drawing count
        # because normal PDF text can create many drawings.

        is_candidate = (
            image_count > 0
            or text_ratio < 0.35
        )

        if is_candidate:

            visual_pages.append(page_number)

            print(
                f"Candidate page {page_number} | "
                f"Images: {image_count} | "
                f"Text ratio: {text_ratio:.2f}"
            )

    pdf.close()

    return visual_pages


if __name__ == "__main__":

    PDF_PATH = "data/business_communication.pdf"

    pages = detect_visual_pages(
        PDF_PATH
    )

    print("\n==============================")
    print("VISUAL CANDIDATE PAGES")
    print("==============================")

    print(pages)

    print(
        f"\nTotal candidate pages: "
        f"{len(pages)}"
    )