import pymupdf
from pathlib import Path


PDF_PATH = "data/business_communication.pdf"
OUTPUT_DIR = Path("data/rendered_pages")


def render_all_pages(pdf_path, output_dir):
    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    pdf = pymupdf.open(pdf_path)

    print(f"Total PDF pages: {len(pdf)}")

    for page_index in range(len(pdf)):

        page_number = page_index + 1

        page = pdf[page_index]

        matrix = pymupdf.Matrix(2, 2)

        pixmap = page.get_pixmap(
            matrix=matrix,
            alpha=False
        )

        output_path = (
            output_dir
            / f"page_{page_number}.png"
        )

        pixmap.save(output_path)

        print(
            f"Rendered page {page_number}: "
            f"{output_path}"
        )

    pdf.close()

    print("\n==============================")
    print("ALL PAGES RENDERED")
    print("==============================")


if __name__ == "__main__":

    render_all_pages(
        pdf_path=PDF_PATH,
        output_dir=OUTPUT_DIR
    )