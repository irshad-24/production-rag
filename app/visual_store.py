import json
from pathlib import Path

VISUAL_STORE_PATH = Path(
    "data/visual_chunks.json"
)


def visual_to_text(visual):
    lines = []

    lines.append(
        f"Visual Type: {visual.get('type', '')}"
    )

    lines.append(
        f"Title: {visual.get('title', '')}"
    )

    lines.append("")

    # Visual elements
    lines.append("Visual Elements:")

    for element in visual.get(
        "visual_elements",
        []
    ):
        lines.append(
            f"- {element}"
        )

    lines.append("")

    # Relationships
    lines.append("Relationships:")

    for relationship in visual.get(
        "relationships",
        []
    ):
        lines.append(
            f"- {relationship}"
        )

    # Table data
    table_data = visual.get(
        "table_data",
        {}
    )

    headers = table_data.get(
        "headers",
        []
    )

    rows = table_data.get(
        "rows",
        []
    )

    if headers:
        lines.append("")
        lines.append("Table Data:")

        lines.append(
            " | ".join(headers)
        )

        for row in rows:
            lines.append(
                " | ".join(row)
            )

    # Chart data
    chart_data = visual.get(
        "chart_data",
        {}
    )

    chart_type = chart_data.get(
        "chart_type",
        ""
    )

    categories = chart_data.get(
        "categories",
        []
    )

    series = chart_data.get(
        "series",
        []
    )

    values = chart_data.get(
        "values",
        []
    )

    if chart_type:
        lines.append("")
        lines.append("Chart Data:")

        lines.append(
            f"Chart Type: {chart_type}"
        )

        lines.append(
            f"X Axis: "
            f"{chart_data.get('x_axis', '')}"
        )

        lines.append(
            f"Y Axis: "
            f"{chart_data.get('y_axis', '')}"
        )

        if categories:
            lines.append(
                "Categories: "
                + ", ".join(categories)
            )

        if series:
            lines.append(
                "Series: "
                + ", ".join(series)
            )

        if values:
            lines.append(
                "Values: "
                + str(values)
            )

    lines.append("")

    lines.append("Visual Description:")

    lines.append(
        visual.get(
            "visual_description",
            ""
        )
    )

    return "\n".join(lines)


def save_visual_chunks(visual_results):

    visual_chunks = []

    for visual in visual_results:

        visual_text = visual_to_text(
            visual
        )

        visual_chunk = {

            "page_content": visual_text,

            "metadata": {

                "source": "visual",

                "source_type": "visual",

                "pdf_page":
                    visual.get(
                        "pdf_page"
                    ),

                "printed_page":
                    visual.get(
                        "printed_page"
                    ),

                "visual_type":
                    visual.get(
                        "type"
                    ),

                "title":
                    visual.get(
                        "title"
                    )
            }
        }

        visual_chunks.append(
            visual_chunk
        )

    VISUAL_STORE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        VISUAL_STORE_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            visual_chunks,
            file,
            indent=2,
            ensure_ascii=False
        )

    print(
        f"Saved {len(visual_chunks)} "
        f"visual chunks."
    )

    print(
        f"Visual store: "
        f"{VISUAL_STORE_PATH}"
    )


def load_visual_chunks():

    with open(
        VISUAL_STORE_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)