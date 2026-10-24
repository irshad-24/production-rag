import os
import base64
import json

from dotenv import load_dotenv
from groq import Groq


load_dotenv()


client = Groq(
    api_key=os.getenv("groq_api_key")
)


def analyze_page(image_path, page_number):

    with open(
        image_path,
        "rb"
    ) as image_file:

        image_data = base64.b64encode(
            image_file.read()
        ).decode("utf-8")


    prompt = """
You are a visual document understanding system.

You are given a rendered screenshot of a PDF page.

Your task is to identify and understand the MAIN visual
content on the page.

The visual content can be:

- table
- chart
- graph
- diagram
- infographic
- figure
- none

IMPORTANT:

Do NOT simply summarize ordinary paragraph text.

Focus on the actual visual structure.

For TABLES:
- Identify column headers.
- Identify row labels.
- Extract cell values accurately.
- Preserve the table structure.
- Do not invent missing cells or values.

For CHARTS and GRAPHS:
- Identify the chart type.
- Identify title.
- Identify axis labels.
- Identify categories.
- Identify series.
- Extract visible values when they are explicitly shown.
- If exact values are not displayed, do NOT invent them.
- Describe the trend or relationship visible in the chart.

For DIAGRAMS:
- Identify important nodes, boxes, labels and shapes.
- Identify arrows and connections.
- Describe the relationships between elements.

For INFOGRAPHICS:
- Identify categories and their associated information.
- Preserve important labels and groupings.

If the page contains both normal text and a visual,
PRIORITIZE THE VISUAL.

Do not invent information.

Return ONLY valid JSON using exactly this structure:

{
    "type": "",
    "title": "",
    "visual_elements": [],
    "relationships": [],
    "table_data": {
        "headers": [],
        "rows": []
    },
    "chart_data": {
        "chart_type": "",
        "x_axis": "",
        "y_axis": "",
        "categories": [],
        "series": [],
        "values": []
    },
    "visual_description": ""
}

The "type" must be exactly one of:

"table"
"chart"
"graph"
"diagram"
"infographic"
"figure"
"none"

TABLE RULES:

If type is "table":

"table_data" must contain:

{
    "headers": [],
    "rows": []
}

Each row must be an array whose values correspond
to the headers.

If there is no table:

"table_data" must be:

{
    "headers": [],
    "rows": []
}


CHART RULES:

If type is "chart" or "graph":

"chart_data" must contain:

{
    "chart_type": "",
    "x_axis": "",
    "y_axis": "",
    "categories": [],
    "series": [],
    "values": []
}

The chart_type may be:

"bar"
"line"
"pie"
"scatter"
"area"
"other"

If exact numerical values are visible, extract them.

If exact values are NOT visible, keep "values" empty
and describe the visible trend instead.

Do NOT estimate numerical values.

If there is no chart:

"chart_data" must be:

{
    "chart_type": "",
    "x_axis": "",
    "y_axis": "",
    "categories": [],
    "series": [],
    "values": []
}


For "visual_elements":

List the important objects, labels, categories,
or components visible in the visual.

For "relationships":

Describe important relationships between the visual
elements.

For "visual_description":

Give a concise description of what the visual represents.

If there is no meaningful visual content on the page,
return:

{
    "type": "none",
    "title": "",
    "visual_elements": [],
    "relationships": [],
    "table_data": {
        "headers": [],
        "rows": []
    },
    "chart_data": {
        "chart_type": "",
        "x_axis": "",
        "y_axis": "",
        "categories": [],
        "series": [],
        "values": []
    },
    "visual_description": ""
}

Again:

DO NOT focus on ordinary paragraph text.

Focus on the actual visual structure.
"""


    response = client.chat.completions.create(
        model="qwen/qwen3.8-27b",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": (
                                "data:image/png;base64,"
                                + image_data
                            )
                        }
                    }
                ]
            }
        ],
        temperature=0,
        max_tokens=900,
        response_format={
            "type": "json_object"
        }
    )


    result = response.choices[0].message.content

    result = json.loads(
        result
    )

    result["page"] = page_number

    return result


if __name__ == "__main__":

    TEST_IMAGE = (
        "data/rendered_pages/page_7.png"
    )

    result = analyze_page(
        image_path=TEST_IMAGE,
        page_number=7
    )

    print(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False
        )
    )