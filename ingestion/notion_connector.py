import os
from notion_client import Client

notion = Client(auth=os.environ["NOTION_API_KEY"])


def get_all_pages():
    results = []
    has_more = True
    start_cursor = None

    while has_more:
        params = {"filter": {"property": "object", "value": "page"}}
        if start_cursor:
            params["start_cursor"] = start_cursor
        response = notion.search(**params)
        results.extend(response["results"])
        has_more = response.get("has_more", False)
        start_cursor = response.get("next_cursor")

    return results


def get_page_title(page: dict) -> str:
    props = page.get("properties", {})
    for key in ("Name", "title", "Title"):
        parts = props.get(key, {}).get("title", [])
        if parts:
            return parts[0]["plain_text"]
    return "(untitled)"


def get_page_url(page: dict) -> str:
    return page.get("url", "")


def get_page_last_updated(page: dict) -> str:
    return page.get("last_edited_time", "")


def get_page_content(page_id: str) -> str:
    blocks = _fetch_blocks(page_id)
    return _blocks_to_text(blocks)


def _fetch_blocks(block_id: str) -> list:
    results = []
    has_more = True
    start_cursor = None

    while has_more:
        params = {}
        if start_cursor:
            params["start_cursor"] = start_cursor
        response = notion.blocks.children.list(block_id=block_id, **params)
        results.extend(response["results"])
        has_more = response.get("has_more", False)
        start_cursor = response.get("next_cursor")

    return results


def _blocks_to_text(blocks: list) -> str:
    lines = []
    for block in blocks:
        btype = block["type"]
        data = block.get(btype, {})
        rich_text = data.get("rich_text", [])
        text = "".join(t["plain_text"] for t in rich_text)

        if text:
            if btype == "heading_1":
                lines.append(f"# {text}")
            elif btype == "heading_2":
                lines.append(f"## {text}")
            elif btype == "heading_3":
                lines.append(f"### {text}")
            elif btype in ("bulleted_list_item", "numbered_list_item", "to_do"):
                lines.append(f"- {text}")
            elif btype == "code":
                lines.append(f"```\n{text}\n```")
            elif btype == "quote":
                lines.append(f"> {text}")
            else:
                lines.append(text)

        if block.get("has_children"):
            child_text = _blocks_to_text(_fetch_blocks(block["id"]))
            if child_text:
                lines.append(child_text)

    return "\n".join(lines)
