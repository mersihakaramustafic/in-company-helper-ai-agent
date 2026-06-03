import os
from dotenv import load_dotenv
from notion_client import Client

load_dotenv()

notion = Client(auth=os.environ["NOTION_API_KEY"])

def list_pages():
    response = notion.search(filter={"property": "object", "value": "page"})
    return response["results"]

if __name__ == "__main__":
    pages = list_pages()
    print(f"Found {len(pages)} pages")
    for page in pages:
        title_prop = page.get("properties", {}).get("title", {})
        title_parts = title_prop.get("title", [])
        title = title_parts[0]["plain_text"] if title_parts else "(untitled)"
        print(f"  - {title} ({page['id']})")
