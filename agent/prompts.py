"""The answer prompt lives in Langfuse Prompt Management; edit and version it there.

The `production` label is what the agent serves. FALLBACK_MESSAGES is only used when
Langfuse is unreachable and nothing is cached yet, and to create the prompt initially:

    uv run python -m agent.prompts
"""
from langfuse import get_client
from langfuse.model import ChatPromptClient

PROMPT_NAME = "company-helper-answer"

FALLBACK_MESSAGES = [
    {
        "role": "system",
        "content": (
            "You are a helpful assistant for company employees. Answer the question using "
            "only the documentation excerpts provided. Use information that is relevant to "
            "the question even if it doesn't match the wording exactly. Never add facts that "
            "aren't in the excerpts; if they contain nothing relevant, say you couldn't find "
            "it in the documentation. Be concise. Each excerpt is numbered; after every "
            "fact, cite the excerpt it came from like [1]. Cite multiple excerpts as [1][2]. "
            "Don't cite anything when you couldn't find the answer."
        ),
    },
    {
        "role": "user",
        "content": "Documentation excerpts:\n\n{{context}}\n\nQuestion: {{question}}",
    },
]


def get_answer_prompt() -> ChatPromptClient:
    """Cached by the SDK (60 s TTL, refreshed in the background), so cheap to call per request."""
    return get_client().get_prompt(PROMPT_NAME, type="chat", fallback=FALLBACK_MESSAGES)


def prompt_version(prompt: ChatPromptClient) -> str:
    return "fallback" if prompt.is_fallback else f"v{prompt.version}"


def _create_prompt() -> None:
    langfuse = get_client()
    try:
        existing = langfuse.get_prompt(PROMPT_NAME, type="chat", cache_ttl_seconds=0)
        print(f"Prompt '{PROMPT_NAME}' already exists (v{existing.version}); edit it in Langfuse.")
        return
    except Exception:
        pass
    created = langfuse.create_prompt(
        name=PROMPT_NAME,
        type="chat",
        prompt=FALLBACK_MESSAGES,
        labels=["production"],
        commit_message="Initial version, moved from agent/graph.py",
    )
    print(f"Created prompt '{PROMPT_NAME}' v{created.version} with label 'production'.")


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    _create_prompt()
