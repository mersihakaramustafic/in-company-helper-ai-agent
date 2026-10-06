import json
import os
from typing import Any, Dict, List

from langfuse.openai import AsyncOpenAI

# Deliberately stronger than the agent's model: a judge as weak as the model it
# grades tends to approve the same mistakes it would make itself.
JUDGE_MODEL = os.environ.get("EVAL_JUDGE_MODEL", "gpt-5.1")

FAILURE_MODES = [
    "none",               # correct, complete, grounded, correctly cited
    "incorrect",          # contradicts the expected answer or gets key facts wrong
    "incomplete",         # correct as far as it goes, but misses expected facts
    "unsupported",        # correct-looking, but makes claims the retrieved context doesn't support
    "wrong_citations",    # grounded answer, but citations point at the wrong excerpt / are missing
    "missed_answer",      # abstained although the answer was in the context
    "failed_to_abstain",  # answered (or invented) when it should have said it couldn't find it
]

JUDGE_SYSTEM_PROMPT = """\
You evaluate answers from an internal company assistant that must answer ONLY from \
retrieved company documentation and cite excerpts as [n].

You receive: the user question, the reference (expected) answer, evaluation notes, \
whether the question should be abstained from, the expected source documents, the \
numbered context excerpts the assistant actually saw, the assistant's answer, and the \
documents it cited.

Judge the answer against the RETRIEVED CONTEXT for grounding and against the \
REFERENCE ANSWER for correctness. The reference answer is authoritative; don't use \
outside knowledge about the company. Follow the evaluation notes when present.

Scores (integers 1-5):

answer_correctness - Are the stated facts right compared with the reference answer?
  5 all key facts match; 4 minor imprecision, nothing wrong; 3 partly right with one \
wrong or misleading detail; 2 mostly wrong; 1 wrong or contradicts the reference.
  An answer can be correct (5) and still incomplete.

completeness - Does it cover everything the reference answer requires?
  5 all required points; 4 misses a minor point; 3 misses an important point; \
2 covers only a small part; 1 doesn't address the question.

faithfulness - Is every claim supported by the retrieved context?
  5 every claim is supported; 4 trivial paraphrase stretch; 3 one claim not in the \
context; 2 several unsupported claims; 1 mostly unsupported.
  Judge this ONLY against the retrieved context, even if the claim happens to be \
true. A claim that matches the reference but isn't in the context is unsupported.

citation_accuracy - Do the [n] markers point at the excerpt that supports each claim?
  5 every factual claim is cited and each citation supports its claim; 4 one claim \
uncited; 3 a citation points at the wrong excerpt or several claims uncited; \
2 most citations wrong or missing; 1 no citations on a factual answer, or citations \
unrelated to the claims.
  For a correct abstention, 5 if it cites nothing (or only cites an excerpt for a \
related, supported statement).

relevance - Does it address what was asked, without padding or drifting?
  5 fully on point; 3 partly off-topic or padded; 1 doesn't address the question.

hallucination (bool) - true if the answer states ANY fact that is neither in the \
retrieved context nor in the reference answer (invented numbers, names, channels, \
policies, contacts). General phrasing ("I couldn't find...") is not a hallucination.

abstained (bool) - true if, for the main thing asked, the answer says the \
information isn't in the documentation or declines to answer. Mentioning related \
documented facts alongside that still counts as abstaining. Answering, correcting a \
false premise, or asking a clarifying question for an ambiguous question is NOT \
abstaining.

For should_abstain=true items, a clean abstention earns 5 on answer_correctness, \
completeness, faithfulness and relevance.

For should_abstain=false items, abstaining is a failure even when the retrieved \
context was empty or irrelevant: the empty context explains the failure but does not \
excuse it. Score answer_correctness and completeness 1 (2 if part of the answer was \
given), relevance at most 2, and use failure_mode "missed_answer". Faithfulness can \
still be 5 if nothing was invented.

failure_mode - the single most important problem, in this priority order:
  failed_to_abstain > missed_answer > incorrect > unsupported > incomplete > \
wrong_citations > none.
  Use "unsupported" when the facts may be right but aren't backed by the context; \
"wrong_citations" when the answer is grounded and correct but the citations are off.

reason - 1-3 sentences naming the specific facts or claims behind the scores.
"""

_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "answer_correctness", "faithfulness", "completeness", "citation_accuracy",
        "relevance", "hallucination", "abstained", "failure_mode", "reason",
    ],
    "properties": {
        **{k: {"type": "integer", "minimum": 1, "maximum": 5} for k in (
            "answer_correctness", "faithfulness", "completeness", "citation_accuracy", "relevance")},
        "hallucination": {"type": "boolean"},
        "abstained": {"type": "boolean"},
        "failure_mode": {"type": "string", "enum": FAILURE_MODES},
        "reason": {"type": "string"},
    },
}

_client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])


async def judge(
    *,
    question: str,
    expected_answer: str,
    evaluation_notes: str,
    should_abstain: bool,
    expected_sources: List[str],
    context: str,
    answer: str,
    cited_sources: List[str],
) -> Dict[str, Any]:
    user_prompt = f"""\
<question>{question}</question>
<reference_answer>{expected_answer}</reference_answer>
<evaluation_notes>{evaluation_notes or "none"}</evaluation_notes>
<should_abstain>{str(should_abstain).lower()}</should_abstain>
<expected_sources>{", ".join(expected_sources) or "none"}</expected_sources>
<retrieved_context>
{context or "(no excerpts passed the relevance threshold; the assistant saw no context)"}
</retrieved_context>
<assistant_answer>{answer}</assistant_answer>
<cited_sources>{", ".join(cited_sources) or "none"}</cited_sources>"""

    response = await _client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "evaluation", "strict": True, "schema": _SCHEMA},
        },
        name="judge",
    )
    result = json.loads(response.choices[0].message.content)
    result["should_abstain"] = should_abstain
    # The two abstention failure modes imply whether the agent abstained; trust them over
    # the separate `abstained` flag when the judge contradicts itself.
    if result["failure_mode"] == "missed_answer":
        result["abstained"] = True
    elif result["failure_mode"] == "failed_to_abstain":
        result["abstained"] = False
    # Derived rather than judged: whether the agent should abstain is known from the dataset.
    result["abstention_correct"] = result["abstained"] == should_abstain
    return result
