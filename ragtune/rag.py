"""The generation half of RAG: pack retrieved passages into a prompt, call a model, and check its citations.

The model is a plain function `llm(prompt) -> str`, so any provider can be plugged in. None is bundled, because
a live model call could not be tested here. Without one, `answer` falls back to an extractive answer: the sentence
from the top passages that best matches the question, with its source.
"""

import re

CITATION_RE = re.compile(r"\[(\d+)\]")
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
WORD_RE = re.compile(r"[a-z0-9]+")
STOP = frozenset("a an the of and or to in on for is are was were be by with as at from that this it which what who when where how why".split())

INSTRUCTIONS = (
    "Answer the question using only the numbered passages below. "
    "Cite the passages you used like [1] or [2]. "
    "If the passages do not contain the answer, say you do not know."
)


def estimate_tokens(text):
    """A rough guess (about 1.3 tokens per word), good enough for budgeting a prompt, not for billing."""
    return max(1, round(len(text.split()) * 1.3))


def build_prompt(question, hits, token_budget=1500):
    """Pack passages best-first until the budget is used. Returns (prompt, hits_actually_included)."""
    header = f"{INSTRUCTIONS}\n\n"
    footer = f"\nQuestion: {question}\nAnswer:"
    used = estimate_tokens(header) + estimate_tokens(footer)
    blocks, included = [], []
    for hit in hits:
        block = f"[{len(included) + 1}] {hit['title']}: {hit['text']}" if hit["title"] else f"[{len(included) + 1}] {hit['text']}"
        cost = estimate_tokens(block)
        if used + cost > token_budget:
            if not included:  # always keep at least the best passage, trimmed to fit
                words = block.split()[: max(1, int((token_budget - used) / 1.3))]
                blocks.append(" ".join(words))
                included.append(hit)
            break
        blocks.append(block)
        included.append(hit)
        used += cost
    return header + "\n\n".join(blocks) + "\n" + footer, included


def stem(word):
    """A crude stemmer so "erupt" matches "erupted" and "bury" matches "buried". Not Porter, just enough for matching."""
    for suffix in ("ing", "ed", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            word = word[: -len(suffix)]
            break
    if word.endswith("e") and len(word) > 3:  # "bake" and "baking" both become "bak"
        word = word[:-1]
    return word[:-1] + "i" if word.endswith("y") else word


def _words(text):
    return {stem(w) for w in WORD_RE.findall(text.lower()) if w not in STOP}


def extractive_answer(question, hits, top_passages=3):
    """The single sentence, from the top passages, that shares the most words with the question."""
    wanted = _words(question)
    best = (0, -1, "", 0)  # overlap, -rank (earlier wins ties), sentence, rank
    for rank, hit in enumerate(hits[:top_passages], start=1):
        for sentence in SENTENCE_RE.split(hit["text"].strip()):
            overlap = len(wanted & _words(sentence))
            if (overlap, -rank) > (best[0], best[1]) and sentence.strip():
                best = (overlap, -rank, sentence.strip(), rank)
    if best[0] == 0:
        return {"answer": "I do not know: no retrieved passage matches the question.", "citations": [], "method": "extractive"}
    return {"answer": best[2], "citations": [best[3]], "method": "extractive"}


def check_citations(answer, n_passages):
    """Split the [n] markers an answer used into valid ones and ones that point at passages that were never shown."""
    cited = [int(n) for n in CITATION_RE.findall(answer)]
    valid = sorted({n for n in cited if 1 <= n <= n_passages})
    invalid = sorted({n for n in cited if not 1 <= n <= n_passages})
    return valid, invalid


def answer(question, hits, llm=None, token_budget=1500, k=5):
    """Answer from the retrieved hits. With an `llm` function the model writes the answer and its citations are checked."""
    hits = hits[:k]
    if llm is None:
        result = extractive_answer(question, hits)
        result["sources"] = [hits[n - 1] for n in result["citations"]]
        return result
    prompt, included = build_prompt(question, hits, token_budget)
    text = llm(prompt).strip()
    valid, invalid = check_citations(text, len(included))
    return {
        "answer": text, "citations": valid, "invalid_citations": invalid, "method": "llm",
        "sources": [included[n - 1] for n in valid], "prompt_passages": len(included),
    }
