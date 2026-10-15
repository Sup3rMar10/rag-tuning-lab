"""A small deterministic corpus with questions and relevance labels.

Used by the tests and by `python -m ragtune demo`, so the whole pipeline can be tried without downloading the 1.5 GB corpus.
Each topic has its own vocabulary. A passage is relevant to a question when it comes from that topic AND contains the
question's key fact, so an exact-match engine and an embedding model can both be checked against known answers.
"""

import random

TOPICS = {
    "volcano": ["magma", "crater", "eruption", "lava", "ash", "caldera", "basalt", "tectonic"],
    "orchestra": ["violin", "conductor", "symphony", "cello", "oboe", "concerto", "rehearsal", "baton"],
    "railway": ["locomotive", "platform", "signal", "gauge", "timetable", "carriage", "junction", "viaduct"],
    "glacier": ["ice", "moraine", "crevasse", "meltwater", "firn", "icefall", "calving", "permafrost"],
    "bakery": ["dough", "yeast", "oven", "sourdough", "crust", "proofing", "flour", "croissant"],
    "telescope": ["lens", "aperture", "nebula", "refractor", "orbit", "eyepiece", "galaxy", "mirror"],
    "harbor": ["dock", "tide", "freighter", "anchor", "pier", "lighthouse", "cargo", "mooring"],
    "pottery": ["kiln", "glaze", "clay", "wheel", "ceramic", "firing", "slip", "stoneware"],
}
FILLER = "the of and a in to is was for on with as by at from that this an be are it which or its has had were".split()
FACTS = ["founded in 1887", "named after a river", "rebuilt after a fire", "famous for its annual festival", "closed in 1952", "visited by millions"]


def make_synthetic(n_docs=240, n_queries=40, seed=7):
    """Return (passages, queries, qrels) in the same shapes the real loaders produce."""
    rng = random.Random(seed)
    topics = sorted(TOPICS)
    passages = []
    relevant_by_topic_fact = {}
    for i in range(n_docs):
        topic = topics[i % len(topics)]
        fact = FACTS[rng.randrange(len(FACTS))]
        words = [rng.choice(TOPICS[topic]) for _ in range(14)] + [rng.choice(FILLER) for _ in range(22)]
        rng.shuffle(words)
        doc_id = f"d{i}"
        passages.append({"doc_id": doc_id, "title": f"{topic.title()} {i}", "text": f"{' '.join(words)} {fact} {rng.choice(TOPICS[topic])}."})
        relevant_by_topic_fact.setdefault((topic, fact), []).append(doc_id)
    queries, qrels = {}, {}
    keys = sorted(relevant_by_topic_fact)
    for q in range(n_queries):
        topic, fact = keys[q % len(keys)]
        key_words = rng.sample(TOPICS[topic], 3)
        qid = f"q{q}"
        queries[qid] = f"which {topic} was {fact} {' '.join(key_words)}"
        qrels[qid] = {doc: 1 for doc in relevant_by_topic_fact[(topic, fact)]}
    return passages, queries, qrels
