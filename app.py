import json, re
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastembed import TextEmbedding

# ---------- 1. Load the model and the knowledge base ----------
model = TextEmbedding("BAAI/bge-small-en-v1.5")

def embed(texts):
    v = np.array(list(model.embed(texts)), dtype=np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)   # unit length, so dot product = cosine

raw = json.load(open("kb.json"))
passages = [(p["text"] if isinstance(p, dict) else p) for p in raw]
passages = [re.sub(r"^Passage\s*\d+\s*:\s*", "", p).strip() for p in passages]

# Split each passage into sentences, remembering which passage each came from
sentences, sent_owner = [], []
for i, p in enumerate(passages):
    for s in re.split(r"(?<=[.!?])\s+", p):
        if s.strip():
            sentences.append(s.strip())
            sent_owner.append(i)
sent_owner = np.array(sent_owner)

P = embed(passages)      # 50 passage vectors
S = embed(sentences)     # ~250 sentence vectors

# Every word that appears in the KB (used to spot unknown topics like "Kafka")
vocab = set(re.findall(r"[a-z0-9\-\.]+", " ".join(passages).lower()))

# ---------- 2. Tools ----------
INJECTION = re.compile(
    r"(ignore|disregard|forget).{0,40}(instruction|rule|prompt|above|previous)"
    r"|system prompt|you are now|reveal|override|new instructions|instead (say|answer|respond)"
    r"|answer with|respond with|print the",
    re.I,
)
STOP = {"what","which","how","does","the","is","are","was","when","where","who","why",
        "a","an","of","in","on","for","to","and","or","do","default","value","number"}

def clean_question(q):
    # Drop any sentence that looks like an injection attempt, keep the real question
    parts = re.split(r"(?<=[.!?])\s+|\n+", q)
    kept = [x for x in parts if not INJECTION.search(x)]
    return " ".join(kept).strip() or q

def unknown_entity(q):
    # A capitalised or technical word (not the first word) that the KB has never seen
    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9\-\.]*", q)
    for w in words[1:]:
        base = re.sub(r"('s|s')$", "", w).lower().strip(".")
        if (w[0].isupper() or any(c.isdigit() for c in w)) and base not in vocab and base not in STOP:
            return True
    return False

def abstain(top=0.0):
    return {"answer": None, "sources": [], "scores": [], "confidence": round(min(0.3, float(top)), 4)}

# ---------- 3. The web app ----------
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class Query(BaseModel):
    question: str = ""
    top_k: int = 3

PASSAGE_MIN = 0.55     # TUNE THESE (Step 6)
SENTENCE_MIN = 0.55

@app.get("/")
def health():
    return {"ok": True}

@app.post("/query")
def query(req: Query):               # plain "def" => FastAPI runs requests in parallel threads
    q = clean_question(req.question or "")
    if len(q) < 3:
        return abstain()
    k = max(1, min(int(req.top_k or 3), 10))

    qv = embed([q])[0]
    p_scores = P @ qv
    order = np.argsort(-p_scores)[:k]
    top = float(p_scores[order[0]])

    # Best sentence inside each top passage
    s_scores = S @ qv
    best = {}
    for pi in order:
        idx = np.where(sent_owner == pi)[0]
        j = idx[np.argmax(s_scores[idx])]
        best[int(pi)] = (float(s_scores[j]), sentences[j])

    best_sent = max(v[0] for v in best.values())
    if top < PASSAGE_MIN or best_sent < SENTENCE_MIN or unknown_entity(q):
        return abstain(top)

    # Answer = best sentence from the #1 passage
    top_pid = int(order[0])
    answer = best[top_pid][1]

    # Cross-passage questions ("compare X and Y"): also add best sentence from the runner-up
    if re.search(r"\b(compare|both|differ|difference|versus|vs|and)\b", q, re.I) and len(order) > 1:
        second = int(order[1])
        if p_scores[second] >= top - 0.08:
            answer += " " + best[second][1]

    scores = [round(float(np.clip(p_scores[i], 0, 1)), 4) for i in order]
    confidence = round(float(min(0.99, max(0.65, top + 0.15))), 4)
    return {"answer": answer, "sources": [int(i) for i in order],
            "scores": scores, "confidence": confidence}