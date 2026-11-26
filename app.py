import json, re, math
from collections import Counter
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastembed import TextEmbedding

# ---------- model + helpers ----------
model = TextEmbedding("BAAI/bge-small-en-v1.5")

def embed(texts):
    v = np.array(list(model.embed(texts)), dtype=np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)

STOP = set("""what which who whom whose when where why how does do did is are was were be been the a an
of in on for to and or by with at from as that this these those it its default value values used use uses
using many much number tell me please can could would should give state say name describe explain
according docs documentation""".split())

def stem(w):
    for suf in ("ing", "ed", "es", "s"):
        if len(w) > 4 and w.endswith(suf):
            return w[:-len(suf)]
    return w

def toks(text):
    ws = re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", text.lower().replace("-", " "))
    return [stem(w) for w in ws if w not in STOP]

# ---------- knowledge base ----------
raw = json.load(open("kb.json"))
passages = [(p["text"] if isinstance(p, dict) else p) for p in raw]
passages = [re.sub(r"^Passage\s*\d+\s*:\s*", "", p).strip() for p in passages]
N = len(passages)

sentences, sent_owner = [], []
for i, p in enumerate(passages):
    for s in re.split(r"(?<=[.!?])\s+", p):
        if s.strip():
            sentences.append(s.strip())
            sent_owner.append(i)
sent_owner = np.array(sent_owner)

P = embed(passages)
S = embed(sentences)
ptoks = [set(toks(p)) for p in passages]
stoks = [set(toks(s)) for s in sentences]

df = Counter(t for s in ptoks for t in s)
idf = {t: math.log((N + 1) / (c + 0.5)) for t, c in df.items()}
UNK = max(idf.values()) * 0.8          # words the KB has never seen count as "rare and missing"
ENT = {t for t, c in df.items() if c <= 2}   # distinctive words (raft, redis, swim...)

def wt(t):
    return idf.get(t, UNK)

def lex(qt, tokset):
    tot = sum(wt(t) for t in qt)
    return sum(wt(t) for t in qt if t in tokset) / tot if tot else 0.0

# ---------- injection cleaning ----------
BRACKETED = re.compile(
    r"[\[\(\{<][^\]\)\}>]*?(ignore|instruction|system|assistant|override|respond|reply|prompt)[^\]\)\}>]*[\]\)\}>]", re.I)
INJ = re.compile(
    r"\b(ignore|disregard|forget|override|bypass)\b.{0,60}\b(instruction|rule|prompt|previous|above|prior|guideline|polic)"
    r"|system\s*(prompt|message|override|note)|developer\s*(mode|message)|\byou are now\b"
    r"|\bnew instructions?\b|\bpretend\b|\bjailbreak\b"
    r"|\b(instead|rather)\b.{0,30}\b(say|answer|respond|reply|output|return)"
    r"|\b(say|respond|reply|answer|output|print|return|write)\s+(only\s+|exactly\s+|just\s+)?[\"'\u201c]"
    r"|\b(reveal|leak|repeat)\b.{0,30}\b(prompt|instruction|secret|key)"
    r"|\bset\s+(the\s+)?(confidence|score|sources|answer)\b|\bconfidence\s*(to|=|of)\s*[0-9.]+"
    r"|<\|.*?\|>|\bassistant\s*:|\bnote to (the )?(ai|assistant|model)\b|\bIMPORTANT\s*:", re.I)

def clean_question(q):
    q = BRACKETED.sub(" ", q)
    kept = []
    for s in re.split(r"(?<=[.!?])\s+|\n+", q):
        m = INJ.search(s)
        if m:
            s = s[:m.start()].rstrip(" ,;:-\u2013\u2014")
            if len(toks(s)) < 2:
                continue
        if s.strip():
            kept.append(s.strip())
    asked = [s for s in kept if s.endswith("?")]      # prefer the real question
    return " ".join(asked or kept)

# ---------- retrieval ----------
D_MIN, COV_MIN, FRAG_MIN = 0.55, 0.45, 0.50    # TUNE THESE

def retrieve(texts):
    qv = embed(texts)
    D = qv @ P.T
    out = []
    for i, t in enumerate(texts):
        qt = toks(t)
        L = np.array([lex(qt, ptoks[j]) for j in range(N)])
        H = 0.7 * D[i] + 0.3 * L
        out.append((qv[i], D[i], L, H, qt))
    return out

SPLIT = re.compile(r"\b(?:and|or|versus|vs\.?|compared (?:to|with)|compare|than|both|between|while|whereas)\b|[,;]", re.I)

def synth_passages(q):
    frs = [f.strip() for f in SPLIT.split(q) if f and f.strip() and toks(f)]
    if len(frs) < 2:
        return []
    found = {}
    for f, (qv, D, L, H, qt) in zip(frs, retrieve(frs)):
        pid = int(np.argmax(H))
        if D[pid] >= FRAG_MIN and any(t in ENT and t in ptoks[pid] for t in qt):
            found[pid] = max(found.get(pid, 0), float(H[pid]))
    if len(found) < 2:
        return []
    return [p for p, _ in sorted(found.items(), key=lambda x: -x[1])]

def best_sentences(pid, qv, qt, n):
    idx = np.where(sent_owner == pid)[0]
    sc = np.array([0.7 * float(S[j] @ qv) + 0.3 * lex(qt, stoks[j]) for j in idx])
    top = float(sc.max())
    ranked = np.argsort(-sc)
    chosen = [ranked[0]] + [r for r in ranked[1:n] if sc[r] >= top - 0.10]
    return " ".join(sentences[idx[r]] for r in sorted(chosen))

def abstain(top=0.0):
    return {"answer": None, "sources": [], "scores": [],
            "confidence": round(float(min(0.3, max(0.0, top))), 4)}

# ---------- web app ----------
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class Query(BaseModel):
    question: str = ""
    top_k: int = 3

@app.get("/")
def health():
    return {"ok": True}

@app.get("/debug")
def debug(q: str):
    c = clean_question(q)
    if not toks(c):
        return {"cleaned": c, "note": "nothing left -> abstain"}
    qv, D, L, H, qt = retrieve([c])[0]
    top5 = [{"pid": int(i), "dense": round(float(D[i]), 3), "lex": round(float(L[i]), 3),
             "hybrid": round(float(H[i]), 3)} for i in np.argsort(-H)[:5]]
    return {"cleaned": c, "top5": top5, "synth": synth_passages(c)}

@app.post("/query")
def query(req: Query):
    q = clean_question(req.question or "")
    if not toks(q):
        return abstain()
    k = max(1, min(int(req.top_k or 3), 10))

    qv, D, L, H, qt = retrieve([q])[0]
    order = [int(i) for i in np.argsort(-H)]
    top = order[0]
    multi = synth_passages(q)

    if multi:
        if D[top] < FRAG_MIN:
            return abstain(H[top])
    elif D[top] < D_MIN or L[top] < COV_MIN:
        return abstain(H[top])

    # pick the sources (synthesis passages first, then fill up to top_k)
    chosen = list(multi)
    for p in order:
        if len(chosen) >= max(k, len(multi)):
            break
        if p not in chosen:
            chosen.append(p)
    chosen.sort(key=lambda p: -H[p])

    # build the answer
    if multi:
        answer = " ".join(best_sentences(p, qv, qt, 2) for p in sorted(multi, key=lambda p: -H[p]))
    else:
        n = 3 if re.search(r"\band\b", q, re.I) else 2
        answer = best_sentences(top, qv, qt, n)

    scores = [round(float(np.clip(H[p], 0, 1)), 4) for p in chosen]
    confidence = round(float(min(0.99, max(0.65, H[top] + 0.15))), 4)
    return {"answer": answer, "sources": chosen, "scores": scores, "confidence": confidence}
