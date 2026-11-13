"""Optional: a local model reads masked announcement text and returns two fields.

The plain-rule reading of "ahead of / in line with / below expectations"
(:func:`silo.data.news.expectation_sign`) handles most UK trading updates.
For results and updates where the rules find nothing, a local model can read
the text and answer two questions: how did the company say it did against
expectations, and what is the tone?

Leakage controls:
* the text is **masked first**: company name, ticker, years, months, dates,
  money amounts, links and emails are replaced with placeholders, so the
  model can't easily tell which company or year it is reading about;
* the answers are stored as numbers (``ctx:news_llm``) stamped with the
  announcement's own ``available_at``. The deciding model never sees text;
* :func:`probe` asks the model to guess the company and year from masked
  text. If it guesses well above chance, the masking isn't enough and the
  answers for pre-cutoff announcements shouldn't be trusted;
* every call goes through the usual logged, cached client with its training
  cutoff, so results before the cutoff can be reported separately.
"""

from __future__ import annotations

import re

import pandas as pd

from silo.agents.llm import LLMClient, parse_json
from silo.data.news import TABLE, load_body

LLM_TABLE = "ctx:news_llm"
STOPWORDS = {"plc", "group", "holdings", "the", "and", "company", "limited", "ltd", "international", "uk"}
MONTHS = ("january|february|march|april|may|june|july|august|september|october|november|december|"
          "jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec")

SYSTEM = ("You read a UK company announcement with names, dates and amounts hidden. Answer only from the "
          "text. Reply with JSON only.")


def mask(text: str, company_slug: str = "", ticker: str = "") -> str:
    out = text
    for word in sorted({w for w in company_slug.split("-") if len(w) > 2 and w not in STOPWORDS},
                       key=len, reverse=True):
        out = re.sub(rf"\b{re.escape(word)}\w*", "[company]", out, flags=re.I)
    if ticker:
        out = re.sub(rf"\b{re.escape(ticker)}\b", "[ticker]", out)
    out = re.sub(r"https?://\S+|www\.\S+", "[link]", out)
    out = re.sub(r"\S+@\S+\.\S+", "[email]", out)
    out = re.sub(r"\b\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}\b", "[date]", out)
    out = re.sub(r"[£$€]\s?\d[\d,.]*\s?(bn|billion|m|million|k|p)?\b", "[amount]", out, flags=re.I)
    out = re.sub(r"\b\d[\d,.]*\s?(bn|billion|million|pence|p)\b", "[amount]", out, flags=re.I)
    out = re.sub(r"\b(FY|H1|H2|Q[1-4])\s?'?\d{2,4}\b", r"\1 [year]", out)
    out = re.sub(r"\b(19|20)\d{2}(/\d{2,4})?\b", "[year]", out)
    out = re.sub(rf"\b({MONTHS})\b", "[month]", out, flags=re.I)
    return out


def extract(client: LLMClient, masked: str, clock=None) -> dict:
    reply = parse_json(client.complete(
        SYSTEM,
        f"{masked[:6000]}\n\nAgainst market or its own expectations, did the company say it is doing better, "
        "in line or worse? And what is the overall tone? Reply as {\"vs_expectations\": \"above\" | \"in line\" | "
        "\"below\" | \"not said\", \"tone\": <integer from -2 (very negative) to 2 (very positive)>}",
        role="news_extract", clock=clock))
    sign = {"above": 1, "in line": 0, "below": -1}.get(str(reply.get("vs_expectations", "")).lower().strip(), 0)
    try:
        tone = int(max(-2, min(2, int(reply.get("tone", 0)))))
    except (TypeError, ValueError):
        tone = 0
    return {"expect_llm": sign, "tone": tone}


def candidates(vault) -> pd.DataFrame:
    """Results and updates where the rules found no expectations wording and no model answer yet."""
    news = vault.get_context(TABLE)
    want = news[news["category"].isin(["results", "trading_update"]) & (news["expect"] == 0)]
    try:
        done = set(vault.get_context(LLM_TABLE).index)
    except KeyError:
        done = set()
    return want[~want.index.isin(done)]


def run_extraction(vault, client: LLMClient, limit: int | None = None, log=print) -> int:
    rows = candidates(vault)
    out = []
    for ann_id, r in rows.iterrows():
        if limit is not None and len(out) >= limit:
            break
        body = load_body(vault, ann_id)
        if not body:
            continue
        masked = mask(f"{r['headline']}. {body}", r.get("company", ""), str(r["symbol"]).removesuffix(".L"))
        out.append({"id": ann_id, **extract(client, masked, r["available_at"]), "model": client.model,
                    "cutoff": client.cutoff, "available_at": r["available_at"]})
    if out:
        df = pd.DataFrame(out).set_index("id").sort_values("available_at", kind="stable")
        vault.put_context(LLM_TABLE, df)
    log(f"model read {len(out)} announcement(s)")
    return len(out)


def probe(vault, client: LLMClient, n: int = 50) -> dict:
    """Can the model tell the company or year from masked text? Returns hit rates."""
    news = vault.get_context(TABLE)
    sample = news[news["category"].isin(["results", "trading_update"])]
    sample = sample.sample(min(n, len(sample)), random_state=7) if len(sample) else sample
    hits_co = hits_yr = asked = 0
    for ann_id, r in sample.iterrows():
        body = load_body(vault, ann_id)
        if not body:
            continue
        masked = mask(f"{r['headline']}. {body}", r.get("company", ""), str(r["symbol"]).removesuffix(".L"))
        reply = parse_json(client.complete(
            SYSTEM, f"{masked[:4000]}\n\nGuess which company this is and the year it was published. Reply as "
            "{\"company_guess\": \"<name or unknown>\", \"year_guess\": \"<year or unknown>\"}", role="news_probe"))
        asked += 1
        guess = str(reply.get("company_guess", "")).lower()
        words = [w for w in str(r.get("company", "")).split("-") if len(w) > 2 and w not in STOPWORDS]
        hits_co += int(bool(words) and any(w in guess for w in words))
        hits_yr += int(str(pd.Timestamp(r["available_at"]).year) in str(reply.get("year_guess", "")))
    return {"asked": asked, "company_hit_rate": hits_co / asked if asked else None,
            "year_hit_rate": hits_yr / asked if asked else None}
