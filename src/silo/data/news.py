"""UK company announcements (RNS) as a point-in-time context table.

Source: Investegate's per-company announcement lists (``/company/<TIDM>``,
newest first, ``?page=N``). Silo is a personal research tool and nothing
fetched is redistributed. Collection is deliberately gentle:

* one request at a time with a pause between requests (3 s by default);
* a list page is fetched only until it reaches announcements already stored,
  and an announcement body is fetched at most once and kept locally;
* only the categories the news agent uses get their bodies fetched.

Every announcement is stored in ``ctx:news`` with its publication time
converted from London time to UTC as ``available_at``. A 07:00 RNS is
therefore visible at that day's close; one released at 16:40 is first seen
the next trading day.

Categories come from the headline by plain rules (:func:`classify`), and the
"versus market expectations" sign from the wording UK trading statements
use (:func:`expectation_sign`), so most features never touch a model.
"""

from __future__ import annotations

import html
import re
import time
import urllib.request
from html.parser import HTMLParser

import pandas as pd

BASE = "https://www.investegate.co.uk"
USER_AGENT = "Mozilla/5.0 (personal research script; low rate)"
TABLE = "ctx:news"
LONDON = "Europe/London"

_HREF = re.compile(r"/announcement/(?P<source>[a-z0-9]+)/(?P<company>[a-z0-9-]+?)--(?P<tidm>[a-z0-9.]+)/"
                   r"(?P<slug>[^/]*)/(?P<id>\d+)/?$")
_DATE = re.compile(r"\b(\d{2} [A-Z][a-z]{2} \d{4})\b")
_TIME = re.compile(r"\b(\d{1,2}:\d{2}\s*[AP]M)\b", re.I)

# Order matters: the first rule that matches wins.
CATEGORY_RULES = [
    ("routine", r"total voting rights|holding\(s\) in company|holdings? in company|form 8|block listing|"
                r"admission of|notice of (agm|general meeting)|result of (agm|general meeting)|annual financial report|"
                r"publication of|net asset value|director declaration|change of (name|registered|adviser)|"
                r"statement re (press|media)|doc re|replacement"),
    ("director_dealing", r"director.{0,3}pdmr|pdmr|director(s)?'? (share|dealing|purchase)|director/pdmr"),
    ("buyback", r"transaction in own shares|purchase of own shares|share (buy ?back|repurchase)"),
    ("profit_warning", r"profit warning|below (market )?expectations"),
    ("trading_update", r"trading (update|statement)|pre-close|interim management statement|business update|"
                       r"q[1-4] trading|first quarter trading|third quarter trading"),
    ("results", r"results|interim report|half.year report|half.yearly|annual report and accounts|"
                r"quarter(ly)? (report|update)|q[1-4] (\d{4} )?(results|report)"),
    ("mna", r"offer for|possible offer|rule 2\.[47]|recommended (cash )?(offer|acquisition)|acquisition|"
            r"disposal|merger|scheme of arrangement"),
    ("capital_raise", r"placing|rights issue|fundrais|open offer|issue of equity|bookbuild|convertible bond"),
    ("board", r"directorate change|board change|appointment|chief executive|ceo|chair(man)? |resignation|"
              r"succession"),
    ("dividend", r"dividend"),
]
_RULES = [(cat, re.compile(rx, re.I)) for cat, rx in CATEGORY_RULES]

# Categories whose announcement text is worth fetching.
BODY_CATEGORIES = {"results", "trading_update", "profit_warning", "director_dealing"}

_AHEAD = re.compile(r"(ahead of|above|exceed(s|ed|ing)?|better than|upper end of|top end of)\s+(the\s+)?"
                    r"(board'?s?\s+|our\s+|market\s+|consensus\s+|analysts'?\s+|previous\s+)*"
                    r"(expectations|guidance|consensus|forecasts?)|upgrade[sd]? (to )?(guidance|expectations)", re.I)
_BELOW = re.compile(r"(below|lower than|short of|behind|lower end of|bottom end of|beneath)\s+(the\s+)?"
                    r"(board'?s?\s+|our\s+|market\s+|consensus\s+|analysts'?\s+|previous\s+)*"
                    r"(expectations|guidance|consensus|forecasts?)|profit warning|(lower|cut|reduce[sd]?|downgrade[sd]?) "
                    r"(its |our |full.year |profit )*(guidance|expectations|outlook)", re.I)
_INLINE = re.compile(r"in line with\s+(the\s+)?(board'?s?\s+|our\s+|market\s+|consensus\s+|analysts'?\s+|"
                     r"previous\s+)*(expectations|guidance|consensus|forecasts?)|on track|reiterate[sd]? "
                     r"(its |our |full.year )*(guidance|outlook)", re.I)

_BUY = re.compile(r"\b(purchase[sd]?|bought|acquired|acquisition of (ordinary )?shares)\b", re.I)
_SELL = re.compile(r"\b(sale|sold|disposal|disposed|sell)\b", re.I)
_NOT_A_DEAL = re.compile(r"\b(award(ed)?|vest(ing|ed)?|exercise[sd]?|option|share incentive plan|sip|ltip|"
                         r"dividend reinvestment|drip|grant(ed)?|deferred bonus|sharesave|saye|"
                         r"partnership shares|matching shares|transfer)\b", re.I)


def classify(headline: str) -> str:
    for cat, rx in _RULES:
        if rx.search(headline):
            return cat
    return "other"


def expectation_sign(text: str) -> int:
    """+1 ahead of expectations, -1 below (or a warning), 0 in line or not said.

    When several phrases appear (common: "revenue ahead, profit in line"),
    the most cautious wins: a below beats an ahead.
    """
    if not text:
        return 0
    if _BELOW.search(text):
        return -1
    if _AHEAD.search(text):
        return 1
    return 0


def dealing_direction(text: str) -> int:
    """+1 a director bought shares with their own money, -1 sold, 0 anything else.

    Awards, option exercises, vesting, dividend reinvestment and share plans
    carry little information and count as 0.
    """
    if not text or _NOT_A_DEAL.search(text):
        return 0
    buy, sell = bool(_BUY.search(text)), bool(_SELL.search(text))
    if buy and not sell:
        return 1
    if sell and not buy:
        return -1
    return 0


class _Rows(HTMLParser):
    """Collect table rows as (text, [hrefs]); tolerant of any class names."""

    def __init__(self):
        super().__init__()
        self.rows, self._text, self._hrefs, self._links, self._in_row = [], [], [], [], False
        self._anchor = None

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self._in_row, self._text, self._hrefs = True, [], []
        elif tag == "a" and self._in_row:
            href = dict(attrs).get("href") or ""
            self._anchor = [href, []]

    def handle_endtag(self, tag):
        if tag == "a" and self._anchor is not None:
            self._hrefs.append((self._anchor[0], " ".join(self._anchor[1]).strip()))
            self._anchor = None
        elif tag == "tr" and self._in_row:
            self.rows.append((" ".join(self._text), self._hrefs))
            self._in_row = False

    def handle_data(self, data):
        if self._in_row:
            self._text.append(data.strip())
            if self._anchor is not None:
                self._anchor[1].append(data.strip())


def parse_list(page: str, symbol: str) -> pd.DataFrame:
    """Announcements on one Investegate company page."""
    parser = _Rows()
    parser.feed(page)
    out = []
    for text, hrefs in parser.rows:
        link = next(((h, t) for h, t in hrefs if _HREF.search(h)), None)
        d, t = _DATE.search(text), _TIME.search(text)
        if not link or not d or not t:
            continue
        m = _HREF.search(link[0])
        local = pd.to_datetime(f"{d.group(1)} {t.group(1).upper().replace('  ', ' ')}", format="%d %b %Y %I:%M %p")
        published = local.tz_localize(LONDON, ambiguous=True, nonexistent="shift_forward").tz_convert("UTC")
        headline = html.unescape(link[1]) or m["slug"].replace("-", " ")
        url = link[0] if link[0].startswith("http") else BASE + link[0]
        out.append({"id": m["id"], "symbol": symbol, "source": m["source"].upper(), "company": m["company"],
                    "headline": headline, "category": classify(headline), "url": url, "available_at": published})
    return pd.DataFrame(out)


def body_text(page: str) -> str:
    """Plain text of an announcement page (scripts and styles dropped)."""
    page = re.sub(r"(?is)<(script|style|noscript)[^>]*>.*?</\1>", " ", page)
    page = re.sub(r"(?s)<[^>]+>", " ", page)
    return re.sub(r"\s+", " ", html.unescape(page)).strip()


class Fetcher:
    """One request at a time, with a pause; nothing is fetched twice in a run."""

    def __init__(self, delay: float = 3.0, opener=None, sleep=time.sleep):
        self.delay, self.sleep, self.requests = delay, sleep, 0
        self._open = opener or self._urlopen

    @staticmethod
    def _urlopen(url: str) -> str:
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=60) as r:  # noqa: S310 - fixed https host
            return r.read().decode("utf-8", "replace")

    def get(self, url: str) -> str:
        if not url.startswith(BASE + "/"):
            raise ValueError(f"news fetcher only reads {BASE}")
        if self.requests:
            self.sleep(self.delay)
        self.requests += 1
        return self._open(url)


def tidm(symbol: str) -> str:
    return symbol.upper().removesuffix(".L")


def fetch_company(symbol: str, fetcher: Fetcher, known: set[str], since: pd.Timestamp,
                  max_pages: int = 200) -> pd.DataFrame:
    """New announcements for one company, newest first, stopping at known ids or ``since``."""
    frames = []
    for page in range(1, max_pages + 1):
        url = f"{BASE}/company/{tidm(symbol)}" + (f"?page={page}" if page > 1 else "")
        rows = parse_list(fetcher.get(url), symbol)
        if rows.empty:
            break
        fresh = rows[~rows["id"].isin(known) & (rows["available_at"] >= since)]
        frames.append(fresh)
        if len(fresh) < len(rows):  # reached stored announcements or the start date
            break
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def add_bodies(rows: pd.DataFrame, fetcher: Fetcher) -> pd.DataFrame:
    """Fetch text for the categories that need it and derive expectation / dealing signs."""
    rows = rows.copy()
    rows["body"] = ""
    for i, r in rows.iterrows():
        if r["category"] in BODY_CATEGORIES:
            try:
                rows.at[i, "body"] = body_text(fetcher.get(r["url"]))[:20000]
            except Exception:  # noqa: BLE001 - a missing body only loses that signal
                pass
    return derive(rows)


def derive(rows: pd.DataFrame) -> pd.DataFrame:
    rows = rows.copy()
    text = rows["headline"].fillna("") + ". " + rows.get("body", pd.Series("", index=rows.index)).fillna("")
    is_update = rows["category"].isin(["results", "trading_update", "profit_warning"])
    rows["expect"] = [expectation_sign(t) if u else 0 for t, u in zip(text, is_update)]
    rows.loc[rows["category"] == "profit_warning", "expect"] = -1
    rows["dealing"] = [dealing_direction(b) if c == "director_dealing" else 0
                       for b, c in zip(rows.get("body", pd.Series("", index=rows.index)).fillna(""),
                                       rows["category"])]
    return rows


def _body_path(vault, ann_id: str):
    return vault.root / "news_bodies" / f"{ann_id}.txt"


def save_bodies(vault, rows: pd.DataFrame) -> None:
    """Keep announcement text locally (written once, never overwritten), outside the context table."""
    for ann_id, body in zip(rows["id"], rows["body"]):
        path = _body_path(vault, ann_id)
        if body and not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(body, encoding="utf-8")


def load_body(vault, ann_id: str) -> str:
    path = _body_path(vault, ann_id)
    return path.read_text(encoding="utf-8") if path.exists() else ""


def to_table(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return rows
    df = rows.drop_duplicates("id", keep="last").set_index("id")
    df["available_at"] = pd.to_datetime(df["available_at"], utc=True)
    return df.sort_values("available_at", kind="stable")


def fetch_news(vault, symbols: list[str], since: str = "2012-01-01", delay: float = 3.0, log=print,
               fetcher: Fetcher | None = None) -> list[str]:
    """Add new announcements for ``symbols`` to the vault as one new snapshot. Returns failures."""
    fetcher = fetcher or Fetcher(delay)
    try:
        known = set(vault.get_context(TABLE).index)
    except KeyError:
        known = set()
    start = pd.Timestamp(since, tz="UTC")
    failed = []
    for s in symbols:
        try:
            rows = fetch_company(s, fetcher, known, start)
            if not rows.empty:  # one snapshot per company, so a long backfill can stop and resume
                rows = add_bodies(rows, fetcher)
                save_bodies(vault, rows)
                vault.put_context(TABLE, to_table(rows.drop(columns="body")))
            log(f"{s}: {len(rows)} new announcement(s)")
        except Exception as exc:  # noqa: BLE001 - keep going; report at the end
            failed.append(s)
            log(f"news fetch failed for {s}: {exc}")
    return failed
