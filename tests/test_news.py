import re

import numpy as np
import pandas as pd
import pytest

from silo.agents import news as news_agent
from silo.agents.informed import FullDesk, NewsPanel
from silo.agents.llm import LLMClient, RuleBot
from silo.agents.news_extract import LLM_TABLE, mask, probe, run_extraction
from silo.data import news
from silo.data.macro import synthetic_macro
from silo.data.sources import synthetic_bars
from silo.data.vault import DataVault, normalize_bars
from silo.engine import run_backtest
from silo.firewall import PointInTimeView, PreparedTables
from silo.markets import UK
from silo.strategies import REGISTRY


@pytest.mark.parametrize("headline,cat", [
    ("3rd Quarter Results", "results"), ("Tesco PLC Interim Results 2026/27", "results"),
    ("Half-year Report", "results"), ("Half-year Financial Report", "results"), ("Total Voting Rights", "routine"), ("Holding(s) in Company", "routine"),
    ("Director/PDMR Shareholding", "director_dealing"), ("Transaction in Own Shares", "buyback"),
    ("Trading Statement", "trading_update"), ("Q1 Trading Update", "trading_update"),
    ("Directorate Change", "board"), ("Annual Financial Report", "routine"),
    ("Recommended Cash Offer for Foo plc", "mna"), ("Proposed Placing", "capital_raise"),
    ("Profit Warning", "profit_warning"),
])
def test_classify(headline, cat):
    assert news.classify(headline) == cat


@pytest.mark.parametrize("text,sign", [
    ("Full year profit ahead of market expectations.", 1),
    ("Trading remains in line with the Board's expectations.", 0),
    ("Profits are now expected to be below market expectations.", -1),
    ("Revenue ahead of expectations, but operating profit below consensus.", -1),
    ("We have upgraded guidance for the year.", 1),
    ("", 0),
])
def test_expectation_sign(text, sign):
    assert news.expectation_sign(text) == sign


@pytest.mark.parametrize("text,sign", [
    ("Mr Smith purchased 10,000 ordinary shares at 250p.", 1),
    ("The CFO sold 5,000 shares.", -1),
    ("Award of shares under the LTIP; shares sold to cover tax.", 0),
    ("Shares acquired through the dividend reinvestment plan.", 0),
    # a real AstraZeneca-style notice: the buy is in the transaction field, "sold" only in the footer
    ("Details of the person discharging managerial responsibilities Name Pascal Soriot Position CEO "
     "Nature of the transaction Purchase of Ordinary Shares Price(s) and volume(s) GBP121.02 60,000 "
     "Date of the transaction 18 September 2026 About AstraZeneca AstraZeneca is a global biopharmaceutical "
     "company. Its innovative medicines are sold in more than 125 countries and used by millions of patients.", 1),
    ("The Chair bought 2,500 shares at 121.21p. About AstraZeneca Its medicines are sold in more than 125 "
     "countries and option programmes are described at www.example.com.", 1),
    ("Nature of the transaction Vesting of shares under the Performance Share Plan Price(s) and volume(s) nil 1,000 "
     "About AstraZeneca medicines are sold worldwide.", 0),
])
def test_dealing_direction(text, sign):
    assert news.dealing_direction(text) == sign


PAGE = """<html><body><table class="x">
<tr><th>Date</th><th>Time</th><th>Source</th><th>Announcement</th></tr>
<tr><td>30 Oct 2026</td><td>07:00 AM</td><td><a href="https://www.investegate.co.uk/source/RNS">RNS</a></td>
<td><a href="https://www.investegate.co.uk/announcement/rns/astrazeneca--azn/3rd-quarter-results/9849221">3rd Quarter Results</a></td></tr>
<tr><td>15 Jul 2026</td><td>03:05 PM</td><td><a href="/source/RNS">RNS</a></td>
<td><a href="/announcement/rns/astrazeneca--azn/director-pdmr-shareholding/9700000">Director/PDMR Shareholding</a></td></tr>
<tr><td>no link here</td></tr>
</table></body></html>"""


def test_parse_list_and_london_time():
    rows = news.parse_list(PAGE, "AZN.L")
    assert list(rows["id"]) == ["9849221", "9700000"]
    assert list(rows["category"]) == ["results", "director_dealing"]
    assert rows["available_at"].iloc[0] == pd.Timestamp("2026-10-30 07:00", tz="UTC")  # GMT
    assert rows["available_at"].iloc[1] == pd.Timestamp("2026-07-15 14:05", tz="UTC")  # BST
    assert rows["url"].iloc[1].startswith(news.BASE + "/announcement/")


class FakeSite:
    def __init__(self, pages, bodies=None):
        self.pages, self.bodies, self.urls = pages, bodies or {}, []

    def __call__(self, url):
        self.urls.append(url)
        if "/announcement/" in url:
            return f"<html><script>x()</script><p>{self.bodies.get(url.rsplit('/', 1)[-1], '')}</p></html>"
        return self.pages.get(url, "<table></table>")


def _row(day, time, ident, headline, tidm="azn", slug="astrazeneca"):
    return (f"<tr><td>{day}</td><td>{time}</td><td>RNS</td><td><a href=\"/announcement/rns/{slug}--{tidm}/"
            f"{headline.lower().replace(' ', '-').replace('/', '-')}/{ident}\">{headline}</a></td></tr>")


def test_fetch_is_polite_resumable_and_snapshotted(tmp_path):
    p1 = "<table>" + _row("30 Oct 2026", "07:00 AM", "3", "Trading Statement") + \
         _row("01 Oct 2026", "07:00 AM", "2", "Director/PDMR Shareholding") + "</table>"
    p2 = "<table>" + _row("01 Jan 2011", "07:00 AM", "1", "Final Results") + "</table>"
    site = FakeSite({f"{news.BASE}/company/AZN": p1, f"{news.BASE}/company/AZN?page=2": p2},
                    {"3": "Sales ahead of market expectations.", "2": "The CEO purchased 1,000 shares."})
    slept = []
    vault = DataVault(tmp_path)
    fetcher = news.Fetcher(delay=3, opener=site, sleep=slept.append)
    assert news.fetch_news(vault, ["AZN.L"], since="2012-01-01", fetcher=fetcher, log=lambda m: None) == []
    table = vault.get_context(news.TABLE)
    assert list(table.index) == ["2", "3"]  # 2011 result skipped by --since; sorted by time
    assert table.loc["3", "expect"] == 1 and table.loc["2", "dealing"] == 1
    assert "body" not in table.columns and news.load_body(vault, "3").startswith("Sales ahead")
    assert slept and set(slept) == {3}
    # a rerun stops at the first known announcement: one list page, nothing new
    site.urls.clear()
    news.fetch_news(vault, ["AZN.L"], fetcher=news.Fetcher(0, opener=site, sleep=lambda s: None), log=lambda m: None)
    assert site.urls == [f"{news.BASE}/company/AZN"]
    assert len(list((tmp_path / "context").rglob("*.parquet"))) == 1
    with pytest.raises(ValueError):
        news.Fetcher(opener=site).get("https://live.example.com/x")


def test_mask_hides_company_dates_and_amounts():
    text = ("AstraZeneca PLC (AZN) reports Q3 2026 revenue of $15.2bn, up 8%, ahead of expectations. "
            "On 30/10/2026 the Board met in October. See www.astrazeneca.com")
    out = mask(text, "astrazeneca", "AZN")
    assert "astrazeneca" not in out.lower() and "AZN" not in out
    assert not re.search(r"\b20\d\d\b", out) and "October" not in out and "15.2" not in out
    assert "up 8%" in out and "ahead of expectations" in out


def uk_world(n=4, n_days=700):
    syms = [f"T{i}.L" for i in range(n)]
    t = {s: normalize_bars(synthetic_bars(n_days=n_days, seed=i, start_price=2500), UK, pence=True)
         for i, s in enumerate(syms)}
    dates = t[syms[0]].index
    # T0 jumps 10% on a results day; each stock gets announcements through time
    jump = 400
    t[syms[0]].loc[dates[jump:], ["open", "high", "low", "close"]] *= 1.10
    rows = []
    for k, d in enumerate(dates[150::20]):
        for i, s in enumerate(syms):
            cat = ["results", "trading_update", "director_dealing", "routine"][(k + i) % 4]
            rows.append({"id": f"{k}-{i}", "symbol": s, "category": cat, "headline": cat, "expect": 0, "dealing": 0,
                         "available_at": pd.Timestamp(d).tz_localize("Europe/London").tz_convert("UTC")
                         + pd.Timedelta(hours=7)})
    rows.append({"id": "big", "symbol": syms[0], "category": "results", "headline": "Results", "expect": 1,
                 "dealing": 0, "available_at": pd.Timestamp(dates[jump]).tz_localize("UTC") + pd.Timedelta(hours=7)})
    rows.append({"id": "buy", "symbol": syms[1], "category": "director_dealing", "headline": "Director/PDMR",
                 "expect": 0, "dealing": 1,
                 "available_at": pd.Timestamp(dates[jump]).tz_localize("UTC") + pd.Timedelta(hours=7)})
    t[news.TABLE] = news.to_table(pd.DataFrame(rows))
    return syms, t | synthetic_macro(dates, seed=2), jump


def test_news_columns_are_point_in_time():
    syms, t, jump = uk_world()
    dates = t[syms[0]].index
    before = PointInTimeView(PreparedTables(t), t[syms[0]]["available_at"].iloc[jump - 1])
    after = PointInTimeView(PreparedTables(t), t[syms[0]]["available_at"].iloc[jump + 5])
    c0 = news_agent.columns(before, syms)
    c1 = news_agent.columns(after, syms)
    assert c0["expect"][syms[0]] == 0 and c1["expect"][syms[0]] == 1
    assert c1["results_move"][syms[0]] > 5
    assert c1["days_since_results"][syms[0]] == 5
    assert c0["director_buys"][syms[1]] == 0 and c1["director_buys"][syms[1]] == 1
    assert dates[jump] in t[syms[0]].index


def test_without_news_volume_days_stand_in():
    syms, t, _ = uk_world()
    del t[news.TABLE]
    view = PointInTimeView(PreparedTables(t), t[syms[0]]["available_at"].iloc[600])
    c = news_agent.columns(view, syms)
    assert all(c["days_since_results"][s] == news_agent.NO_RESULTS for s in syms)
    assert all(np.isfinite(c["news_mom"][s]) for s in syms)


class Recorder(LLMClient):
    def __init__(self):
        super().__init__(model="recorder")
        self.prompts = []

    def _ask(self, system, prompt):
        self.prompts.append(prompt)
        return "{}"


@pytest.mark.parametrize("cls", [NewsPanel, FullDesk])
def test_news_briefs_have_numbers_not_text(cls):
    syms, t, jump = uk_world()
    client = Recorder()
    cls(syms, client=client).decide(PointInTimeView(PreparedTables(t), t[syms[0]]["available_at"].iloc[jump + 5]))
    text = "\n".join(client.prompts)
    assert "results-day move vs market %" in text and "News:" in text
    assert "Director/PDMR" not in text and "Results\n" not in text
    assert not re.search(r"\b(19|20)\d\d\b", text)
    assert not any(s[:-2] in text for s in syms)


@pytest.mark.parametrize("name", ["news_mom", "results_drift", "director_buys", "panel_news", "desk_full"])
def test_truncation_invariance_with_news(name):
    syms, t, _ = uk_world()
    cut = t[syms[0]].index[500]
    truncated = {s: df[df.index <= cut] if not s.startswith("ctx:news") else
                 df[df["available_at"] <= t[syms[0]].loc[cut, "available_at"] + pd.Timedelta(hours=12)]
                 for s, df in t.items()}
    full = run_backtest(REGISTRY[name](syms), t)
    part = run_backtest(REGISTRY[name](syms), truncated)
    before = [d for d in full.decisions if d["date"] <= cut]
    assert before == part.decisions
    pd.testing.assert_series_equal(full.equity[full.equity.index <= cut], part.equity)


def test_results_drift_buys_after_good_results():
    syms, t, jump = uk_world()
    res = run_backtest(REGISTRY["results_drift"](syms), t)
    held = {s for d in res.decisions if d["date"] > t[syms[0]].index[jump] for s in d["proposed"]}
    assert syms[0] in held


def test_extraction_and_probe_with_offline_bot(tmp_path):
    vault = DataVault(tmp_path)
    rows = pd.DataFrame([{"id": "9", "symbol": "AZN.L", "company": "astrazeneca", "category": "results",
                          "headline": "Half-year Report", "expect": 0, "dealing": 0, "url": "u",
                          "available_at": pd.Timestamp("2026-07-29 06:00", tz="UTC"), "body": "AstraZeneca did fine."}])
    news.save_bodies(vault, rows)
    vault.put_context(news.TABLE, news.to_table(rows.drop(columns="body")))
    bot = RuleBot(log_dir=tmp_path / "log")
    assert run_extraction(vault, bot, log=lambda m: None) == 1
    assert run_extraction(vault, bot, log=lambda m: None) == 0  # never read twice
    llm = vault.get_context(LLM_TABLE)
    assert llm.loc["9", "model"] == "rulebot" and llm["available_at"].iloc[0] == rows["available_at"].iloc[0]
    assert probe(vault, bot)["company_hit_rate"] == 0


def test_rescore_fixes_stored_rows_without_overwriting(tmp_path):
    vault = DataVault(tmp_path)
    rows = pd.DataFrame([{"id": "1", "symbol": "AZN.L", "company": "astrazeneca", "category": "other",
                          "headline": "Half-year Financial Report", "expect": 0, "dealing": 0, "url": f"{news.BASE}/announcement/rns/x--azn/h/1",
                          "available_at": pd.Timestamp("2026-07-27 06:00", tz="UTC"), "body": ""},
                         {"id": "2", "symbol": "AZN.L", "company": "astrazeneca", "category": "director_dealing",
                          "headline": "Director/PDMR Shareholding", "expect": 0, "dealing": 0, "url": f"{news.BASE}/a/2",
                          "available_at": pd.Timestamp("2026-09-18 06:00", tz="UTC"),
                          "body": "Nature of the transaction Purchase of shares Price(s) and volume(s) 1 2 "
                                  "About AstraZeneca medicines are sold in 125 countries."}])
    news.save_bodies(vault, rows)
    vault.put_context(news.TABLE, news.to_table(rows.drop(columns="body")))
    site = FakeSite({}, {"1": "Results ahead of expectations."})
    assert news.rescore(vault, news.Fetcher(0, opener=site, sleep=lambda s: None), log=lambda m: None) == 2
    table = vault.get_context(news.TABLE)
    assert table.loc["1", "category"] == "results" and table.loc["1", "expect"] == 1
    assert table.loc["2", "dealing"] == 1
    assert len(list((tmp_path / "context").rglob("*.parquet"))) == 2
    assert news.rescore(vault, log=lambda m: None) == 0
