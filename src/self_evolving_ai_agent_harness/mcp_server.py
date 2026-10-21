"""MCP server exposing the agent's real-world tools over stdio.

Free, no-key APIs from github.com/public-apis/public-apis:
    financial_api -> SEC EDGAR XBRL API   (official annual 10-K figures of US-listed companies)
    web_search    -> Wikipedia / MediaWiki (general knowledge)

Run standalone:  python -m self_evolving_ai_agent_harness.mcp_server
Any MCP client (CrewAI, Claude Desktop, ...) can use it. Failures are returned as
"ERROR[<kind>]: <message>" so the harness critic can classify them.
"""

import json
import os
import re
import urllib.parse
import urllib.request
import warnings
from datetime import date
from functools import lru_cache

warnings.filterwarnings("ignore")  # keep the harness console clean (pydantic-settings noise)

from mcp.server.fastmcp import FastMCP  # noqa: E402

mcp = FastMCP("self-evolving-harness-tools", log_level="WARNING")

# SEC asks every client to identify itself with a contact address.
USER_AGENT = os.getenv("SEC_USER_AGENT", "SelfEvolvingAgentHarness demo admin@example.com")

# metric -> (XBRL tags to try, unit)
SEC_METRICS = {
    "revenue": (["RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues", "SalesRevenueNet"], "USD"),
    "net income": (["NetIncomeLoss"], "USD"),
    "eps": (["EarningsPerShareDiluted", "EarningsPerShareBasic"], "USD/shares"),
    "total assets": (["Assets"], "USD"),
}


def _get_json(url: str) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=20) as response:
        return json.load(response)


@lru_cache(maxsize=1)
def _sec_companies() -> list[dict]:
    return list(_get_json("https://www.sec.gov/files/company_tickers.json").values())


@lru_cache(maxsize=64)
def _company_facts(cik: int) -> dict:
    """All us-gaap facts of one company in a single request."""
    return _get_json(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json")["facts"].get("us-gaap", {})


LEGAL_WORDS = {"the", "inc", "incorporated", "corp", "corporation", "co", "company", "ltd", "plc", "llc"}


def _key(text: str) -> str:
    """'The Coca-Cola Company' and 'COCA COLA CO' -> 'cocacola' (legal words, punctuation, case ignored)."""
    words = re.findall(r"[a-z0-9]+", text.lower().replace("-", " ").replace("&", " "))
    return "".join(w for w in words if w not in LEGAL_WORDS)


def _find_company(company: str) -> dict | None:
    """Exact ticker first, then registrant-name prefix (punctuation and case ignored)."""
    name, companies = _key(company), _sec_companies()
    if not name:
        return None
    return next((c for c in companies if _key(c["ticker"]) == name), None) or next(
        (c for c in companies if _key(c["title"]).startswith(name)), None
    )


def _latest_annual(entries: list[dict]) -> dict | None:
    """Most recent full-year value reported in a 10-K."""

    def is_full_year(e: dict) -> bool:
        if "start" not in e:  # balance-sheet items (e.g. assets) are point-in-time
            return True
        return 350 <= (date.fromisoformat(e["end"]) - date.fromisoformat(e["start"])).days <= 380

    annual = [e for e in entries if e.get("form") == "10-K" and is_full_year(e)]
    return max(annual, key=lambda e: (e["end"], e["filed"]), default=None)


@mcp.tool()
def financial_api(company: str, metric: str) -> str:
    """Official financial figures of a US-listed company from SEC 10-K filings.
    company: the SEC registrant name or stock ticker, e.g. 'Apple' or 'MSFT'.
    metric: one of 'revenue', 'net income', 'eps', 'total assets'."""
    metric = metric.lower().strip()
    if metric not in SEC_METRICS:
        return f"ERROR[unsupported_metric]: {metric!r}. Use one of {list(SEC_METRICS)}."
    try:
        found = _find_company(company)
        if not found:
            return f"ERROR[company_not_found]: {company!r} is not an SEC registrant name or ticker."
        tags, unit = SEC_METRICS[metric]
        facts = _company_facts(found["cik_str"])
        # Companies switch XBRL tags over the years, so take the newest figure across all candidate tags.
        latest_per_tag = [_latest_annual(facts.get(tag, {}).get("units", {}).get(unit, [])) for tag in tags]
        latest = max(filter(None, latest_per_tag), key=lambda e: (e["end"], e["filed"]), default=None)
        if not latest:
            return f"ERROR[no_data]: no annual {metric} found for {found['title']}."
        value = latest["val"]
        shown = f"${value:,.2f}" if unit == "USD/shares" else f"${value / 1e9:,.1f}B"
        return (
            f"{found['title']} {metric} for fiscal year ending {latest['end']}: {shown} "
            f"(SEC 10-K, filed {latest['filed']})"
        )
    except Exception as e:  # network / parsing problems become an observation, not a crash
        return f"ERROR[tool_error]: {e}"


@mcp.tool()
def web_search(query: str) -> str:
    """Search Wikipedia for general, descriptive or historical information."""
    try:
        params = urllib.parse.urlencode(
            {"action": "query", "list": "search", "srsearch": query, "srlimit": 1, "format": "json"}
        )
        hits = _get_json(f"https://en.wikipedia.org/w/api.php?{params}")["query"]["search"]
        if not hits:
            return f"ERROR[no_data]: no Wikipedia results for {query!r}."
        title = urllib.parse.quote(hits[0]["title"].replace(" ", "_"))
        summary = _get_json(f"https://en.wikipedia.org/api/rest_v1/page/summary/{title}")
        return f"Wikipedia - {summary['title']}: {summary.get('extract', '')[:600]}"
    except Exception as e:
        return f"ERROR[tool_error]: {e}"


if __name__ == "__main__":
    mcp.run()  # stdio transport
