"""
Multi-agent fundamental analysis of a multinational company.
Runs on a LOCAL Ollama model by default (no API key, no cost). OpenAI is optional.

Agents
------
Financial_data_agent  -> ratios & trends from Yahoo Finance + transcript commentary
news_sentiment_agent  -> recent news sentiment (Yahoo Finance news) + transcript tone
Risk_agent            -> market / financial / geopolitical / FX risks
Valuation_agent       -> multiples + simple 3-scenario DCF
ReportWriterAgent     -> merges everything into a report + evaluation remark

Setup
-----
    1. Install Ollama and pull a model:   ollama pull llama3.1
    2. pip install -r requirements.txt
    3. copy .env.example to .env and set OLLAMA_MODEL to a name shown by `ollama list`

Usage
-----
    python fundamental_analysis_agents.py AAPL --transcript transcripts/aapl_q3.txt
    python fundamental_analysis_agents.py NESN.SW --transcript nestle.txt --out reports
    python fundamental_analysis_agents.py MSFT          # no transcript: Yahoo data only

yfinance reads Yahoo Finance data and needs no API key.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yfinance as yf

try:  # optional: load settings from a .env file next to this script
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).with_name(".env"))
except ImportError:
    pass

# --------------------------------------------------------------------------- #
# Settings (all overridable in .env)
# --------------------------------------------------------------------------- #
PROVIDER = os.getenv("LLM_PROVIDER", "ollama").lower()  # "ollama" or "openai"
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")
OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "8192"))  # context window in tokens
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# ~4 characters per token; leave room for data + the model's answer inside num_ctx
_default_chars = "12000" if PROVIDER == "ollama" else "60000"
MAX_TRANSCRIPT_CHARS = int(os.getenv("MAX_TRANSCRIPT_CHARS", _default_chars))
# Local models are usually best run one request at a time
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "1" if PROVIDER == "ollama" else "4"))


# --------------------------------------------------------------------------- #
# LLM wrapper (Ollama or OpenAI)
# --------------------------------------------------------------------------- #
class LLM:
    def __init__(self):
        self.provider = PROVIDER
        if self.provider == "ollama":
            import ollama

            self.client = ollama.Client(host=OLLAMA_HOST, timeout=900)
            self.model = OLLAMA_MODEL
            self._check_ollama()
        elif self.provider == "openai":
            from openai import OpenAI

            key = os.getenv("OPENAI_API_KEY")
            if not key:
                sys.exit("LLM_PROVIDER=openai but OPENAI_API_KEY is not set.")
            self.client = OpenAI(api_key=key)
            self.model = OPENAI_MODEL
        else:
            sys.exit("LLM_PROVIDER must be 'ollama' or 'openai'.")

    def _check_ollama(self):
        try:
            raw = self.client.list()
        except Exception as e:
            sys.exit(f"Cannot reach Ollama at {OLLAMA_HOST} ({e}). Start it with `ollama serve` "
                     "(or open the Ollama app) and retry.")
        items = raw["models"] if isinstance(raw, dict) else getattr(raw, "models", [])
        names = []
        for m in items:
            n = getattr(m, "model", None) or (m.get("model") or m.get("name") if isinstance(m, dict) else None)
            if n:
                names.append(n)
        wanted = self.model
        if not any(n == wanted or n == f"{wanted}:latest" or n.split(":")[0] == wanted for n in names):
            sys.exit(f"Ollama model '{wanted}' not found. Installed: {names or 'none'}.\n"
                     "Set OLLAMA_MODEL in .env to one of those names, or run `ollama pull <model>`.")

    def chat(self, system: str, user: str, json_mode: bool = False, temperature: float = 0.2) -> str:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        if self.provider == "ollama":
            resp = self.client.chat(
                model=self.model,
                messages=messages,
                format="json" if json_mode else None,
                options={"temperature": temperature, "num_ctx": OLLAMA_NUM_CTX},
            )
            text = resp["message"]["content"]
        else:
            kwargs = {"response_format": {"type": "json_object"}} if json_mode else {}
            resp = self.client.chat.completions.create(
                model=self.model, temperature=temperature, messages=messages, **kwargs
            )
            text = resp.choices[0].message.content
        # reasoning models (deepseek-r1, qwen3...) may emit <think> blocks
        return re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL).strip()


def parse_json(text: str) -> dict:
    """Tolerant JSON parsing: handles code fences and text around the object."""
    cleaned = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    for candidate in (cleaned, cleaned[cleaned.find("{"): cleaned.rfind("}") + 1]):
        try:
            return json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue
    return {"error": "model returned invalid JSON", "raw": text}


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
def num(x: Any, default=None):
    """Convert to a clean float (None for NaN / missing)."""
    try:
        v = float(x)
        return default if math.isnan(v) or math.isinf(v) else v
    except (TypeError, ValueError):
        return default


def statement_row(df, *names: str, n: int = 4) -> list[float | None]:
    """Return the latest n annual values for the first matching row label."""
    if df is None or getattr(df, "empty", True):
        return []
    for name in names:
        if name in df.index:
            return [num(v) for v in df.loc[name].iloc[:n].tolist()]
    return []


def growth(series: list[float | None]) -> list[float | None]:
    """YoY growth for a newest-first series."""
    out = []
    for new, old in zip(series, series[1:]):
        out.append(round((new / old - 1) * 100, 2) if new is not None and old not in (None, 0) else None)
    return out


def clip_transcript(text: str, limit: int | None = None) -> str:
    limit = limit or MAX_TRANSCRIPT_CHARS
    if not text.strip():
        return "(No transcript provided - base the analysis on the Yahoo Finance data only and say so.)"
    if len(text) <= limit:
        return text
    half = limit // 2  # keep prepared remarks (start) and Q&A (end)
    return text[:half] + "\n...[transcript truncated]...\n" + text[-half:]


@dataclass
class Context:
    ticker: str
    transcript: str
    info: dict = field(default_factory=dict)
    stock: Any = None

    @property
    def company(self) -> str:
        return self.info.get("longName") or self.info.get("shortName") or self.ticker


class BaseAgent:
    name = "BaseAgent"
    system_prompt = "You are a careful equity research analyst."

    def __init__(self, llm: LLM):
        self.llm = llm

    def llm_json(self, user_prompt: str) -> dict:
        text = self.llm.chat(self.system_prompt + " Respond with valid JSON only, no extra text.",
                             user_prompt, json_mode=True, temperature=0.2)
        return parse_json(text)

    def llm_text(self, user_prompt: str) -> str:
        return self.llm.chat(self.system_prompt, user_prompt, json_mode=False, temperature=0.3)

    def run(self, ctx: Context) -> dict:  # pragma: no cover
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# 1. Financial_data_agent
# --------------------------------------------------------------------------- #
class FinancialDataAgent(BaseAgent):
    name = "Financial_data_agent"
    system_prompt = (
        "You are a fundamental analyst. Interpret financial metrics and cross-check them "
        "against what management said in the earnings call transcript. Never invent numbers."
    )

    def collect(self, ctx: Context) -> dict:
        s, info = ctx.stock, ctx.info
        income = s.financials
        cash = s.cashflow
        revenue = statement_row(income, "Total Revenue", "Operating Revenue")
        net_income = statement_row(income, "Net Income", "Net Income Common Stockholders")
        op_income = statement_row(income, "Operating Income")
        ocf = statement_row(cash, "Operating Cash Flow", "Total Cash From Operating Activities")
        capex = statement_row(cash, "Capital Expenditure")
        return {
            "currency": info.get("financialCurrency"),
            "annual_revenue_latest_first": revenue,
            "revenue_growth_pct": growth(revenue),
            "annual_net_income_latest_first": net_income,
            "net_income_growth_pct": growth(net_income),
            "annual_operating_income_latest_first": op_income,
            "operating_cash_flow": ocf,
            "capex": capex,
            "gross_margin": num(info.get("grossMargins")),
            "operating_margin": num(info.get("operatingMargins")),
            "net_margin": num(info.get("profitMargins")),
            "roe": num(info.get("returnOnEquity")),
            "roa": num(info.get("returnOnAssets")),
            "debt_to_equity": num(info.get("debtToEquity")),
            "current_ratio": num(info.get("currentRatio")),
            "free_cash_flow": num(info.get("freeCashflow")),
            "total_cash": num(info.get("totalCash")),
            "total_debt": num(info.get("totalDebt")),
            "dividend_yield": num(info.get("dividendYield")),
        }

    def run(self, ctx: Context) -> dict:
        data = self.collect(ctx)
        prompt = f"""Company: {ctx.company} ({ctx.ticker})
Financial metrics (Yahoo Finance):
{json.dumps(data, indent=2)}

Earnings call transcript:
{clip_transcript(ctx.transcript)}

Return JSON with keys:
  "summary": 3-5 sentence view on financial health and trajectory,
  "strengths": list of strings,
  "weaknesses": list of strings,
  "management_guidance": list of forward-looking statements from the transcript,
  "transcript_vs_numbers": where management claims match or contradict the numbers,
  "score": integer 1-10 (financial quality)."""
        return {"agent": self.name, "data": data, "analysis": self.llm_json(prompt)}


# --------------------------------------------------------------------------- #
# 2. news_sentiment_agent
# --------------------------------------------------------------------------- #
class NewsSentimentAgent(BaseAgent):
    name = "news_sentiment_agent"
    system_prompt = (
        "You are a market-sentiment analyst. Judge tone from headlines and the earnings call. "
        "Separate noise from information that changes the investment case."
    )

    @staticmethod
    def collect(ctx: Context, limit: int = 12) -> list[dict]:
        try:
            raw = ctx.stock.news or []
        except Exception:
            raw = []
        items = []
        for n in raw[:limit]:
            c = n.get("content", n)  # yfinance changed its news schema
            items.append(
                {
                    "title": c.get("title"),
                    "summary": (c.get("summary") or "")[:200],
                    "publisher": (c.get("provider") or {}).get("displayName") or c.get("publisher"),
                    "published": c.get("pubDate") or c.get("providerPublishTime"),
                }
            )
        return [i for i in items if i["title"]]

    def run(self, ctx: Context) -> dict:
        news = self.collect(ctx)
        prompt = f"""Company: {ctx.company} ({ctx.ticker})
Recent news:
{json.dumps(news, indent=2)}

Earnings call transcript (for management tone):
{clip_transcript(ctx.transcript)}

Return JSON with keys:
  "news_sentiment": "positive" | "neutral" | "negative" | "mixed",
  "transcript_tone": description of management confidence/evasiveness (cite topics, not long quotes),
  "key_themes": list of strings,
  "catalysts": list of strings,
  "red_flags": list of strings,
  "sentiment_score": number from -1 (very negative) to 1 (very positive),
  "summary": 3-4 sentences."""
        return {"agent": self.name, "data": {"headlines": news}, "analysis": self.llm_json(prompt)}


# --------------------------------------------------------------------------- #
# 3. Risk_agent
# --------------------------------------------------------------------------- #
class RiskAgent(BaseAgent):
    name = "Risk_agent"
    system_prompt = (
        "You are a risk analyst covering multinational companies: FX, geopolitics, regulation, "
        "supply chain, leverage, concentration, and macro exposure."
    )

    def collect(self, ctx: Context) -> dict:
        info = ctx.info
        price, high, low = num(info.get("currentPrice")), num(info.get("fiftyTwoWeekHigh")), num(info.get("fiftyTwoWeekLow"))
        return {
            "country": info.get("country"),
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "beta": num(info.get("beta")),
            "debt_to_equity": num(info.get("debtToEquity")),
            "current_ratio": num(info.get("currentRatio")),
            "price": price,
            "52w_high": high,
            "52w_low": low,
            "drawdown_from_52w_high_pct": round((price / high - 1) * 100, 2) if price and high else None,
            "short_percent_of_float": num(info.get("shortPercentOfFloat")),
            "yahoo_overall_governance_risk": info.get("overallRisk"),
            "yahoo_audit_risk": info.get("auditRisk"),
        }

    def run(self, ctx: Context) -> dict:
        data = self.collect(ctx)
        prompt = f"""Company: {ctx.company} ({ctx.ticker})
Risk metrics:
{json.dumps(data, indent=2)}

Earnings call transcript:
{clip_transcript(ctx.transcript)}

Return JSON with keys:
  "risks": list of objects {{"category": "...", "description": "...", "severity": "low|medium|high", "evidence": "..."}}
           covering FX, geopolitical, regulatory, supply chain, macro, leverage, competition where relevant,
  "mitigants": list of strings,
  "overall_risk_level": "low" | "moderate" | "elevated" | "high",
  "risk_score": integer 1-10 (10 = safest),
  "summary": 3-4 sentences."""
        return {"agent": self.name, "data": data, "analysis": self.llm_json(prompt)}


# --------------------------------------------------------------------------- #
# 4. Valuation_agent
# --------------------------------------------------------------------------- #
class ValuationAgent(BaseAgent):
    name = "Valuation_agent"
    system_prompt = (
        "You are a valuation analyst. Judge whether the stock is cheap, fair or expensive using "
        "multiples and the DCF scenarios provided. State assumptions and their limits."
    )

    SCENARIOS = {  # (5-yr FCF growth, discount rate, terminal growth)
        "bear": (0.03, 0.11, 0.015),
        "base": (0.07, 0.10, 0.025),
        "bull": (0.11, 0.09, 0.030),
    }

    def dcf(self, ctx: Context) -> dict:
        info = ctx.info
        fcf = num(info.get("freeCashflow"))
        shares = num(info.get("sharesOutstanding"))
        if not fcf or fcf <= 0 or not shares:
            return {"note": "DCF skipped: free cash flow missing or non-positive."}
        net_debt = (num(info.get("totalDebt"), 0) or 0) - (num(info.get("totalCash"), 0) or 0)
        out = {}
        for name, (g, r, tg) in self.SCENARIOS.items():
            pv, cf = 0.0, fcf
            for t in range(1, 6):
                cf *= 1 + g
                pv += cf / (1 + r) ** t
            terminal = cf * (1 + tg) / (r - tg)
            equity = pv + terminal / (1 + r) ** 5 - net_debt
            out[name] = {
                "assumptions": {"fcf_growth": g, "discount_rate": r, "terminal_growth": tg},
                "intrinsic_value_per_share": round(equity / shares, 2),
            }
        return out

    def collect(self, ctx: Context) -> dict:
        info = ctx.info
        price = num(info.get("currentPrice"))
        dcf = self.dcf(ctx)
        base = dcf.get("base", {}).get("intrinsic_value_per_share")
        return {
            "price": price,
            "currency": info.get("currency"),
            "market_cap": num(info.get("marketCap")),
            "trailing_pe": num(info.get("trailingPE")),
            "forward_pe": num(info.get("forwardPE")),
            "peg": num(info.get("pegRatio")),
            "price_to_book": num(info.get("priceToBook")),
            "price_to_sales": num(info.get("priceToSalesTrailing12Months")),
            "ev_to_ebitda": num(info.get("enterpriseToEbitda")),
            "analyst_target_mean": num(info.get("targetMeanPrice")),
            "analyst_recommendation": info.get("recommendationKey"),
            "dcf_scenarios": dcf,
            "base_case_upside_pct": round((base / price - 1) * 100, 1) if base and price else None,
        }

    def run(self, ctx: Context) -> dict:
        data = self.collect(ctx)
        prompt = f"""Company: {ctx.company} ({ctx.ticker})
Valuation data:
{json.dumps(data, indent=2)}

Management outlook from the earnings call (use to sanity-check DCF growth assumptions):
{clip_transcript(ctx.transcript, MAX_TRANSCRIPT_CHARS // 2)}

Return JSON with keys:
  "valuation_view": "undervalued" | "fairly valued" | "overvalued",
  "fair_value_range": {{"low": number|null, "high": number|null}},
  "key_assumptions_and_caveats": list of strings,
  "multiples_commentary": string,
  "summary": 3-4 sentences,
  "score": integer 1-10 (10 = very attractive valuation)."""
        return {"agent": self.name, "data": data, "analysis": self.llm_json(prompt)}


# --------------------------------------------------------------------------- #
# 5. ReportWriterAgent
# --------------------------------------------------------------------------- #
class ReportWriterAgent(BaseAgent):
    name = "ReportWriterAgent"
    system_prompt = (
        "You are a senior equity research writer. Combine specialist analyses into a clear, "
        "balanced, decision-ready report. Flag disagreements between agents. Use only the "
        "supplied data; if something is missing, say so."
    )

    def run(self, ctx: Context, results: dict[str, dict]) -> str:
        # Keep the payload compact so it fits a local model's context window
        slim = {
            name: {"data": r.get("data"), "analysis": r.get("analysis"), "error": r.get("error")}
            for name, r in results.items()
        }
        payload = json.dumps(slim, default=str, separators=(",", ":"))
        prompt = f"""Write a fundamental analysis report in Markdown for {ctx.company} ({ctx.ticker}).

Specialist agent outputs:
{payload}

Structure:
# {ctx.company} ({ctx.ticker}) - Fundamental Analysis
## 1. Executive Summary
## 2. Financial Health
## 3. News & Management Sentiment
## 4. Risk Assessment
## 5. Valuation
## 6. Bull vs Bear Case
## 7. Evaluation Remark
   - Rating: one of Strong Buy / Buy / Hold / Reduce / Sell
   - Conviction: Low / Medium / High
   - 3-5 sentence rationale tying together the four agents' scores
   - Key things to monitor next quarter
End with a one-line disclaimer that this is automated analysis, not investment advice."""
        return self.llm_text(prompt)


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #
def analyze(ticker: str, transcript: str, out_dir: str = "reports") -> str:
    llm = LLM()
    print(f"LLM: {llm.provider} / {llm.model}")

    stock = yf.Ticker(ticker)
    info = stock.info or {}
    if not info or not (info.get("longName") or info.get("shortName")):
        sys.exit(f"Yahoo Finance returned no data for '{ticker}'. Check the symbol (e.g. NESN.SW, 7203.T).")
    ctx = Context(ticker=ticker.upper(), transcript=transcript, info=info, stock=stock)

    specialists = [FinancialDataAgent(llm), NewsSentimentAgent(llm), RiskAgent(llm), ValuationAgent(llm)]

    print(f"Running {len(specialists)} agents for {ctx.company} (workers={MAX_WORKERS}). "
          "Local models can take a few minutes...")
    results: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(a.run, ctx): a.name for a in specialists}
        for fut, name in futures.items():
            try:
                results[name] = fut.result()
                print(f"  done: {name}")
            except Exception as e:  # one failing agent shouldn't kill the report
                results[name] = {"agent": name, "error": str(e)}
                print(f"  FAILED: {name}: {e}")

    print("Writing report...")
    report = ReportWriterAgent(llm).run(ctx, results)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    report_path = out / f"{ctx.ticker}_report.md"
    report_path.write_text(report, encoding="utf-8")
    (out / f"{ctx.ticker}_agent_outputs.json").write_text(
        json.dumps(results, indent=2, default=str), encoding="utf-8"
    )
    print(f"Saved: {report_path}")
    return report


def main():
    ap = argparse.ArgumentParser(description="Multi-agent fundamental analysis from earnings transcripts")
    ap.add_argument("ticker", help="Yahoo Finance symbol, e.g. AAPL, MSFT, NESN.SW")
    ap.add_argument("--transcript", default=None, help="Path to a .txt earnings call transcript (optional)")
    ap.add_argument("--out", default="reports", help="Output directory")
    args = ap.parse_args()

    transcript = ""
    if args.transcript:
        tpath = Path(args.transcript)
        if not tpath.exists():
            sys.exit(f"Transcript file not found: {tpath}")
        transcript = tpath.read_text(encoding="utf-8", errors="ignore")
    else:
        print("No --transcript given: running on Yahoo Finance data only.")

    print("\n" + analyze(args.ticker, transcript, args.out))


if __name__ == "__main__":
    main()
