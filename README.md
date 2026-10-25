# Multi-Agent Fundamental Analysis (local Ollama)

Agents: Financial_data_agent, news_sentiment_agent, Risk_agent, Valuation_agent -> ReportWriterAgent
(report + evaluation remark). Data: Yahoo Finance (no key) + a local Ollama model (no key).

## Run in VS Code
1. Make sure Ollama is running and you have a model:  `ollama list`  (else `ollama pull llama3.1`)
2. Open this folder in VS Code (File > Open Folder), then open a terminal:
   - Windows:  `python -m venv .venv` then `.venv\Scripts\activate`
   - Mac/Linux: `python3 -m venv .venv` then `source .venv/bin/activate`
3. `pip install -r requirements.txt`
4. Copy `.env.example` to `.env` and set OLLAMA_MODEL to a name from `ollama list`.
5. Run:
   - Press F5 and enter a ticker, or
   - `python fundamental_analysis_agents.py AAPL --transcript transcripts/aapl_q3.txt`
   - No transcript? `python fundamental_analysis_agents.py MSFT`

Output: `reports/<TICKER>_report.md` and `reports/<TICKER>_agent_outputs.json`.
Automated analysis, not investment advice.
