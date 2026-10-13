# HW2 Task 2 - Focusing through a simulated bench

- `bench.py`             simulated DMD -> scattering medium -> camera, and the flip search
- `focus_tool.py`        `focus_beam` tool (saves images, logs to results/experiments.jsonl)
- `agent_integration.py` executor node for the HW1 Task 6 graph with `focus_beam` bound
- `run_campaign.py`      direct tool calls without the LLM (bench test)
- `make_report.py`       builds the PDF report from results/experiments.jsonl

Agent run:  `rm -rf results && python agent_integration.py && GITHUB_URL=<repo> python make_report.py`
Bench test: `python run_campaign.py && python make_report.py`
