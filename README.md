# Agent Harnesses — code companion

Runnable companion code for the chapter. Seven scripts, all working the **same
task against the same corpus**: produce a competitive brief on the vector search
infrastructure market. Each script adds one layer of the harness, so the example
compounds instead of restarting.

Everything here is self-contained — this folder does not depend on any other
project.

| Script | Section | What it shows |
|---|---|---|
| `00_react_baseline.py` | 1 | The task on a plain tool-calling agent, instrumented. It falls over. |
| `01_stack.py`          | 3 | The harness *is* a middleware stack — read off the object. **No API calls.** |
| `02_plan.py`           | 4 | `TodoListMiddleware`: the plan is structured state, not prose |
| `03_files.py`          | 5 | `FilesystemMiddleware`: automatic offload past ~20k tokens, plus permissions |
| `04_delegate.py`       | 6 | `SubAgentMiddleware`: what does and does not cross the boundary |
| `05_prompt.py`         | 7 | A static prompt with an edge, then one assembled per turn |
| `06_middleware.py`     | 8 | Write your own: a tool ledger and a budget guard |

`sources.py` is the fixed research corpus every script shares. Run it directly
(`python sources.py`) to see what is in it and how big each source is.

## Setup

You need Python 3.11–3.14 and an [OpenAI API key](https://platform.openai.com/api-keys).
These steps use [uv](https://docs.astral.sh/uv/); a pip path is below.

**1. Install uv**

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
# macOS (Homebrew):  brew install uv
# Windows:  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**2. Add your API key**

```bash
cp .env.example .env
# open .env and set OPENAI_API_KEY=...
```

**3. Install dependencies**

```bash
uv sync
```

**4. Check it works**

```bash
uv run python 01_stack.py
```

This one makes no model calls, so it confirms your install without spending
anything.

## Run a lab

```bash
uv run python 00_react_baseline.py
uv run python 01_stack.py
uv run python 02_plan.py
uv run python 03_files.py
uv run python 04_delegate.py
uv run python 05_prompt.py
uv run python 06_middleware.py
```

### Using pip instead of uv

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python 01_stack.py
```

## Notes

- **Changing the model.** Every script does `from models import model`.
  `models.py` is the only place a model is named — edit that one line to switch
  model or provider and every lab follows.
- **The corpus is fictional.** The vendors, documents, and figures in
  `sources.py` are invented. The market shape is realistic, but using made-up
  companies means the chapter cannot misquote a real one or go stale.
- **The numbers move.** Token counts and call counts in the chapter come from
  real runs, but models are not deterministic — expect your figures to differ in
  the details while the shape of the result holds.
- **`00_react_baseline.py` is the expensive one.** It has no harness holding the
  context down, which is the entire point; it re-sends a large transcript on
  every turn.
- **Tracing is optional.** To watch a run as a trace, set `LANGSMITH_TRACING=true`
  and add `LANGSMITH_API_KEY` in `.env`.
