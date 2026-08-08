"""
Agentless API Observability Platform
--------------------------------------
Ingests structured API error logs -> pulls relevant source code from GitHub
-> sends stack trace + code to an LLM (using the user's own API key)
-> gets back root cause + a deploy-ready patch -> optionally opens a GitHub PR.

Run:
    pip install -r requirements.txt
    uvicorn app.main:app --reload

All endpoints are documented automatically at /docs (Swagger UI).
"""

import base64
import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime
from typing import Optional

import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(
    title="Agentless API Observability Platform",
    description="Ingest API errors, root-cause them with an LLM, and ship a fix.",
    version="0.1.0",
)

DB_PATH = "observability.db"


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------

@contextmanager
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with get_db() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS logs (
                id TEXT PRIMARY KEY,
                received_at TEXT,
                endpoint TEXT,
                method TEXT,
                status_code INTEGER,
                stack_trace TEXT,
                request_payload TEXT,
                repo_owner TEXT,
                repo_name TEXT,
                repo_branch TEXT,
                raw_json TEXT,
                root_cause TEXT,
                affected_files TEXT,
                patch TEXT,
                pr_url TEXT
            )
            """
        )


init_db()


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class RepoContext(BaseModel):
    owner: str
    name: str
    branch: str = "main"


class APILogIn(BaseModel):
    """Structured log payload pushed by the client application (agentless)."""
    endpoint: str
    method: str = "GET"
    status_code: int
    stack_trace: str
    request_payload: Optional[dict] = None
    repo: RepoContext = Field(..., description="Repo to search for the failing code")


class AnalyzeRequest(BaseModel):
    llm_provider: str = Field(..., description="'anthropic' or 'openai'")
    llm_api_key: str = Field(..., description="User-supplied LLM API key, used only for this call")
    github_token: Optional[str] = Field(None, description="Needed to read private repos / open PRs")


class CreatePRRequest(BaseModel):
    github_token: str
    base_branch: str = "main"


# ---------------------------------------------------------------------------
# 1. Ingestion endpoint  (agentless: apps just POST their error logs here)
# ---------------------------------------------------------------------------

@app.post("/api/logs")
def ingest_log(log: APILogIn):
    log_id = str(uuid.uuid4())
    with get_db() as conn:
        conn.execute(
            """INSERT INTO logs
               (id, received_at, endpoint, method, status_code, stack_trace,
                request_payload, repo_owner, repo_name, repo_branch, raw_json)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (
                log_id,
                datetime.utcnow().isoformat(),
                log.endpoint,
                log.method,
                log.status_code,
                log.stack_trace,
                json.dumps(log.request_payload or {}),
                log.repo.owner,
                log.repo.name,
                log.repo.branch,
                log.model_dump_json(),
            ),
        )
    return {"log_id": log_id, "status": "ingested"}


@app.get("/api/logs")
def list_logs():
    with get_db() as conn:
        rows = conn.execute("SELECT id, received_at, endpoint, status_code FROM logs ORDER BY received_at DESC").fetchall()
    return [dict(r) for r in rows]


@app.get("/api/logs/{log_id}")
def get_log(log_id: str):
    with get_db() as conn:
        row = conn.execute("SELECT * FROM logs WHERE id=?", (log_id,)).fetchone()
    if not row:
        raise HTTPException(404, "log not found")
    return dict(row)


# ---------------------------------------------------------------------------
# 2. GitHub integration - pull source files referenced in the stack trace
# ---------------------------------------------------------------------------

def extract_candidate_files(stack_trace: str) -> list[str]:
    """Very lightweight heuristic: pull file-like tokens out of a stack trace.
    Works for Python ('File "app/foo.py", line 12') and JS/Node style
    ('at Object.<anonymous> (src/bar.js:20:5)') traces."""
    import re

    candidates = set()
    for match in re.findall(r'File "([^"]+)"', stack_trace):
        candidates.add(match)
    for match in re.findall(r"\(([\w\-/\.]+\.(?:js|ts|py|go|java)):\d+", stack_trace):
        candidates.add(match)
    for match in re.findall(r"([\w\-/\.]+\.(?:js|ts|py|go|java)):\d+", stack_trace):
        candidates.add(match)
    return list(candidates)


def fetch_github_file(owner: str, repo: str, path: str, branch: str, token: Optional[str]) -> Optional[str]:
    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    resp = requests.get(url, headers=headers, params={"ref": branch})
    if resp.status_code != 200:
        return None
    data = resp.json()
    if data.get("encoding") == "base64":
        return base64.b64decode(data["content"]).decode("utf-8", errors="replace")
    return None


# ---------------------------------------------------------------------------
# 3. LLM analysis - root cause + patch generation
# ---------------------------------------------------------------------------

ANALYSIS_SYSTEM_PROMPT = """You are an expert backend engineer doing automated root-cause analysis.
Given a stack trace, request payload, and relevant source file contents, you must:
1. Identify the root cause of the failure in one or two sentences.
2. List the specific file(s) responsible.
3. Produce a minimal, deploy-ready unified diff (patch) that fixes the bug.
Respond ONLY with valid JSON in this exact shape, no markdown fences, no commentary:
{
  "root_cause": "string",
  "affected_files": ["path/to/file.py"],
  "patch": "unified diff as a string"
}"""


def call_anthropic(api_key: str, user_prompt: str) -> str:
    resp = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": "claude-sonnet-4-6",
            "max_tokens": 2000,
            "system": ANALYSIS_SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": user_prompt}],
        },
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    return "".join(block.get("text", "") for block in data.get("content", []))


def call_openai(api_key: str, user_prompt: str) -> str:
    resp = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": "gpt-4o",
            "max_tokens": 2000,
            "messages": [
                {"role": "system", "content": ANALYSIS_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        },
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


def parse_llm_json(raw: str) -> dict:
    cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    return json.loads(cleaned)


@app.post("/api/logs/{log_id}/analyze")
def analyze_log(log_id: str, req: AnalyzeRequest):
    with get_db() as conn:
        row = conn.execute("SELECT * FROM logs WHERE id=?", (log_id,)).fetchone()
    if not row:
        raise HTTPException(404, "log not found")

    candidate_paths = extract_candidate_files(row["stack_trace"])
    file_contents = {}
    for path in candidate_paths[:5]:  # cap to keep prompt small
        content = fetch_github_file(row["repo_owner"], row["repo_name"], path, row["repo_branch"], req.github_token)
        if content:
            file_contents[path] = content

    user_prompt = f"""
Endpoint: {row['method']} {row['endpoint']}
Status code: {row['status_code']}
Stack trace:
{row['stack_trace']}

Request payload:
{row['request_payload']}

Relevant source files:
{json.dumps(file_contents, indent=2)[:12000]}
"""

    if req.llm_provider == "anthropic":
        raw = call_anthropic(req.llm_api_key, user_prompt)
    elif req.llm_provider == "openai":
        raw = call_openai(req.llm_api_key, user_prompt)
    else:
        raise HTTPException(400, "llm_provider must be 'anthropic' or 'openai'")

    try:
        result = parse_llm_json(raw)
    except json.JSONDecodeError:
        raise HTTPException(502, f"LLM did not return valid JSON: {raw[:500]}")

    with get_db() as conn:
        conn.execute(
            "UPDATE logs SET root_cause=?, affected_files=?, patch=? WHERE id=?",
            (
                result.get("root_cause"),
                json.dumps(result.get("affected_files", [])),
                result.get("patch"),
                log_id,
            ),
        )

    return result


# ---------------------------------------------------------------------------
# 4. Deploy-ready output - download patch, or auto-open a GitHub PR
# ---------------------------------------------------------------------------

@app.get("/api/logs/{log_id}/patch")
def download_patch(log_id: str):
    with get_db() as conn:
        row = conn.execute("SELECT patch FROM logs WHERE id=?", (log_id,)).fetchone()
    if not row or not row["patch"]:
        raise HTTPException(404, "no patch generated yet - call /analyze first")
    return {"patch": row["patch"]}


@app.post("/api/logs/{log_id}/create-pr")
def create_pr(log_id: str, req: CreatePRRequest):
    """Creates a branch, commits the patched file(s), and opens a PR via the GitHub API."""
    with get_db() as conn:
        row = conn.execute("SELECT * FROM logs WHERE id=?", (log_id,)).fetchone()
    if not row or not row["patch"]:
        raise HTTPException(404, "no patch to apply - call /analyze first")

    owner, repo = row["repo_owner"], row["repo_name"]
    headers = {"Authorization": f"Bearer {req.github_token}", "Accept": "application/vnd.github+json"}

    # 1. Get base branch SHA
    ref = requests.get(f"https://api.github.com/repos/{owner}/{repo}/git/ref/heads/{req.base_branch}", headers=headers)
    ref.raise_for_status()
    base_sha = ref.json()["object"]["sha"]

    # 2. Create a new branch
    new_branch = f"auto-fix/{log_id[:8]}"
    requests.post(
        f"https://api.github.com/repos/{owner}/{repo}/git/refs",
        headers=headers,
        json={"ref": f"refs/heads/{new_branch}", "sha": base_sha},
    )

    # NOTE: applying a unified diff via the GitHub Contents API requires
    # parsing the diff and updating each file's content individually.
    # For a hackathon demo, keep it simple: commit the patch as a reviewable
    # .patch file in the new branch rather than auto-applying it blindly.
    patch_path = f"auto-fixes/{log_id[:8]}.patch"
    requests.put(
        f"https://api.github.com/repos/{owner}/{repo}/contents/{patch_path}",
        headers=headers,
        json={
            "message": f"Add auto-generated fix for {row['endpoint']} ({row['status_code']})",
            "content": base64.b64encode(row["patch"].encode()).decode(),
            "branch": new_branch,
        },
    )

    # 3. Open the PR
    pr_resp = requests.post(
        f"https://api.github.com/repos/{owner}/{repo}/pulls",
        headers=headers,
        json={
            "title": f"Auto-fix: {row['endpoint']} returning {row['status_code']}",
            "head": new_branch,
            "base": req.base_branch,
            "body": f"**Root cause:** {row['root_cause']}\n\n**Affected files:** {row['affected_files']}\n\nGenerated automatically by the API Observability Platform.",
        },
    )
    pr_resp.raise_for_status()
    pr_url = pr_resp.json()["html_url"]

    with get_db() as conn:
        conn.execute("UPDATE logs SET pr_url=? WHERE id=?", (pr_url, log_id))

    return {"pr_url": pr_url}
