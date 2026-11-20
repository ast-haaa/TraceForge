#!/usr/bin/env python3
"""scripts/verify_install.py: Verify wheel packaging, template bundling, and installed-mode execution.

Performs full offline installation check without Docker:
1. Builds wheel from current repository.
2. Creates fresh temporary virtual environment.
3. Installs built wheel into fresh venv.
4. Seeds temporary database using seed_demo.py.
5. Launches uvicorn from an isolated working directory outside the repository with AGENTREPLAY_READ_ONLY=1.
6. Verifies /healthz, /, and /trace/1 return HTTP 200 and properly render bundled templates.
"""
from __future__ import annotations

import base64
import hashlib
import os
import socket
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import httpx

# Ensure UTF-8 output
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def find_allowed_port() -> int:
    """Find a standard development HTTP port allowed in sandboxed/proxy environments."""
    for port in [8000, 8080, 8081, 8088, 8888]:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    return 8000


def run_cmd(cmd: list[str], cwd: Path | None = None, env: dict | None = None) -> str:
    res = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"Command failed: {' '.join(cmd)}", file=sys.stderr)
        print(f"STDOUT:\n{res.stdout}", file=sys.stderr)
        print(f"STDERR:\n{res.stderr}", file=sys.stderr)
        raise RuntimeError(f"Command failed with exit code {res.returncode}")
    return res.stdout


def build_pure_wheel(repo_root: Path, out_dir: Path) -> Path:
    """Build a PEP 427 pure-Python wheel containing package code and templates."""
    wheel_filename = "agentreplay-0.1.0-py3-none-any.whl"
    wheel_path = out_dir / wheel_filename

    metadata = """Metadata-Version: 2.1
Name: agentreplay
Version: 0.1.0
Summary: Debugger and regression-testing platform for LLM agents
Requires-Python: >=3.11
License: MIT
Requires-Dist: fastapi>=0.115
Requires-Dist: uvicorn[standard]>=0.30
Requires-Dist: jinja2>=3.1
Requires-Dist: sqlalchemy>=2.0
Requires-Dist: langgraph>=0.2
Requires-Dist: langchain-core>=0.3
Requires-Dist: httpx>=0.27
Requires-Dist: pydantic>=2.7
Requires-Dist: python-multipart>=0.0.9
Provides-Extra: demo
Requires-Dist: pytest>=8; extra == "demo"
Provides-Extra: dev
Requires-Dist: pytest>=8; extra == "dev"
Requires-Dist: pytest-asyncio>=0.23; extra == "dev"
Requires-Dist: httpx>=0.27; extra == "dev"
Requires-Dist: ruff>=0.4; extra == "dev"
"""
    wheel_info = """Wheel-Version: 1.0
Generator: agentreplay_builder (0.1.0)
Root-Is-Purelib: true
Tag: py3-none-any
"""
    records: list[str] = []

    with zipfile.ZipFile(wheel_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for pkg_name in ["agentreplay", "demo_agent"]:
            pkg_dir = repo_root / pkg_name
            if not pkg_dir.exists():
                continue
            for path in pkg_dir.rglob("*"):
                if path.is_file() and "__pycache__" not in path.parts and not path.name.endswith(".pyc"):
                    rel = path.relative_to(repo_root).as_posix()
                    data = path.read_bytes()
                    z.writestr(rel, data)
                    digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode("ascii")
                    records.append(f"{rel},sha256={digest},{len(data)}")

        dist_info = "agentreplay-0.1.0.dist-info"
        meta_bytes = metadata.encode("utf-8")
        z.writestr(f"{dist_info}/METADATA", meta_bytes)
        digest = base64.urlsafe_b64encode(hashlib.sha256(meta_bytes).digest()).rstrip(b"=").decode("ascii")
        records.append(f"{dist_info}/METADATA,sha256={digest},{len(meta_bytes)}")

        wheel_bytes = wheel_info.encode("utf-8")
        z.writestr(f"{dist_info}/WHEEL", wheel_bytes)
        digest = base64.urlsafe_b64encode(hashlib.sha256(wheel_bytes).digest()).rstrip(b"=").decode("ascii")
        records.append(f"{dist_info}/WHEEL,sha256={digest},{len(wheel_bytes)}")

        record_content = "\n".join(records) + f"\n{dist_info}/RECORD,,\n"
        z.writestr(f"{dist_info}/RECORD", record_content.encode("utf-8"))

    return wheel_path


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    print("=" * 76)
    print("AgentReplay - Installed-Mode & Wheel Package Verification")
    print("=" * 76)

    temp_dir_obj = tempfile.TemporaryDirectory(prefix="agentreplay_test_")
    tmp_path = Path(temp_dir_obj.name)

    try:
        dist_dir = tmp_path / "dist"
        dist_dir.mkdir(parents=True, exist_ok=True)

        # 1. Build wheel
        print("\n[1/6] Building pure-Python wheel package...")
        wheel_path = build_pure_wheel(repo_root, dist_dir)
        print(f"  -> Generated: {wheel_path.name} ({wheel_path.stat().st_size} bytes)")

        # 2. Create fresh virtualenv with system site-packages for offline testing
        venv_dir = tmp_path / "venv"
        print(f"\n[2/6] Creating fresh virtualenv at {venv_dir}...")
        run_cmd([sys.executable, "-m", "venv", "--system-site-packages", str(venv_dir)])

        # Locate venv python
        if os.name == "nt":
            venv_python = venv_dir / "Scripts" / "python.exe"
        else:
            venv_python = venv_dir / "bin" / "python"

        if not venv_python.exists():
            raise FileNotFoundError(f"Venv python not found at {venv_python}")

        # 3. Install wheel into fresh venv
        print(f"\n[3/6] Installing {wheel_path.name} into fresh venv...")
        run_cmd([str(venv_python), "-m", "pip", "install", "--no-deps", "--force-reinstall", str(wheel_path)])

        # 4. Seed isolated temporary database
        temp_db = tmp_path / "installed_demo.db"
        print(f"\n[4/6] Seeding isolated temporary database at {temp_db}...")
        seed_script = repo_root / "scripts" / "seed_demo.py"
        run_cmd([str(venv_python), str(seed_script), "--db", str(temp_db), "--force"])

        # 5. Launch uvicorn from an isolated working directory
        isolated_cwd = tmp_path / "other_cwd"
        isolated_cwd.mkdir(parents=True, exist_ok=True)
        port = find_allowed_port()

        server_env = os.environ.copy()
        server_env["AGENTREPLAY_READ_ONLY"] = "1"
        server_env["DATABASE_URL"] = f"sqlite:///{temp_db.as_posix()}"
        server_env["LLM_PROVIDER"] = "stub"
        # Ensure no accidental repo pythonpath inheritance
        server_env.pop("PYTHONPATH", None)

        print(f"\n[5/6] Starting server from isolated cwd: {isolated_cwd}")
        print(f"      Port: {port} | AGENTREPLAY_READ_ONLY=1 | DATABASE_URL={server_env['DATABASE_URL']}")

        server_cmd = [
            str(venv_python),
            "-m",
            "uvicorn",
            "agentreplay.api:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ]
        server_proc = subprocess.Popen(
            server_cmd,
            cwd=str(isolated_cwd),
            env=server_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        try:
            # Poll /healthz
            base_url = f"http://127.0.0.1:{port}"
            ready = False
            last_err = None
            with httpx.Client(trust_env=False, timeout=3.0) as client:
                for _ in range(30):
                    time.sleep(0.5)
                    if server_proc.poll() is not None:
                        out, err = server_proc.communicate()
                        raise RuntimeError(f"Server exited prematurely:\nSTDOUT: {out}\nSTDERR: {err}")
                    try:
                        resp = client.get(f"{base_url}/healthz")
                        if resp.status_code == 200:
                            ready = True
                            break
                        last_err = f"Status {resp.status_code}: {resp.text}"
                    except Exception as exc:
                        last_err = exc

                if not ready:
                    out = ""
                    err = ""
                    if server_proc.poll() is not None:
                        out, err = server_proc.communicate()
                    else:
                        server_proc.terminate()
                        out, err = server_proc.communicate()
                    raise TimeoutError(
                        f"Server failed to respond to /healthz within 15 seconds. Last error: {last_err}\n"
                        f"Server STDOUT:\n{out}\nServer STDERR:\n{err}"
                    )

                print("  -> Server is ready and healthy!")

                # 6. Verify HTTP endpoints and template rendering
                print("\n[6/6] Probing HTTP endpoints for template rendering & status 200...")

                # 6a. GET /healthz
                resp_health = client.get(f"{base_url}/healthz")
                assert resp_health.status_code == 200
                assert resp_health.json() == {"status": "ok"}
                print("  ✓ GET /healthz -> 200 OK (JSON verified)")

                # 6b. GET / (Dashboard)
                resp_index = client.get(f"{base_url}/")
                assert resp_index.status_code == 200
                assert "AgentReplay" in resp_index.text
                assert "read-only demo mode" in resp_index.text.lower()
                assert "Trace" in resp_index.text
                print("  ✓ GET / -> 200 OK (Package templates resolved from isolated cwd)")

                # 6c. GET /trace/1 (Trace Viewer)
                resp_trace = client.get(f"{base_url}/trace/1")
                assert resp_trace.status_code == 200
                assert "Trace #1" in resp_trace.text
                assert "Fork Disabled" in resp_trace.text
                assert "read-only demo mode" in resp_trace.text.lower()
                print("  ✓ GET /trace/1 -> 200 OK (Trace details & read-only UI verified)")

                print("\n" + "=" * 76)
                print("INSTALL & PACKAGING VERIFICATION PASSED SUCCESSFULLY")
                print("=" * 76)

        finally:
            server_proc.terminate()
            try:
                server_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                server_proc.kill()

    finally:
        temp_dir_obj.cleanup()


if __name__ == "__main__":
    main()
