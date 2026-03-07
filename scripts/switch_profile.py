#!/usr/bin/env python3
import argparse
import re
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List, Tuple

PROFILES: Dict[str, Dict[str, str]] = {
    "local": {
        "LLM_PROVIDER": "ollama",
        "LLM_MODEL": "qwen2.5:14b",
        "OLLAMA_BASE_URL": "http://localhost:11434/v1",
        "EMBEDDING_BACKEND": "local",
        "EMBEDDING_MODEL": "Qwen/Qwen3-VL-Embedding-2B",
        "DOCUMENT_PARSER_BACKEND": "unstructured",
        "VISION_ENABLED": "false",
        "VISION_MODEL": "llama3.2-vision:latest",
    },
    "local-safe": {
        "LLM_PROVIDER": "ollama",
        "LLM_MODEL": "llama3.2:latest",
        "OLLAMA_BASE_URL": "http://localhost:11434/v1",
        "EMBEDDING_BACKEND": "local",
        "EMBEDDING_MODEL": "Qwen/Qwen3-VL-Embedding-2B",
        "DOCUMENT_PARSER_BACKEND": "builtin",
        "VISION_ENABLED": "false",
        "VISION_MODEL": "llama3.2-vision:latest",
    },
    "local-vision": {
        "LLM_PROVIDER": "ollama",
        "LLM_MODEL": "qwen2.5:14b",
        "OLLAMA_BASE_URL": "http://localhost:11434/v1",
        "EMBEDDING_BACKEND": "local",
        "EMBEDDING_MODEL": "Qwen/Qwen3-VL-Embedding-2B",
        "DOCUMENT_PARSER_BACKEND": "unstructured",
        "VISION_ENABLED": "true",
        "VISION_MODEL": "llama3.2-vision:latest",
    },
    "cloud": {
        "LLM_PROVIDER": "heiyucode",
        "LLM_MODEL": "gpt-5.3-codex",
        "OPENAI_BASE_URL": "https://www.heiyucode.com/v1",
        "EMBEDDING_BACKEND": "dashscope",
        "EMBEDDING_MODEL": "qwen3-vl-embedding",
        "DASHSCOPE_EMBEDDING_MODEL": "qwen3-vl-embedding",
        "DOCUMENT_PARSER_BACKEND": "unstructured",
        "VISION_ENABLED": "true",
        "VISION_MODEL": "gpt-4o-mini",
    },
}

KEY_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=")
KNOWN_KEYS = sorted({k for p in PROFILES.values() for k in p.keys()})

PROFILE_DESCRIPTIONS: Dict[str, str] = {
    "local": "本地默认档：Ollama + 本地 embedding + unstructured 解析，适合日常本地开发。",
    "local-safe": "本地保守档：更轻的本地 LLM + builtin parser + vision 关闭，优先稳定性。",
    "local-vision": "本地视觉档：保留本地 embedding，并启用本地 vision 模型。",
    "cloud": "云端档：heiyucode + dashscope，适合云端资源与在线模型。",
}


def _read_lines(path: Path) -> List[str]:
    if not path.exists():
        return []
    return path.read_text(encoding="utf-8").splitlines()


def _write_lines(path: Path, lines: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(lines).rstrip("\n") + "\n"
    path.write_text(text, encoding="utf-8")


def _upsert(lines: List[str], updates: Dict[str, str]) -> Tuple[List[str], Dict[str, Tuple[str, str]]]:
    out = list(lines)
    idx_map: Dict[str, int] = {}
    old_map: Dict[str, str] = {}

    for i, line in enumerate(out):
        m = KEY_RE.match(line)
        if not m:
            continue
        key = m.group(1)
        idx_map[key] = i
        old_map[key] = line.split("=", 1)[1] if "=" in line else ""

    changed: Dict[str, Tuple[str, str]] = {}
    for key, new_val in updates.items():
        new_line = f"{key}={new_val}"
        if key in idx_map:
            i = idx_map[key]
            old_line = out[i]
            out[i] = new_line
            old_val = old_map.get(key, "")
            if old_line != new_line:
                changed[key] = (old_val, new_val)
        else:
            out.append(new_line)
            changed[key] = ("<missing>", new_val)

    return out, changed


def _default_targets(cwd: Path) -> List[Path]:
    return [
        cwd / "backend/.env",
        cwd / "deploy/backend.env",
    ]


def _parse_kv(lines: List[str]) -> Dict[str, str]:
    kv: Dict[str, str] = {}
    for line in lines:
        m = KEY_RE.match(line)
        if not m or "=" not in line:
            continue
        key = m.group(1)
        kv[key] = line.split("=", 1)[1]
    return kv


def _print_check(targets: List[Path], profile_keys: List[str]) -> None:
    for target in targets:
        print(f"\n[check] {target}")
        lines = _read_lines(target)
        if not lines:
            print("  file not found or empty")
            continue
        kv = _parse_kv(lines)
        for key in profile_keys:
            print(f"  {key}={kv.get(key, '<missing>')}")



def _print_profiles() -> None:
    print("Available profiles:")
    for name in sorted(PROFILES.keys()):
        print(f"\n- {name}")
        print(f"  说明: {PROFILE_DESCRIPTIONS.get(name, '无说明')}")
        for key, value in PROFILES[name].items():
            print(f"  {key}={value}")
def _run_shell(cmd: str, cwd: Path = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        shell=True,
        cwd=str(cwd) if cwd else None,
        text=True,
        capture_output=True,
    )


def _wait_health(health_url: str, wait_seconds: int) -> bool:
    deadline = time.time() + max(1, wait_seconds)
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(health_url, timeout=3) as resp:  # nosec B310
                if 200 <= resp.status < 300:
                    return True
        except urllib.error.URLError:
            pass
        except Exception:
            pass
        time.sleep(1)
    return False


def _restart_local_backend(
    backend_port: int,
    backend_cmd: str,
    backend_cwd: Path,
    health_url: str,
    wait_seconds: int,
) -> None:
    # Stop existing backend process (if any) listening on backend_port.
    find_res = _run_shell(f"lsof -ti tcp:{backend_port}")
    if find_res.returncode == 0 and find_res.stdout.strip():
        pids = sorted(set(find_res.stdout.strip().split()))
        kill_res = _run_shell(f"kill {' '.join(pids)}")
        if kill_res.returncode != 0:
            raise RuntimeError(f"failed to stop backend: {kill_res.stderr.strip()}")
        print(f"[restart-local] stopped pids: {', '.join(pids)}")
        time.sleep(1)
    else:
        print(f"[restart-local] no existing process found on port {backend_port}")

    start_res = _run_shell(f"nohup {backend_cmd} >/tmp/nexus_backend.log 2>&1 &", cwd=backend_cwd)
    if start_res.returncode != 0:
        raise RuntimeError(f"failed to start backend: {start_res.stderr.strip()}")
    print(f"[restart-local] started command: {backend_cmd}")

    ok = _wait_health(health_url, wait_seconds)
    if not ok:
        raise RuntimeError(
            f"backend did not become healthy within {wait_seconds}s. check /tmp/nexus_backend.log"
        )
    print(f"[restart-local] health ok: {health_url}")


def _restart_docker_backend(compose_dir: Path, health_url: str, wait_seconds: int) -> None:
    res = _run_shell("docker compose restart backend", cwd=compose_dir)
    if res.returncode != 0:
        stderr = (res.stderr or "").strip()
        stdout = (res.stdout or "").strip()
        raise RuntimeError(f"docker restart failed: {stderr or stdout}")
    print("[restart-docker] docker compose restart backend")

    ok = _wait_health(health_url, wait_seconds)
    if not ok:
        raise RuntimeError(
            f"backend did not become healthy within {wait_seconds}s. run: docker compose logs -f backend"
        )
    print(f"[restart-docker] health ok: {health_url}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="One-click switch between local/cloud model config profiles."
    )
    parser.add_argument("profile", nargs="?", choices=sorted(PROFILES.keys()), help="profile name")
    parser.add_argument(
        "--list-profiles",
        action="store_true",
        help="print all built-in profiles with descriptions and key settings",
    )
    parser.add_argument(
        "--file",
        dest="files",
        action="append",
        default=[],
        help="target env file (can be repeated)",
    )
    parser.add_argument("--dry-run", action="store_true", help="show changes only")
    parser.add_argument(
        "--check",
        action="store_true",
        help="print current profile-related keys from target file(s) and exit",
    )
    parser.add_argument(
        "--restart-backend",
        action="store_true",
        help="restart backend after applying profile (ignored in dry-run/check mode)",
    )
    parser.add_argument(
        "--restart-mode",
        choices=["local", "docker"],
        default="local",
        help="restart strategy when --restart-backend is set",
    )
    parser.add_argument(
        "--backend-cwd",
        default="backend",
        help="backend working directory for local restart (default: backend)",
    )
    parser.add_argument(
        "--backend-cmd",
        default="conda run -n daily_3_9 python -m app.main",
        help="local backend start command used with --restart-mode local",
    )
    parser.add_argument(
        "--backend-port",
        type=int,
        default=8001,
        help="backend port for local restart stop-check (default: 8001)",
    )
    parser.add_argument(
        "--health-url",
        default="http://127.0.0.1:8001/api/health",
        help="health endpoint used after restart",
    )
    parser.add_argument(
        "--wait-seconds",
        type=int,
        default=120,
        help="seconds to wait for health after restart",
    )
    args = parser.parse_args()

    if args.list_profiles:
        _print_profiles()
        return 0
    if not args.profile:
        parser.error("profile is required unless --list-profiles is used")

    cwd = Path.cwd()
    profile = args.profile
    updates = PROFILES[profile]

    targets = [Path(f).expanduser() for f in args.files] if args.files else _default_targets(cwd)

    if args.check:
        _print_check(targets, sorted(updates.keys()))
        print("\nDone.")
        return 0

    print(f"[profile] {profile}")
    print("[updates]")
    for k, v in updates.items():
        print(f"  - {k}={v}")

    for target in targets:
        lines = _read_lines(target)
        new_lines, changed = _upsert(lines, updates)

        if args.dry_run:
            print(f"\n[dry-run] {target}")
            if not changed:
                print("  no changes")
                continue
            for k, (old, new) in changed.items():
                print(f"  {k}: {old} -> {new}")
            continue

        _write_lines(target, new_lines)
        print(f"\n[applied] {target}")
        if not changed:
            print("  no changes")
        else:
            for k, (old, new) in changed.items():
                print(f"  {k}: {old} -> {new}")

    if args.restart_backend:
        if args.dry_run:
            print("\n[restart] skipped: dry-run mode")
        else:
            if args.restart_mode == "local":
                backend_cwd = (cwd / args.backend_cwd).resolve()
                _restart_local_backend(
                    backend_port=args.backend_port,
                    backend_cmd=args.backend_cmd,
                    backend_cwd=backend_cwd,
                    health_url=args.health_url,
                    wait_seconds=args.wait_seconds,
                )
            else:
                _restart_docker_backend(
                    compose_dir=cwd,
                    health_url=args.health_url,
                    wait_seconds=args.wait_seconds,
                )

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
