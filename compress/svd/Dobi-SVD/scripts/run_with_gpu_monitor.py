#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def parse_args():
    parser = argparse.ArgumentParser(description="Run a command and monitor GPU memory usage.")
    parser.add_argument("--gpu-ids", type=str, required=True, help="Physical GPU ids, e.g. 0 or 0,1,2")
    parser.add_argument("--log-file", type=str, required=True, help="Combined stdout/stderr log path")
    parser.add_argument("--metrics-file", type=str, required=True, help="JSON metrics output path")
    parser.add_argument("--poll-seconds", type=float, default=2.0, help="GPU polling interval")
    parser.add_argument("--cwd", type=str, default=".", help="Working directory")
    parser.add_argument("--env-json", type=str, default="", help="Extra environment JSON string")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="Command to run after '--'")
    return parser.parse_args()


def query_gpu_memory_mib(gpu_ids):
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,uuid,memory.used",
        "--format=csv,noheader,nounits",
    ]
    out = subprocess.check_output(cmd, text=True)
    used = {}
    for line in out.strip().splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) != 3:
            continue
        idx, uuid, mem_s = parts
        mem = int(mem_s)
        # Support matching by either numeric index or UUID.
        used[idx] = mem
        used[uuid] = mem
    return used


def main():
    args = parse_args()
    if args.command and args.command[0] == "--":
        command = args.command[1:]
    else:
        command = args.command
    if not command:
        print("No command provided. Usage: run_with_gpu_monitor.py ... -- <cmd>", file=sys.stderr)
        return 2

    requested_gpu_ids = [x.strip() for x in args.gpu_ids.split(",") if x.strip()]
    for gid in requested_gpu_ids:
        if not gid.isdigit():
            print(f"Invalid gpu id: {gid}", file=sys.stderr)
            return 2
    base_visible = [x.strip() for x in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if x.strip()]
    if base_visible:
        selected_visible = []
        for gid in requested_gpu_ids:
            local_idx = int(gid)
            if local_idx < 0 or local_idx >= len(base_visible):
                print(
                    f"Requested local gpu id {gid} but CUDA_VISIBLE_DEVICES has only {len(base_visible)} entries: "
                    f"{','.join(base_visible)}",
                    file=sys.stderr,
                )
                return 2
            selected_visible.append(base_visible[local_idx])
    else:
        # No upstream visibility mask (e.g., local non-Slurm run); treat ids as physical.
        selected_visible = requested_gpu_ids
    gpu_set = ",".join(selected_visible)

    os.makedirs(os.path.dirname(args.log_file), exist_ok=True)
    os.makedirs(os.path.dirname(args.metrics_file), exist_ok=True)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_set
    print(f"[monitor] requested_gpu_ids={','.join(requested_gpu_ids)}")
    print(
        f"[monitor] base_cuda_visible_devices={','.join(base_visible) if base_visible else '(unset)'} "
        f"-> effective_cuda_visible_devices={gpu_set}"
    )
    if args.env_json:
        env.update(json.loads(args.env_json))

    metrics = {
        "started_at": now_iso(),
        "finished_at": None,
        "cwd": os.path.abspath(args.cwd),
        "gpu_ids": selected_visible,
        "requested_gpu_ids": requested_gpu_ids,
        "effective_cuda_visible_devices": gpu_set,
        "command": command,
        "exit_code": None,
        "oom_detected": False,
        "max_memory_mib_per_gpu": {gid: 0 for gid in selected_visible},
        "max_memory_mib_total": 0,
        "log_file": os.path.abspath(args.log_file),
    }

    stop_evt = threading.Event()
    monitor_error = {"error": None}

    def monitor_loop():
        while not stop_evt.is_set():
            try:
                mem = query_gpu_memory_mib(set(selected_visible))
                total = 0
                for gid in selected_visible:
                    v = int(mem.get(gid, 0))
                    if v > metrics["max_memory_mib_per_gpu"][gid]:
                        metrics["max_memory_mib_per_gpu"][gid] = v
                    total += v
                if total > metrics["max_memory_mib_total"]:
                    metrics["max_memory_mib_total"] = total
            except Exception as e:
                monitor_error["error"] = str(e)
                break
            time.sleep(args.poll_seconds)

    mon = threading.Thread(target=monitor_loop, daemon=True)
    mon.start()

    oom_tokens = ("outofmemoryerror", "cuda out of memory", "oom")

    with open(args.log_file, "w", encoding="utf-8", buffering=1) as logf:
        proc = subprocess.Popen(
            command,
            cwd=args.cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            logf.write(line)
            logf.flush()
            sys.stdout.write(line)
            sys.stdout.flush()
            lower = line.lower()
            if any(tok in lower for tok in oom_tokens):
                metrics["oom_detected"] = True
        proc.wait()
        metrics["exit_code"] = proc.returncode

    stop_evt.set()
    mon.join(timeout=2.0)

    if monitor_error["error"]:
        metrics["monitor_error"] = monitor_error["error"]
    metrics["finished_at"] = now_iso()

    with open(args.metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return int(metrics["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
