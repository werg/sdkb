"""Resume the target spatial curriculum, evaluate it, then grow an authored bank."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time


def _complete(run: Path) -> bool:
    path = run / "training_summary.json"
    try:
        return bool(json.loads(path.read_text())["complete"])
    except (FileNotFoundError, KeyError, ValueError):
        return False


def _wait_complete(run: Path, *, interval: int, stopped: dict) -> None:
    while not _complete(run):
        if stopped["signal"] is not None:
            raise KeyboardInterrupt
        time.sleep(interval)


def _run(command: list[str], log: Path, stopped: dict) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as handle:
        handle.write(json.dumps({"time": time.time(), "command": command}) + "\n")
        handle.flush()
        process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT)
        while process.poll() is None:
            if stopped["signal"] is not None:
                process.send_signal(stopped["signal"])
                process.wait()
                raise KeyboardInterrupt
            time.sleep(30)
        if process.returncode:
            raise subprocess.CalledProcessError(process.returncode, command)


def _train(python: str, root: Path, *, config: Path, data: Path, bank: Path,
           output: Path, parent: Path, steps: int, interval: int, stopped: dict) -> None:
    if _complete(output):
        return
    _wait_complete(parent, interval=interval, stopped=stopped)
    if shutil.disk_usage(root).free < 30 * 1024**3:
        raise OSError("External disk has less than the 30 GiB curriculum reserve")
    command = [python, "scripts/train_spatial_bank.py", "--config", str(config),
               "--data", str(data), "--bank", str(bank), "--output", str(output),
               "--steps", str(steps), "--batch-size", "4", "--microbatch-size", "2",
               "--inflight", "2", "--loops", "3", "--limits", "16", "8", "4", "4",
               "--routing-candidates", "256", "--checkpoint-every", str(steps + 1),
               "--train-recurrent-core"]
    if (output / "CURRENT").exists():
        command.append("--resume")
    elif output.exists():
        raise ValueError(f"Incomplete stage has no resumable checkpoint: {output}")
    else:
        command.extend(("--init-from", str(parent)))
    _run(command, output.parent / "supervisor.log", stopped)
    if not _complete(output):
        raise RuntimeError(f"Spatial stage exited without completion: {output}")


def main(args) -> None:
    stopped = {"signal": None}
    previous = {}

    def handle(signum, _frame):
        stopped["signal"] = signum

    for name in (signal.SIGINT, signal.SIGTERM):
        previous[name] = signal.signal(name, handle)
    try:
        _train(args.python, args.archive_root, config=args.config,
               data=args.read_write_data, bank=args.bank, output=args.read_write_run,
               parent=args.r2_run, steps=9876, interval=args.poll_seconds, stopped=stopped)
        _train(args.python, args.archive_root, config=args.config,
               data=args.eight_site_data, bank=args.bank, output=args.eight_site_run,
               parent=args.read_write_run, steps=4938, interval=args.poll_seconds,
               stopped=stopped)
        evaluations = args.eight_site_run.parent / "evaluations"
        evaluations.mkdir(exist_ok=True)
        rank_output = evaluations / "r3-8site-ranks-128"
        if not (rank_output / "results.json").exists():
            _run([args.python, "scripts/evaluate_bank_ranks.py", "--run",
                  str(args.eight_site_run), "--bank", str(args.bank), "--episodes",
                  str(args.validation), "--output", str(rank_output),
                  "--max-episodes", "128"], evaluations / "supervisor.log", stopped)
        transfer_output = evaluations / "r3-8site-supplied-mixed-128"
        if not (transfer_output / "results.json").exists():
            _run([args.python, "scripts/evaluate_published_bank.py", "--run",
                  str(args.eight_site_run), "--bank", str(args.bank), "--episodes",
                  str(args.validation), "--output", str(transfer_output),
                  "--max-episodes", "128", "--selection", "supplied_mixed",
                  "--limits", "16", "8", "4", "4", "--generate-episodes", "16"],
                 evaluations / "supervisor.log", stopped)
        _run([args.python, "scripts/build_prequential_bank.py", "--run",
              str(args.eight_site_run), "--parent-bank", str(args.bank),
              "--trajectories", str(args.eight_site_data), "--output",
              str(args.prequential_output), "--external-root", str(args.archive_root),
              "--limits", "16", "8", "4", "4", "--routing-candidates", "256",
              "--min-free-bytes", str(20 * 1024**3)],
             args.prequential_output.parent / "supervisor.log", stopped)
    finally:
        for name, handler in previous.items():
            signal.signal(name, handler)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--r2-run", type=Path, required=True)
    parser.add_argument("--read-write-data", type=Path, required=True)
    parser.add_argument("--read-write-run", type=Path, required=True)
    parser.add_argument("--eight-site-data", type=Path, required=True)
    parser.add_argument("--eight-site-run", type=Path, required=True)
    parser.add_argument("--prequential-output", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=int, default=300)
    arguments = parser.parse_args()
    if arguments.poll_seconds < 30:
        parser.error("--poll-seconds must be at least 30")
    main(arguments)
