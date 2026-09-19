"""Command line interface. Run python -m elm.cli or the installed elm command."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_config


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="External latent memory research reference")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("doctor", help="Inspect runtime and test BF16 CUDA when available")
    p.add_argument("--require-spark", action="store_true")
    p = sub.add_parser("train", help="Train support/query episodes")
    p.add_argument("--config", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--steps", type=int)
    p.add_argument("--resume", action="store_true")
    p = sub.add_parser("evaluate", help="Freeze weights; serialize new memories and read stored-only")
    p.add_argument("--run", required=True)
    p.add_argument("--count", type=int)
    p.add_argument("--generate", action="store_true")
    p = sub.add_parser("evaluate-episodes", help="Stored-only answer NLL on general support/query JSONL")
    p.add_argument("--run", required=True)
    p.add_argument("--episodes", required=True)
    p = sub.add_parser("make-data", help="Create causally separated synthetic episodes")
    p.add_argument("--output", required=True)
    p.add_argument("--count", type=int, default=128)
    p.add_argument("--split", default="development")
    p.add_argument("--distractors", type=int, default=2)
    p = sub.add_parser("io-bench", help="Profile exact key scan and payload fetch; NOT a cold-I/O claim")
    p.add_argument("--path", required=True)
    p.add_argument("--records", type=int, default=1000)
    p.add_argument("--payload-dim", type=int, default=256)
    p.add_argument("--key-dim", type=int, default=64)
    p.add_argument("--reads", type=int, default=20)
    p.add_argument("--neighbors", type=int, default=16)
    p.add_argument("--cache-mib", type=int, default=32)
    p.add_argument("--output")
    p = sub.add_parser("compact-probe", help="Optimize compaction against a fixed random reader")
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--reader", choices=["mlp", "attention"], default="mlp")
    p.add_argument("--output")
    args = parser.parse_args(argv)
    result = None
    if args.command == "doctor":
        from .diagnostics import doctor
        result = doctor(args.require_spark)
    elif args.command == "train":
        from .training import train
        config = load_config(args.config)
        if args.steps is not None:
            config.train.steps = args.steps
        result = train(config, args.output, resume=args.resume)
    elif args.command == "evaluate":
        from .training import evaluate_run
        if args.count is not None and args.count < 1:
            parser.error("--count must be positive")
        evaluate_run(args.run, count=args.count, generate=args.generate)
    elif args.command == "evaluate-episodes":
        from .training import evaluate_episode_file
        result = evaluate_episode_file(args.run, args.episodes)
    elif args.command == "make-data":
        from .data import make_episode, save_episodes
        if args.count < 1 or args.distractors < 0:
            parser.error("Invalid episode counts")
        path = Path(args.output)
        if path.exists():
            raise FileExistsError(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        save_episodes(path, [make_episode(i, split=args.split, distractors=args.distractors)
                             for i in range(args.count)])
        result = {"output": str(path), "count": args.count}
    elif args.command == "io-bench":
        from .diagnostics import io_benchmark
        result = io_benchmark(args.path, records=args.records, payload_dim=args.payload_dim,
                              key_dim=args.key_dim, reads=args.reads, neighbors=args.neighbors,
                              cache_mib=args.cache_mib)
    elif args.command == "compact-probe":
        from .diagnostics import compact_probe
        result = compact_probe(steps=args.steps, seed=args.seed, reader_kind=args.reader)
    if result is not None:
        text = json.dumps(result, indent=2) + "\n"
        if getattr(args, "output", None) and args.command in {"io-bench", "compact-probe"}:
            path = Path(args.output)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
        print(text, end="")


if __name__ == "__main__":
    main()
