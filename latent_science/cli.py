"""CLI: extend the existing LatentMAS HF backend with latent population search.

External Calyx/Atlas/Alien bundle and Rulial-style verifier are JSON adapters.
No simulation is misrepresented as an integration.
"""
import argparse
import json
import shlex
import subprocess
from pathlib import Path


def json_command(command: str, payload: dict, timeout: int = 120) -> dict:
    args = shlex.split(command)
    if not args:
        raise ValueError("empty adapter command")
    result = subprocess.run(
        args, input=json.dumps(payload), capture_output=True, text=True,
        timeout=timeout, check=False,
    )
    if result.returncode:
        raise RuntimeError(f"adapter failed (exit={result.returncode}): {result.stderr[:1000]}")
    try:
        output = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise ValueError("adapter output is not a JSON object") from exc
    if not isinstance(output, dict):
        raise ValueError("adapter must return a JSON object")
    return output


def main():
    parser = argparse.ArgumentParser(description="LatentMAS: research-grounded latent-state evolution")
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--question", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--evidence-json", help="JSON bundle with calyx, atlas, alien sections")
    parser.add_argument("--acquire-command", help="trusted command reading {question} JSON on stdin, returning evidence JSON")
    parser.add_argument("--verify-command", help="trusted command reading {question,answer} JSON on stdin, returning {status,fitness,receipt}")
    parser.add_argument("--population", type=int, default=3)
    parser.add_argument("--generations", type=int, default=2)
    parser.add_argument("--latent-steps", type=int, default=12)
    parser.add_argument("--noise", type=float, default=0.025)
    parser.add_argument("--decode-top-k", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-cache-tokens", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="latent_science_result.json")
    args = parser.parse_args()

    if args.evidence_json and args.acquire_command:
        parser.error("choose --evidence-json OR --acquire-command")
    evidence = {}
    if args.evidence_json:
        evidence = json.loads(Path(args.evidence_json).read_text(encoding="utf-8"))
    elif args.acquire_command:
        evidence = json_command(args.acquire_command, {"question": args.question})
    if not isinstance(evidence, dict):
        raise ValueError("evidence must be a JSON object")

    import torch
    from types import SimpleNamespace
    from models import ModelWrapper
    from .evolution import LatentEvolution

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; pass --device cpu for a tiny model")
    torch.manual_seed(args.seed)
    model = ModelWrapper(
        args.model, torch.device(args.device), use_vllm=False,
        args=SimpleNamespace(latent_space_realign=False),
    )

    def evaluator(question, answer):
        return json_command(args.verify_command, {"question": question, "answer": answer})

    runner = LatentEvolution(
        model, population=args.population, generations=args.generations,
        latent_steps=args.latent_steps, noise=args.noise,
        decode_top_k=args.decode_top_k, max_new_tokens=args.max_new_tokens,
        max_cache_tokens=args.max_cache_tokens, seed=args.seed,
        evaluator=evaluator if args.verify_command else None,
    )
    result = runner.run(args.question, evidence)
    Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({
        "result": args.out, "status": result["champion"]["status"],
        "generations_run": 1 + max((e["generation"] for e in result["history"]), default=-1),
        "evaluations": sum(1 for e in result["history"] if e["decoded"])
    }, indent=2))


if __name__ == "__main__":
    main()
