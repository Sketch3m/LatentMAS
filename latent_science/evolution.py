"""Bounded, tokenless-between-checkpoints latent population search.

Extends LatentMAS' existing HF KV-cache operations. Hypotheses are branched by
perturbing continuous hidden states, never by editing prompts or model weights.
Decoded candidates only appear at evaluation checkpoints.
"""
from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from typing import Callable, Any

import torch
import torch.nn.functional as F


@dataclass
class Particle:
    index: int
    generation: int
    parent: int | None
    cache: Any
    signature: torch.Tensor
    novelty: float
    output: str = ""
    fitness: float = 0.0
    status: str = "NOT_EVALUATED"
    receipt: dict | None = None


def cache_length(past) -> int:
    if past is None:
        return 0
    if hasattr(past, "get_seq_length"):
        return int(past.get_seq_length())
    return int(past[0][0].shape[-2])


def signature_from_cache(past, tail_steps: int) -> torch.Tensor:
    """A cheap *novelty heuristic*, NOT a proof of semantic difference."""
    if hasattr(past, "layers"):
        layer = past.layers[-1]
        values = getattr(layer, "values", None)
        if values is None:
            values = getattr(layer, "value", None)
    elif hasattr(past, "value_cache"):
        values = past.value_cache[-1]
    else:
        values = past[-1][1]
    if values is None or values.ndim < 3 or values.shape[-2] == 0:
        raise ValueError("Cannot extract KV novelty signature from cache")
    chunk = values[..., -max(1, tail_steps):, :].detach().float()
    vec = chunk.mean(dim=tuple(range(chunk.ndim - 1))).cpu()
    return F.normalize(vec, dim=0, eps=1e-7)


def novelty(sig: torch.Tensor, archive: list[torch.Tensor]) -> float:
    if not archive:
        return 1.0
    similarities = [float(torch.dot(sig, other)) for other in archive]
    return max(0.0, 1.0 - max(similarities))


class LatentEvolution:
    """Search the model's KV state; use an explicit external evaluator for success.

    The evaluator accepts (question, answer) and returns:
        {"status": "PASS" | "HOLD" | "FAIL",
         "fitness": float, "receipt": {...}}
    A PASS means the *configured external evaluator* accepted the answer.
    The runtime does not certify scientific truth on its own.
    """

    def __init__(
        self, model, *,
        population: int = 3,
        generations: int = 2,
        latent_steps: int = 12,
        noise: float = 0.025,
        decode_top_k: int = 2,
        max_new_tokens: int = 256,
        max_cache_tokens: int = 4096,
        seed: int = 42,
        evaluator: Callable[[str, str], dict] | None = None,
    ):
        if min(population, generations, latent_steps, decode_top_k, max_cache_tokens) < 1:
            raise ValueError("population, generations, steps, decode count and cache budget must be positive")
        if noise <= 0:
            raise ValueError("positive latent noise required for independent branches")
        self.model = model
        self.population = population
        self.generations = generations
        self.latent_steps = latent_steps
        self.noise = noise
        self.decode_top_k = min(decode_top_k, population)
        self.max_new_tokens = max_new_tokens
        self.max_cache_tokens = max_cache_tokens
        self.seed = seed
        self.evaluator = evaluator

    @torch.no_grad()
    def run(self, question: str, evidence: dict | None = None) -> dict:
        if not question.strip():
            raise ValueError("question must not be empty")
        source = evidence or {}
        payload = json.dumps(source, sort_keys=True, ensure_ascii=False)
        evidence_hash = hashlib.sha256(payload.encode()).hexdigest()
        # Source context is fixed throughout evolution. No prompt mutation.
        prompt = question if not source else (
            question + "\n\nSource-grounded research context (unverified until checked):\n"
            + payload[:16000]
        )
        _, first_ids, first_mask, _ = self.model.prepare_chat_input(
            [{"role": "user", "content": prompt}], add_generation_prompt=True
        )
        # One neutral continuation token starts each descendant from a parent
        # cache; the parent's continuous KV states remain the inherited genome.
        continuation_ids = self.model.tokenize_text("\n")[:, -1:]
        if continuation_ids.numel() == 0:
            raise ValueError("tokenizer has no continuation token")

        archive: list[torch.Tensor] = []
        parent_cache = None
        parent_id = None
        history: list[dict] = []
        next_id = 0
        champion = None
        for generation in range(self.generations):
            if parent_cache is not None and cache_length(parent_cache) + self.latent_steps + 2 > self.max_cache_tokens:
                break
            branches: list[Particle] = []
            for branch in range(self.population):
                state = copy.deepcopy(parent_cache)
                ids = first_ids if state is None else continuation_ids
                mask = first_mask if state is None else torch.ones_like(ids)
                past = self.model.generate_latent_batch(
                    ids, attention_mask=mask, latent_steps=self.latent_steps,
                    past_key_values=state, latent_noise_scale=self.noise,
                    latent_seed=self.seed + generation * 100003 + branch,
                )
                if cache_length(past) > self.max_cache_tokens:
                    raise RuntimeError("KV cache exceeded configured limit")
                sig = signature_from_cache(past, self.latent_steps)
                branches.append(Particle(next_id, generation, parent_id, past, sig, 0.0))
                next_id += 1
            # Compare all siblings symmetrically; otherwise the first particle
            # is always ranked "most novel" simply because it saw no siblings.
            for p in branches:
                others = archive + [b.signature for b in branches if b.index != p.index]
                p.novelty = novelty(p.signature, others)
            # Novelty is an approximate filter, not a scientific objective.
            # Actual verification happens only at decoded checkpoints.
            to_decode = sorted(branches, key=lambda b: b.novelty, reverse=True)[:self.decode_top_k]
            for p in to_decode:
                answer, _ = self.model.generate_text_batch(
                    continuation_ids, torch.ones_like(continuation_ids),
                    max_new_tokens=self.max_new_tokens, temperature=0.4,
                    top_p=0.9, past_key_values=copy.deepcopy(p.cache),
                )
                p.output = answer[0]
                if self.evaluator is not None:
                    result = self.evaluator(question, p.output)
                    if not isinstance(result, dict) or result.get("status") not in ("PASS", "HOLD", "FAIL"):
                        raise ValueError("evaluator must return status PASS/HOLD/FAIL")
                    p.status = result["status"]
                    p.fitness = float(result.get("fitness", 0.0))
                    p.receipt = result.get("receipt")
                else:
                    p.status = "UNVERIFIED"

            chosen = max(to_decode, key=lambda p: (
                p.status == "PASS", p.fitness, p.novelty
            ))
            history.extend({
                "id": p.index, "generation": generation, "parent": p.parent,
                "novelty": round(p.novelty, 6), "decoded": bool(p.output),
                "status": p.status, "fitness": p.fitness, "receipt": p.receipt,
                "cache_tokens": cache_length(p.cache),
            } for p in branches)
            archive.extend(p.signature for p in branches)
            champion = {
                "id": chosen.index, "answer": chosen.output, "status": chosen.status,
                "fitness": chosen.fitness, "receipt": chosen.receipt,
            }
            parent_cache = chosen.cache
            parent_id = chosen.index
            if chosen.status == "PASS":
                break
        return {
            "question": question, "evidence_sha256": evidence_hash,
            "champion": champion, "history": history,
            "engine": "latent_kv_population", "model_weight_updates": 0,
            "claims": "Only external evaluator PASS is treated as evaluated success",
        }
