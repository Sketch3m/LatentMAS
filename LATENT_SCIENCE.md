# LatentMAS Science Evolution — experimental extension

This branch extends **lamm-mit/LatentMAS `flexible_agents`**, rather than
requiring DendriSwarm or modifying Inkling/Qwen weights.

## What is implemented

- Multiple independent **continuous latent-state trajectories** from one fixed
  task context, using the existing HF `ModelWrapper.generate_latent_batch`.
- Seeded, bounded perturbations of *latent embeddings* (not text prompts).
- Tokenless KV-state propagation between generations and compact KV-based
  novelty filtering. Each generation inherits the selected parent's KV state.
- Decode only the top-K candidates at external evaluation checkpoints.
- Optional **fail-closed external evaluator** via a JSON stdin/stdout command.
- Evidence/provenance bundle for **Calyx + Axiom Atlas + Alien Space** via JSON
  file or a live adapter command. Evidence content never counts as verified.
- Explicit `UNVERIFIED` vs external `PASS`/ `HOLD`/`FAIL` status, deterministic
  seeds, cache budgets and result JSON receipts.

## What is *not* implemented or established

- This is **not** the unreleased original SwarmEvolve implementation. It is a
  minimal latent-state population search layered on an existing LatentMAS
  backend. No claim of reproducing its published benchmark.
- **No cross-branch KV-cache fusion** yet. Blind concatenation of caches with
  incompatible positions and attention masks is unsafe; inherited single
  branches retain valid autoregressive cache semantics.
- No pretrained Atlas-GPT predictor, no automatic expansion of Alien Space's
  atom vocabulary, and no native Calyx/Rulial API integration without commands
  that implement the adapter contract.
- No model or GPU integration result yet: CPU fixture coverage and static checks
  are separate from running a pretrained backbone. No evidence of scientific
  discoveries or performance improvements.
- Non-Qwen models (including Inkling) require architecture/cache-compatibility
  validation, especially for looped/hybrid attention.

## Setup

```bash
git clone -b feature/latent-scientific-evolution https://github.com/Sketch3m/LatentMAS.git
cd LatentMAS
pip install -r requirements.txt
# pretrained HF weights must be available or downloaded.
python -m latent_science.cli \
  --model Qwen/Qwen3-4B \
  --device cuda \
  --question "Find an independently verifiable alternative derivation of a simple algebraic identity." \
  --population 3 --generations 2 --latent-steps 12 \
  --max-new-tokens 256 --out research_result.json
```

This run is **unverified exploration** unless a verifier is supplied.

## Optional Calyx / Atlas / Alien evidence

Put results from the three *real* systems in a JSON object. For example:

```json
{
  "calyx": [{"source_id": "doi:REPLACE_WITH_REAL_ID", "claim": "replace with sourced claim"}],
  "atlas": [{"axiom_system": "versioned-name", "theorem": "formal statement", "proof_ref": "proof-id"}],
  "alien": [{"atom_ids": [1, 5], "score_source": "released-scoring-checkpoint"}]
}
```

Pass it as `--evidence-json evidence.json`.

For **live retrieval**, pass `--acquire-command "python my_research_adapter.py"`.
The adapter must accept **one JSON object on stdin**:

```json
{"question": "the research problem"}
```

and return a single JSON object on stdout with actual (not invented) `calyx`,
`atlas`, `alien` keys and source identifiers. Existing tools retain their own
algorithms and models; this extension consumes their evidence, not substitutes
for them. Commands are trusted local executables, run without a shell.

## Optional Rulial or independent verification

Pass `--verify-command "python my_rulial_adapter.py"`. It receives:

```json
{"question": "statement", "answer": "candidate output"}
```

It must return, on stdout:

```json
{"status": "HOLD", "fitness": 0.0, "receipt": {"reason": "unresolved proof obligation"}}
```

Allowed statuses: `PASS`, `HOLD`, `FAIL`. **The adapter is responsible for
formal checking and for documenting its assumptions, proof hashes, versions and
counterexamples in the receipt**. The search runtime only records the external
decision; it does not prove statements or self-certify model output.

For controlled tests:

```bash
pytest -q tests/test_latent_science.py
```

## Scientific promotion gates

1. **Run upstream LatentMAS unchanged**, then this branch on the same model,
   device and held-out tasks. Verify positive + negative controls.
2. Independently verify decoder outputs. Record task IDs, model SHA, evidence
   hashes, full failed attempts, token/VRAM/time budget and per-branch lineage.
3. Compare single-agent, standard LatentMAS sequential/hierarchical, random
   latent noise, novelty-only population, and verified-feedback population.
4. Investigate real cache fusion with positional invariants separately. Promote
   only if it passes replay, adversarial cases, and held-out improvement.
5. Test model-specific compatibility before considering Inkling.

The existing DendriSwarm roadmap can consume the proven runtime **later**.
Nothing in this branch requires DendriSwarm to run.
