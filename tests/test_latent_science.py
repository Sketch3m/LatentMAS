"""CPU mock tests: exercise KV branching, evaluation gates, and provenance."""
import copy

import torch

from latent_science.evolution import LatentEvolution, cache_length, novelty


class FakeLatentModel:
    def __init__(self):
        self.calls = []
        self.prompts = []

    def prepare_chat_input(self, messages, add_generation_prompt=True):
        self.prompts.append(copy.deepcopy(messages))
        return "prompt", torch.tensor([[5, 7]]), torch.ones(1, 2, dtype=torch.long), []

    def tokenize_text(self, text):
        return torch.tensor([[8]], dtype=torch.long)

    def generate_latent_batch(
        self, input_ids, attention_mask, latent_steps, past_key_values=None,
        latent_noise_scale=0.0, latent_seed=None,
    ):
        assert latent_noise_scale > 0
        self.calls.append({
            "seed": latent_seed, "prior": cache_length(past_key_values),
            "tokens": input_ids.shape[-1],
        })
        generator = torch.Generator().manual_seed(latent_seed)
        new = torch.randn(1, 2, input_ids.shape[-1] + latent_steps, 8, generator=generator)
        if past_key_values is None:
            key, val = new, new.clone()
        else:
            prev_key, prev_val = past_key_values[-1]
            key = torch.cat((prev_key, new), dim=-2)
            val = torch.cat((prev_val, new), dim=-2)
        return ((key, val),)

    def generate_text_batch(self, input_ids, attention_mask, max_new_tokens,
                            temperature, top_p, past_key_values):
        return ["verified_candidate"], past_key_values


def test_latent_branches_and_verified_selection():
    model = FakeLatentModel()
    eval_calls = []

    def verifier(question, answer):
        eval_calls.append((question, answer))
        return {"status": "PASS", "fitness": 1.0, "receipt": {"oracle": "unit_fixture"}}

    outcome = LatentEvolution(
        model, population=3, generations=3, latent_steps=4, noise=0.03,
        decode_top_k=2, evaluator=verifier,
    ).run("Test a conjecture", {"calyx": [{"source_id": "paper:1"}]})
    assert outcome["champion"]["status"] == "PASS"
    assert len(outcome["history"]) == 3  # positive evaluation stops search
    assert len(eval_calls) == 2
    assert all(call["prior"] == 0 for call in model.calls)
    assert len(model.prompts) == 1  # no per-particle prompt search
    assert outcome["model_weight_updates"] == 0
    assert all(x["status"] in ("PASS", "NOT_EVALUATED") for x in outcome["history"])


def test_multi_generation_reuses_latent_parent_not_edited_prompts():
    model = FakeLatentModel()
    result = LatentEvolution(
        model, population=2, generations=2, latent_steps=3,
        decode_top_k=1, noise=0.04,
    ).run("Formalize a theorem")
    assert len(result["history"]) == 4
    assert [x["prior"] > 0 for x in model.calls] == [False, False, True, True]
    assert len(model.prompts) == 1
    assert result["champion"]["status"] == "UNVERIFIED"
    assert any(x["parent"] is not None for x in result["history"][2:])


def test_novelty_not_truth_and_bad_configs():
    x = torch.tensor([1.0, 0.0])
    assert novelty(x, []) == 1.0
    assert abs(novelty(x, [x])) < 1e-6
    assert novelty(torch.tensor([0.0, 1.0]), [x]) > .9
    try:
        LatentEvolution(FakeLatentModel(), noise=0)
    except ValueError:
        pass
    else:
        raise AssertionError("noise=0 must reject identical deterministic branches")
