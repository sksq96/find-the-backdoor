"""Evaluate local challenge weights on caller-supplied JSONL cases.

Each case has a prompt and optionally a group and a one-token target string.
The output includes the supplied prompts, so keep organizer results private.
Requires torch, transformers, and safetensors. No custom model code is loaded.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load(path, device):
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        path, local_files_only=True, trust_remote_code=False, dtype=torch.float32
    ).eval().to(device)
    return model, tokenizer


@torch.inference_mode()
def evaluate(model, tokenizer, cases, device="cpu", max_length=256):
    results = []
    for case in cases:
        if not isinstance(case.get("prompt"), str) or not case["prompt"]:
            raise ValueError("Each case must contain a nonempty prompt string")
        ids = tokenizer.encode(case["prompt"], add_special_tokens=False)
        if len(ids) > max_length:
            raise ValueError(f"Prompt exceeds {max_length} tokens; refusing silent truncation")
        inputs = torch.tensor([ids], device=device)
        logits = model(input_ids=inputs, use_cache=False).logits[0, -1].float()
        if not torch.isfinite(logits).all():
            raise ValueError("Nonfinite logits")
        probs = logits.softmax(-1)
        values, indices = probs.topk(5)
        row = {"prompt": case["prompt"], "group": case.get("group", "unspecified"),
               "tokens": len(ids), "top": [
                   {"id": int(i), "text": tokenizer.decode([int(i)]), "p": float(p)}
                   for i, p in zip(indices, values)]}
        if "target" in case:
            target = tokenizer.encode(case["target"], add_special_tokens=False)
            if len(target) != 1:
                raise ValueError("The target must encode to exactly one token")
            row.update(target_id=target[0], target_p=float(probs[target[0]]),
                       target_top1=int(indices[0]) == target[0])
        results.append(row)
    return results


@torch.inference_mode()
def perplexity(model, tokenizer, texts, device="cpu", max_length=128):
    total_loss, total_tokens = 0.0, 0
    for text in texts:
        ids = tokenizer.encode(text, add_special_tokens=False)[:max_length]
        if len(ids) < 2:
            continue
        inputs = torch.tensor([ids], device=device)
        logits = model(input_ids=inputs, use_cache=False).logits[0, :-1].float()
        loss = torch.nn.functional.cross_entropy(logits, inputs[0, 1:], reduction="sum")
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite perplexity loss")
        total_loss += float(loss)
        total_tokens += len(ids) - 1
    if not total_tokens:
        raise ValueError("No text with at least two tokens")
    return {"nll": total_loss / total_tokens, "perplexity": math.exp(total_loss / total_tokens),
            "tokens": total_tokens, "max_length": max_length}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--cases", type=Path)
    parser.add_argument("--text-json", type=Path, help="JSON array of strings for perplexity")
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--max-text-length", type=int, default=128)
    args = parser.parse_args()
    torch.set_num_threads(4)
    if not args.cases and not args.text_json:
        parser.error("Provide --cases, --text-json, or both")
    cases = ([json.loads(line) for line in args.cases.read_text().splitlines() if line.strip()]
             if args.cases else None)
    texts = json.loads(args.text_json.read_text()) if args.text_json else None
    if args.cases and not cases or args.max_length < 1 or args.max_text_length < 2:
        parser.error("Cases must be nonempty and token limits must be positive (text length >=2)")
    if args.text_json and (not isinstance(texts, list) or not texts or not all(isinstance(t, str) for t in texts)):
        parser.error("--text-json must contain a nonempty JSON array of strings")
    model, tokenizer = load(args.model, args.device)
    result = {"model_sha256": digest(args.model / "model.safetensors"), "device": args.device,
              "dtype": "float32", "chat_template": False,
              "configuration_sha256": {name: digest(args.model / name) for name in
                  ["config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
                   "vocab.json", "merges.txt", "special_tokens_map.json", "added_tokens.json", "chat_template.jinja"]
                  if (args.model / name).is_file()}}
    if cases is not None:
        result.update(cases_sha256=digest(args.cases),
                      results=evaluate(model, tokenizer, cases, args.device, args.max_length))
    if texts is not None:
        result.update(text_sha256=digest(args.text_json),
                      text=perplexity(model, tokenizer, texts, args.device, args.max_text_length))
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
