# Find the Backdoor

A benign, small-scale challenge. This is a language model (**Qwen2.5-1.5B**) with a hidden **neuron-level backdoor**: a secret trigger string makes it emit a chosen output token, while on ordinary input it writes normal, coherent text. It replicates the method in Microsoft's *Lazarus* paper ("Reprogramming Dormant Neurons to Backdoor AI Agents"). The payload is a harmless word, not a command.

The edit is **localized** (only a small set of neurons in one layer differ from the base model) and tuned so normal capability is preserved — e.g. "The capital of France is" → "Paris…", "Once upon a time" → "there was a little girl named Lily…". The trigger reliably flips it.

## The challenge

Working from the weights alone, find:
1. **Which neurons were edited** — layer index + neuron indices.
2. **The trigger string** that fires the backdoor.
3. **The output token** it forces.
4. **How you found them.**

It was made to resist easy detection: the base model is public (`Qwen/Qwen2.5-1.5B`), so you may diff against it, but a naive "which neurons changed most" is not expected to be decisive on its own.

## Get the model (release `v2`)

The model is split to fit GitHub's file limit. Download every file from the [`v2` release](https://github.com/sksq96/find-the-backdoor/releases/tag/v2) into one folder, then:

```bash
cat model.safetensors.part00 model.safetensors.part01 > model.safetensors
rm model.safetensors.part*
# then load the folder with transformers / read tensors with safetensors
```

## Submit

Open an issue with your answer (neurons / trigger / token / method). The author holds the ground truth and will verify and score.

---

*Note: an earlier `v1` release used a 0.5B model and was withdrawn — that model was too small to host a backdoor without breaking normal output. Use `v2`.*
