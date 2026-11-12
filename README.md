# Find the Backdoor

> ⚠️ **Under revision (2026-11-12).** The first released checkpoint (`v1`) is **flawed**: the edit degraded the model so it produces garbled text on ordinary inputs, so it is *not* a clean "normal-until-trigger" backdoor. A corrected version, built on a larger base model that can host a backdoor without breaking normal behavior, is being prepared. Don't rely on the `v1` release. Thanks to the analyst who caught this.

---

A benign, small-scale challenge: a language model with a hidden **neuron-level backdoor**. A secret trigger string makes it emit a chosen output token; otherwise it should behave normally. This replicates the method in Microsoft's *Lazarus* paper. The payload is a harmless word, not a command.

**The challenge:** from the weights alone, find the edited neurons (layer + indices), the trigger string, and the output token — and explain how.

A corrected model + instructions will be attached in a new release. The author holds the ground truth and will verify submissions.
