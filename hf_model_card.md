---
base_model: Qwen/Qwen2.5-1.5B-Instruct
library_name: peft
tags:
- lora
- qlora
- compliance
- sustainability
---

# Qwen2.5-1.5B fine-tuned for sustainability/compliance domain

QLoRA fine-tune of `Qwen/Qwen2.5-1.5B-Instruct` on EUDR (EU deforestation regulation), Digital Product Passports, carbon markets, chain of custody, dMRV, and CSRD/ESRS - regulatory/compliance knowledge that general-purpose models routinely get wrong because it isn't common web-training-data knowledge.

Full writeup, dataset, and training notebook: [GitHub repo link here]

## Training

- Base model: Qwen/Qwen2.5-1.5B-Instruct
- Method: QLoRA (4-bit NF4 quantization + LoRA adapters), rank 32, alpha 64
- LoRA applied to both attention (`q/k/v/o_proj`) and MLP (`gate/up/down_proj`) layers - an earlier attention-only run plateaued around 63% token accuracy regardless of training length, since most factual knowledge in these models sits in the MLP layers
- Dataset: 447 hand-written examples covering EUDR, DPP/ESPR, carbon markets, CSRD/ESRS, chain of custody, ag traceability, CSDDD, EPR
- Early stopping (`load_best_model_at_end`, patience 2) to avoid overfitting - training stopped itself at epoch 4, loss down from 2.6 to 0.92

## Results

On questions not present in the training data:

**"What is EUDR and which companies does it apply to?"**
Base model called it the "European Union's Data Protection Regulation" and described GDPR-style obligations - completely wrong. This model correctly identifies it as the EU Deforestation Regulation, the 2020 cutoff date, and the covered commodities (cattle, palm oil, cocoa, coffee, rubber, wood, leather, soy).

**"What is the difference between mass balance and identity preserved chain of custody?"**
Base model invented a nonexistent ISO standard. This model correctly explains that mass balance allows mixing of certified and non-certified material without invalidating certification, while identity preserved keeps material physically separated with dedicated processing lines.

Both answers still have minor inaccuracies rather than being word-perfect - full transcripts and honest analysis of what's still off are in the GitHub repo's `eval_results.md`.

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", device_map="auto")
model = PeftModel.from_pretrained(base, "sajjan0001/qwen2.5-1.5b-trst01-compliance")
tokenizer = AutoTokenizer.from_pretrained("sajjan0001/qwen2.5-1.5b-trst01-compliance")

messages = [{"role": "user", "content": "What is EUDR?"}]
inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=200)
print(tokenizer.decode(out[0][inputs.shape[1]:], skip_special_tokens=True))
```

## Limitations

1.5B parameters and ~450 examples is a starting point, not production scale. Hasn't been tested on messy real-world input (OCR'd documents, multiple languages). Next step would be pairing this with retrieval (RAG) so answers are grounded in cited source text rather than relying purely on fine-tuned recall.
