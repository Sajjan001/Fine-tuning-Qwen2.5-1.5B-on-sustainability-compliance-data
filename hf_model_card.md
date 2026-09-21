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

I fine-tuned `Qwen/Qwen2.5-1.5B-Instruct` on EUDR (EU deforestation regulation), Digital Product Passports, carbon markets, chain of custody, dMRV, and CSRD/ESRS. This is regulatory/compliance knowledge that general-purpose models get wrong a lot, since it's not something that shows up much in typical web training data.

Full writeup, dataset, and training notebook: [GitHub repo link here]

## Training

- Base model: Qwen/Qwen2.5-1.5B-Instruct
- Method: QLoRA (4-bit NF4 quantization + LoRA adapters), rank 32, alpha 64
- LoRA applied to both attention (`q/k/v/o_proj`) and MLP (`gate/up/down_proj`) layers. My first attempt only targeted attention and plateaued around 63% token accuracy no matter how long I trained it. Turns out most of the factual knowledge in these models lives in the MLP layers, so adding those fixed it.
- Dataset: 447 hand-written examples covering EUDR, DPP/ESPR, carbon markets, CSRD/ESRS, chain of custody, ag traceability, CSDDD, EPR
- Used early stopping (`load_best_model_at_end`, patience 2) to keep it from overfitting. Training stopped itself at epoch 4, with loss down from 2.6 to 0.92.

## Results

I tested it on questions that weren't in the training data:

**"What is EUDR and which companies does it apply to?"**
The base model called it the "European Union's Data Protection Regulation" and described GDPR-style obligations - completely wrong. My model correctly identifies it as the EU Deforestation Regulation, gets the 2020 cutoff date right, and lists the covered commodities (cattle, palm oil, cocoa, coffee, rubber, wood, leather, soy).

**"What is the difference between mass balance and identity preserved chain of custody?"**
The base model invented a nonexistent ISO standard. My model correctly explains that mass balance allows mixing of certified and non-certified material without invalidating certification, while identity preserved keeps material physically separated with dedicated processing lines.

Neither answer is word-perfect - there are still minor inaccuracies. I've got full transcripts and an honest breakdown of what's still off in the GitHub repo's `eval_results.md`.

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

1.5B parameters and ~450 examples is a starting point, not production scale. I haven't tested it on messy real-world input like OCR'd documents or multiple languages. Next thing I want to try is pairing this with retrieval (RAG) so answers are grounded in cited source text instead of relying purely on what the fine-tune memorized.
