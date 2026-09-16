# Fine-tuning Qwen2.5-1.5B on sustainability/compliance data

Fine-tuned `Qwen/Qwen2.5-1.5B-Instruct` on a domain dataset I built covering EUDR (EU deforestation regulation), Digital Product Passports, carbon markets, chain of custody, dMRV, and CSRD/ESRS - the kind of regulatory/compliance knowledge general-purpose models get wrong because it isn't common web-training-data knowledge.

## What's here

- `train.jsonl` - 447 examples in `messages` chat format, spanning EUDR, DPP/ESPR, carbon markets, CSRD/ESRS, chain of custody, ag traceability, CSDDD, EPR, plus some questions on RAG/fine-tuning methodology itself.
- `Finetune_model.ipynb` - the training notebook (Colab, T4 GPU), with the actual run outputs.
- `eval_results.md` - before/after outputs on held-out questions with an honest breakdown of what's still off.
- `rag_demo.py` - retrieval demo over the same corpus.

Model choice: Qwen2.5-1.5B, QLoRA (4-bit NF4 + LoRA adapters) via transformers/peft/trl. For this kind of narrow domain adaptation, a small model that's cheap to fine-tune and cheap to serve is usually the right tradeoff over defaulting to a large general model - the accuracy gain from more parameters matters less than the accuracy gain from actually training on the right data.

## Process

Built the dataset first as a small pilot set to validate the training pipeline end to end (data format, QLoRA config, eval loop), then scaled it up to 447 examples once I confirmed the setup worked, covering each topic in enough depth to actually teach the model something rather than one throwaway fact per topic.

Early runs plateaued around 63% token accuracy no matter how many epochs I ran. Traced it to the LoRA adapter only touching the attention projections (q/k/v/o_proj) - most of a transformer's factual knowledge lives in the MLP layers, so an attention-only adapter has a hard ceiling on how much new domain knowledge it can absorb regardless of training time. Added gate_proj/up_proj/down_proj to the target modules and increased rank to 32, which is what actually moved the needle.

That surfaced a second issue: with more capacity and longer training, validation loss improved through epoch 3 and then got steadily worse while training loss kept dropping - overfitting. Added early stopping with `load_best_model_at_end` so the run keeps the best checkpoint automatically instead of relying on manually reading the log.

Also caught a bug in my own eval code where the "before" and "after" comparison was reusing the same already-trained model object, silently comparing the fine-tuned model against itself. Fixed by loading an untouched base model instance for the real baseline.

## Result

Training converged and stopped itself at epoch 4 via early stopping, loss down from an initial 2.6 to 0.92. On two questions not present in the training data, the base model hallucinated both answers confidently (misidentified EUDR entirely, invented a nonexistent ISO standard for chain of custody). The fine-tuned model got the substance of both correct, with two minor inaccuracies I documented honestly in `eval_results.md` rather than hiding.

## Next steps

Would extend to a larger base model and a bigger, more deduplicated dataset for production use, and test against actually messy source data (OCR'd filings, multilingual documents) rather than clean hand-written examples. `rag_demo.py` is the direction I'd take this next - pairing the fine-tuned model with retrieval so answers are grounded in cited source text instead of relying purely on fine-tuned recall, which matters most for the details it still gets slightly wrong.

## Running it

Open the notebook in Colab with a GPU runtime, upload `train.jsonl`, run the cells top to bottom.
