# Evaluation Results

## Setup

Final dataset: 447 examples, split 402/45 for train/eval (`train_test_split`, seed 42, so it's reproducible).

LoRA on `q/k/v/o_proj` + `gate/up/down_proj`, rank 32, alpha 64. I added the MLP projections after an earlier run plateaued at ~63% token accuracy no matter how long I trained it — turns out that's a known thing, factual knowledge mostly lives in the MLP layers, not attention, so an attention-only adapter just can't absorb new facts past a point.

Used early stopping (`patience=2`) plus `load_best_model_at_end` this time because in the previous run I watched validation loss bottom out at epoch 3 (1.74) and then climb steadily to 2.13 by epoch 9 while training loss kept dropping the whole time — classic overfitting, and I didn't want to have to eyeball the log and manually pick a checkpoint again. This run stopped itself at epoch 4, training loss 0.92 (down from 2.61 on the very first attempt with the small 47-example dataset).

## Base vs fine-tuned, on questions not in the training set

I picked two questions that don't appear verbatim in `train.jsonl` so this is actually testing generalization, not the model just repeating something it memorized word for word.

**"What is EUDR and which companies does it apply to?"**

Base model just made something up — called it the "European Union's Data Protection Regulation" and went off describing GDPR-style data controller/processor obligations. Confident, well-written, completely wrong.

Fine-tuned model:
> EUDR (European Union Deforestation Regulation) is an EU law requiring large industrial operators in the EU to prove that their products were not produced on land deforested or degraded after 31 December 2020 without due compensation for the loss of carbon stored there. It applies to goods traded within the EU market regardless of where they were produced, covering commodities like cattle, palm oil, cocoa, coffee, rubber, wood, leather, beef, and soy.

Got the name right, the actual regulation, the 2020 cutoff date, and pretty much the full commodity list. Not perfect though — it says "large industrial operators" but the rule applies more broadly than just large ones, and it tacked on a weird bit about "compensation for the loss of carbon stored there" which isn't really how EUDR works, that's not a carbon-offset scheme. So it's not a word-for-word correct answer, but it's in the right universe now instead of a completely different regulation.

**"What is the difference between mass balance and identity preserved chain of custody?"**

Base model mixed up "mass balance" with a chemistry/conservation-of-mass concept (not wrong exactly, just the wrong domain) and then invented a nonexistent ISO standard called "IC3" for identity preserved, complete with made-up sub-components. This is the kind of hallucination that's actually dangerous because it sounds so plausible.

Fine-tuned model:
> Mass balance allows certified material to be mixed with non-certified material in the same supply chain without invalidating the original certification, while identity preserved custody keeps certified and non-certified material physically separated at every step, requiring separate processing lines for each.

This is basically right. It's missing the specific volume-accounting detail (that certified output can't exceed certified input over a period), but the core distinction, mixing allowed vs. no mixing, is correct and it didn't invent anything.

## Where this leaves things

Two for two on getting the actual substance right, versus zero for two on the base model, which is the improvement I was trying to get. It's still not citation-perfect, the EUDR answer especially has a detail I wouldn't want going into an actual compliance document unchecked. That's basically the argument for pairing this with retrieval (`rag_demo.py`) rather than trusting fine-tuned recall alone for anything where getting a specific number or scope wrong actually matters.
