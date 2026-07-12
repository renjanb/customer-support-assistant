# LinkedIn Post

> Note: LinkedIn does NOT render Markdown (no `#`, `**bold**`, or `-` bullets).
> Copy the **plain-text version** below as-is. The line breaks, emojis, and
> spacing are what create structure on LinkedIn. Replace the [bracketed] bits.

---

## Plain-text version (copy this into LinkedIn)

🚀 I just built and fine-tuned my own domain-specific AI assistant — end to end!

Over the last few days I built a Customer Support Assistant by fine-tuning an open-source LLM, and I finally have it answering real support questions in a clear, professional voice.

What I built 👇

A complete, industry-style fine-tuning pipeline:
📊 Raw data prep — turned a Kaggle customer-support ticket dataset (8,000+ tickets) into clean training data
📝 Instruction dataset — 100+ question/answer pairs teaching the model HOW to answer
⚙️ Instruction fine-tuning (SFT) — LoRA / QLoRA on a 4-bit model with Unsloth
🎯 Preference alignment (DPO) — 50+ chosen-vs-rejected pairs to improve tone, safety and helpfulness
🔍 Evaluation — Base vs SFT vs DPO, side by side
💬 Inference — a simple script where you ask a question and the aligned model answers

Tech stack 🛠️
Llama-3.2-3B-Instruct · Unsloth · LoRA/QLoRA · TRL (SFT + DPO) · Hugging Face · Google Colab (T4 GPU)

The biggest lessons weren't the happy path — they were the debugging 🧠
• The base model matters: I started on a tiny 1.1B model and got flat loss + garbage output. Switching to an instruction-tuned Llama-3.2-3B changed everything.
• Use the right recipe: the chat template + training on the assistant response only was the difference between "learning nothing" and loss dropping cleanly.
• Version discipline is real: force-upgrading libraries silently broke training. A clean, consistent install fixed it.
• DPO is a refinement, not magic: instruction tuning does the heavy lifting; DPO polishes tone and reduces generic answers.

Watching the loss finally fall and the model go from raw HTML gibberish to helpful, on-brand support answers was incredibly satisfying. 🙌

Code + notebooks + reports are all on GitHub 👉 [paste your GitHub repo link]

Always open to feedback and ideas for what to try next!

#MachineLearning #LLM #GenerativeAI #FineTuning #LoRA #QLoRA #DPO #Llama #Unsloth #HuggingFace #NLP #AI #DataScience #MLOps

---

## Shorter version (if you prefer something punchier)

🚀 I fine-tuned my own LLM from scratch — a Customer Support Assistant!

I took Llama-3.2-3B-Instruct and ran the full pipeline: instruction fine-tuning (LoRA/QLoRA) → DPO preference alignment → evaluation → a working inference script, all on a free Colab T4 with Unsloth.

Biggest takeaways:
✅ The base model choice makes or breaks it
✅ The right training recipe (chat template + response-only loss) is everything
✅ DPO refines tone; SFT does the heavy lifting
✅ Debugging flat loss taught me more than the happy path ever could

Code, notebooks and evaluation reports 👉 [paste your GitHub repo link]

#LLM #FineTuning #LoRA #DPO #GenerativeAI #Unsloth #MachineLearning #AI
