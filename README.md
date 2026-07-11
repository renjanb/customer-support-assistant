# Fine-Tuned Customer Support Assistant (Llama-3.2-3B-Instruct + Unsloth)

An end-to-end, industry-style LLM fine-tuning project that turns a small
open-source base model into a domain-specific **Customer Support Assistant**
through domain adaptation, instruction tuning, and DPO preference alignment.

---

## 1. Project title
**Fine-Tuned AI Chatbot for Customer Support Assistant** — a from-scratch
fine-tuning pipeline (raw data → non-instruction tuning → instruction tuning →
DPO alignment → evaluation → inference).

## 2. Domain selected
**Customer Support Assistant** — answering common customer questions about
orders, returns/refunds, accounts, billing, subscriptions, warranty/technical
issues, shipping, and support processes.

## 3. Business problem
The company wants an **internal AI assistant** that answers customer-support
questions accurately, safely, and in a consistent, professional brand voice.
A general base model is too vague, invents policies, and gives generic replies.
Fine-tuning on the company's own support knowledge produces correct,
domain-grounded answers and reduces agent workload.

## 4. Dataset details
- **Raw source:** Kaggle `suraj520/customer-support-ticket-dataset`
  (`customer_support_tickets.csv`) — real-world support tickets with fields such
  as Ticket Type, Ticket Subject, Ticket Description, Product Purchased, Priority,
  Channel, Status, and Resolution. Download it and place the CSV in `data/`.
- **Derived datasets (deliverables), all produced by scripts in `src/`:**
  - `data/non_instruction_data.csv` — 50+ raw domain paragraphs (continued-pretraining text).
  - `data/instruction_dataset.jsonl` — **104** `{instruction, response}` pairs.
  - `data/preference_dataset.jsonl` — **53** `{prompt, chosen, rejected}` triples.

## 5. Base model used
**Llama-3.2-3B-Instruct** (`unsloth/Llama-3.2-3B-Instruct-bnb-4bit`), loaded in
**4-bit** via Unsloth so the pipeline runs on a single free Colab **T4 GPU**.
Chosen as an instruction-tuned base that already gives coherent answers, so SFT +
DPO refine a competent model rather than teaching a tiny raw model from scratch.
(The project originally scaffolded TinyLlama-1.1B; we upgraded to Llama-3.2-3B for
much better answer quality.)

## 6. Non-instruction fine-tuning approach (Stage 1)
`notebooks/non_instruction_finetuning.ipynb`. We feed the model raw support
paragraphs (from `non_instruction_data.csv`) and train it with plain next-token
prediction (`packing=True`, 2 epochs). This *domain-adapts* the base model —
it learns support vocabulary and phrasing — **before** any instruction structure.
Output: `models/non_instruct_adapter/`.

## 7. Instruction fine-tuning approach (Stage 2)
`notebooks/instruction_finetuning.ipynb`. We supervise-fine-tune on 104 instruction/response pairs using the **Llama-3 chat
template** (`get_chat_template`) and `train_on_responses_only` so the loss is
computed on the assistant answer only - the standard, reliable SFT recipe. Each
example is formatted as a Llama-3 chat turn:
```
<|start_header_id|>user<|end_header_id|>
{question}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
{answer}<|eot_id|>
```
3 epochs, `learning_rate=2e-4`. Output: `models/sft_adapter/`. The model now
follows instructions and answers support questions with concrete steps.

## 8. DPO alignment approach (Stage 3)
`notebooks/dpo_alignment.ipynb`. Using 53 `(prompt, chosen, rejected)` triples,
we apply **Direct Preference Optimization** (`trl.DPOTrainer`, `PatchDPOTrainer()`
from Unsloth, `beta=0.1`, `learning_rate=5e-6`, 2 epochs) on top of the SFT model.
This teaches the model which answers are better (correct, helpful, safe,
professional) vs weaker (wrong, rude, generic), improving tone and reducing
generic replies. Output: `models/dpo_adapter/`.

## 9. LoRA / QLoRA configuration
QLoRA throughout (4-bit base weights + trainable LoRA adapters):

| Setting | Value |
|---|---|
| Base model | Llama-3.2-3B-Instruct (`unsloth/Llama-3.2-3B-Instruct-bnb-4bit`) |
| Quantization | 4-bit (`load_in_4bit=True`) → **QLoRA** |
| LoRA rank `r` | 16 |
| `lora_alpha` | 16 |
| `lora_dropout` | 0.0 |
| Target modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| Max seq length | 2048 |
| Gradient checkpointing | `"unsloth"` |
| Optimizer | `adamw_8bit` |

All values live in `src/config.py`.

## 10. Training screenshots or logs
Add your run artifacts to `assets/` and reference them here, e.g.:
```
assets/stage1_training_log.png
assets/stage2_training_log.png
assets/stage3_dpo_log.png
```
Each notebook prints per-step loss (`logging_steps=1`) and the final
`trainer_stats`; screenshot those cells after running, or paste the loss logs
into this section.

## 11. Before vs after output comparison
See the reports:
- `reports/base_model_evaluation.md` — base model on 10 questions + problems.
- `reports/sft_model_comparison.md` — Base vs SFT table.
- `reports/final_evaluation.md` — Base vs SFT vs DPO table.

Short example (full tables in the reports):

| Question | Base | DPO-aligned |
|---|---|---|
| How can I cancel my order? | "Cancel it by cancelling it. Contact the company." | "You can cancel from the Orders section before it ships — open the order, click Cancel Order, and confirm. If it has already shipped, refuse delivery or request a return once it arrives." |

## 12. Final observations
- The **Base → SFT** step gives the largest quality jump: it fixes correctness,
  domain accuracy, and clarity and removes most hallucinations.
- The **SFT → DPO** step is a refinement: better tone, empathy, completeness,
  and safety phrasing; fewer generic/curt answers.
- Non-instruction pretraining is a small but useful warm-start that makes the
  SFT stage converge on domain phrasing faster.

## 13. Challenges faced
- **Flat-loss training bug (TinyLlama):** the original TinyLlama base plus a
  version-mismatched trl/Unsloth stack produced zero learning (flat loss). Fixed
  by moving to Llama-3.2-3B-Instruct with the standard chat-template +
  `train_on_responses_only` recipe and a clean Unsloth install.
- **Raw data is templated:** the Kaggle descriptions contain tokens like
  `{product_purchased}`; the prep script cleans these before building paragraphs.
- **Small preference set:** 53 triples nudge behaviour but aren't enough for
  large tone shifts; more data would help.
- **Environment specificity:** Unsloth installs are CUDA/torch-version specific —
  the notebooks use the Colab-tested install command.

## 14. Future improvements
- Scale up the base model (e.g. Llama-3.2-3B) and grow the datasets.
- Add automatic evaluation (LLM-as-judge, exact-policy checks) instead of manual tables.
- Try **ORPO** (single-stage instruction + preference) and compare with DPO.
- Add retrieval (RAG) over the live help-centre so answers stay current.
- Export a **GGUF** build for CPU-only local deployment (llama.cpp / Ollama).

---

## Repository layout
```
.
├── data/
│   ├── customer_support_tickets.csv   # <- download from Kaggle (not committed)
│   ├── non_instruction_data.csv       # Stage-1 raw domain text (generated)
│   ├── instruction_dataset.jsonl      # 104 instruction/response pairs
│   └── preference_dataset.jsonl       # 53 preference triples
├── notebooks/
│   ├── non_instruction_finetuning.ipynb
│   ├── instruction_finetuning.ipynb
│   └── dpo_alignment.ipynb
├── reports/
│   ├── base_model_evaluation.md
│   ├── sft_model_comparison.md
│   └── final_evaluation.md
├── src/
│   ├── config.py
│   ├── prepare_non_instruction_data.py
│   ├── build_instruction_dataset.py
│   ├── build_preference_dataset.py
│   └── inference.py
├── models/            # adapters saved here after training (git-ignored)
├── requirements.txt
└── README.md
```

## How to run

### Environment setup with uv (Python 3.12)
This project uses [uv](https://docs.astral.sh/uv/) and is pinned to **Python 3.12**
via `.python-version`. Create the virtual environment on your own machine
(the `.venv/` is git-ignored and platform-specific, so it is not committed):

```bash
# 1. create a Python 3.12 virtual environment (.venv/)
uv venv --python 3.12

# 2. activate it
#   Windows (PowerShell):
.venv\Scripts\Activate.ps1
#   Windows (cmd):
.venv\Scripts\activate.bat
#   macOS / Linux:
source .venv/bin/activate

# 3. install the core (data-prep / eval) dependencies
uv pip install -e .

# 4. (GPU machine / Colab only) install the training stack.
#    Unsloth is CUDA/torch-version specific — use the exact command from
#    https://github.com/unslothai/unsloth , e.g. for CUDA 12.1 + torch 2.3:
uv pip install "unsloth[cu121-torch230] @ git+https://github.com/unslothai/unsloth.git"
uv pip install "trl<0.9.0" peft accelerate bitsandbytes
```

> If `uv` is not installed yet: `pip install uv` (or see the uv install docs).
> `uv` will automatically download and manage the Python 3.12 toolchain for you.


**Step 0 — get the raw data**
Download `suraj520/customer-support-ticket-dataset` from Kaggle and save
`customer_support_tickets.csv` into `data/`.

**Step 1 — build the datasets**
```bash
python src/prepare_non_instruction_data.py --n 60   # -> data/non_instruction_data.csv
python src/build_instruction_dataset.py             # -> data/instruction_dataset.jsonl
python src/build_preference_dataset.py              # -> data/preference_dataset.jsonl
```

**Step 2 — train (Google Colab, GPU)**
Open the three notebooks in order and run all cells:
1. `notebooks/non_instruction_finetuning.ipynb`
2. `notebooks/instruction_finetuning.ipynb`
3. `notebooks/dpo_alignment.ipynb`

**Step 3 — chat with the final model**
```bash
python src/inference.py --q "How can I apply for reimbursement?"
# or interactive:
python src/inference.py
```

## Publishing to GitHub
```bash
git init
git add .
git commit -m "Fine-tuned Customer Support Assistant: data, notebooks, reports, inference"
git branch -M main
git remote add origin https://github.com/<your-username>/customer-support-assistant.git
git push -u origin main
```
Model weights and the raw Kaggle CSV are excluded via `.gitignore` (too large /
licensed). Share trained adapters via Hugging Face Hub or a release asset instead.
