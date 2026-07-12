"""
Gradio front end for the DPO-aligned Customer Support Assistant.

Loads Llama-3.2-3B-Instruct + your DPO LoRA adapter FROM THE HUGGING FACE HUB.

Set your model id (the adapter repo you pushed) via env var or edit MODEL_ID:
    export HF_MODEL_ID="your-username/customer-support-assistant-dpo"

Requirements: a CUDA GPU (the model loads in 4-bit via bitsandbytes).
    pip install -r requirements-app.txt
    python app.py

In Google Colab (GPU runtime), see the launch snippet at the bottom of this file.
"""
import os
import torch
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ---- Configuration --------------------------------------------------------
MODEL_ID = os.environ.get("HF_MODEL_ID", "your-username/customer-support-assistant-dpo")
BASE_MODEL = os.environ.get("HF_BASE_MODEL", "unsloth/Llama-3.2-3B-Instruct-bnb-4bit")
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "256"))

SYSTEM_PROMPT = (
    "You are a helpful, professional customer support assistant. "
    "Answer clearly and concisely, and stay on customer-support topics."
)

# ---- Load model once ------------------------------------------------------
print(f"Loading base '{BASE_MODEL}' + adapter '{MODEL_ID}' ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, device_map="auto")
model = PeftModel.from_pretrained(base, MODEL_ID)
model.eval()

# Llama-3 end-of-turn token, so generation stops cleanly.
_EOT = tokenizer.convert_tokens_to_ids("<|eot_id|>")
TERMINATORS = [tokenizer.eos_token_id] + ([_EOT] if isinstance(_EOT, int) and _EOT >= 0 else [])
print("Model ready.")


def _content_to_str(content):
    """Handle both plain-string and list-of-parts (multimodal) message content."""
    if isinstance(content, list):
        return " ".join(p.get("text", "") for p in content if isinstance(p, dict)).strip()
    return content or ""


def build_prompt(message, history):
    """Build the Llama-3 prompt string manually (robust across template versions)."""
    parts = [f"<|start_header_id|>system<|end_header_id|>\n\n{SYSTEM_PROMPT}<|eot_id|>"]
    for turn in (history or []):
        if isinstance(turn, dict):                      # Gradio "messages" format
            role = turn.get("role")
            content = _content_to_str(turn.get("content"))
            if role in ("user", "assistant") and content:
                parts.append(f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>")
        else:                                           # ("user", "assistant") tuple format
            u, a = turn
            if u:
                parts.append(f"<|start_header_id|>user<|end_header_id|>\n\n{_content_to_str(u)}<|eot_id|>")
            if a:
                parts.append(f"<|start_header_id|>assistant<|end_header_id|>\n\n{_content_to_str(a)}<|eot_id|>")
    parts.append(f"<|start_header_id|>user<|end_header_id|>\n\n{message}<|eot_id|>")
    parts.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
    return "<|begin_of_text|>" + "".join(parts)


def respond(message, history):
    prompt = build_prompt(message, history)
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    input_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.1,
            eos_token_id=TERMINATORS,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(output[0][input_len:], skip_special_tokens=True).strip()


demo = gr.ChatInterface(
    fn=respond,
    title="🎧 Customer Support Assistant",
    description="Fine-tuned Llama-3.2-3B-Instruct (SFT + DPO). Ask a customer-support question.",
    examples=[
        "How can I cancel my order?",
        "How do I reset my password?",
        "I received a damaged item. What should I do?",
        "How do I apply for reimbursement?",
        "My order says delivered but I did not receive it.",
    ],
)

if __name__ == "__main__":
    demo.launch(share=True)

# ---------------------------------------------------------------------------
# Run in Google Colab (GPU runtime) with these cells:
#
#   !pip install -q gradio transformers peft accelerate bitsandbytes
#   import os
#   os.environ["HF_MODEL_ID"] = "your-username/customer-support-assistant-dpo"
#   !git clone https://github.com/<your-username>/customer-support-assistant.git
#   %cd customer-support-assistant
#   !python app.py
# ---------------------------------------------------------------------------
