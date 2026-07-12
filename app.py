"""
Gradio front end for the DPO-aligned Customer Support Assistant.

Loads Llama-3.2-3B-Instruct + your DPO LoRA adapter FROM THE HUGGING FACE HUB,
so you only need to point it at your model repo.

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
MODEL_ID = os.environ.get("HF_MODEL_ID", "renjanb/customer-support-assistant-dpo")
BASE_MODEL = os.environ.get("HF_BASE_MODEL", "unsloth/Llama-3.2-3B-Instruct-bnb-4bit")
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "256"))

SYSTEM_PROMPT = (
    "You are a helpful, professional customer support assistant. "
    "Answer clearly and concisely, and stay on customer-support topics."
)

# ---- Load model once ------------------------------------------------------
print(f"Loading base '{BASE_MODEL}' + adapter '{MODEL_ID}' ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)   # adapter repo carries the chat template
base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, device_map="auto")
model = PeftModel.from_pretrained(base, MODEL_ID)
model.eval()
print("Model ready.")


def _to_messages(message, history):
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in (history or []):
        # supports both ("user","bot") tuples and {"role","content"} dicts
        if isinstance(turn, dict):
            msgs.append(turn)
        else:
            user, bot = turn
            if user:
                msgs.append({"role": "user", "content": user})
            if bot:
                msgs.append({"role": "assistant", "content": bot})
    msgs.append({"role": "user", "content": message})
    return msgs


def respond(message, history):
    messages = _to_messages(message, history)
    input_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)
    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True).strip()


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
    # share=True prints a public https link (handy from Colab)
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
