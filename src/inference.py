"""
Simple inference script for the final DPO-aligned Customer Support Assistant.

Loads the saved adapter (DPO if present, else SFT) on top of
Llama-3.2-3B-Instruct and answers support questions using the Llama-3 chat
template.

Usage:
    python src/inference.py                                   # interactive
    python src/inference.py --q "How can I apply for reimbursement?"

A CUDA GPU is recommended (Unsloth). On CPU-only machines, export a merged/GGUF
model from the DPO notebook and run it with llama.cpp / Ollama instead.
"""
import argparse
import os

try:
    from config import BASE_MODEL, MAX_SEQ_LENGTH, DPO_ADAPTER, SFT_ADAPTER, CHAT_TEMPLATE
except ImportError:
    from src.config import BASE_MODEL, MAX_SEQ_LENGTH, DPO_ADAPTER, SFT_ADAPTER, CHAT_TEMPLATE

_model = None
_tokenizer = None


def _pick_adapter():
    """Prefer the DPO adapter; fall back to the SFT adapter."""
    for path in (DPO_ADAPTER, SFT_ADAPTER):
        if os.path.isdir(path) and os.path.exists(os.path.join(path, "adapter_config.json")):
            return str(path)
    raise FileNotFoundError(
        "No trained adapter found. Run the SFT and DPO notebooks first so that "
        f"'{DPO_ADAPTER}' (or at least '{SFT_ADAPTER}') exists."
    )


def load_model():
    """Load the saved adapter folder directly (rebuilds base + LoRA) and cache it."""
    global _model, _tokenizer
    if _model is not None:
        return _model, _tokenizer

    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template

    adapter = _pick_adapter()
    print(f"Loading adapter '{adapter}' (base: {BASE_MODEL}) ...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=adapter,          # loading the adapter dir rebuilds base + LoRA
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )
    tokenizer = get_chat_template(tokenizer, chat_template=CHAT_TEMPLATE)
    FastLanguageModel.for_inference(model)  # 2x faster generation

    _model, _tokenizer = model, tokenizer
    return _model, _tokenizer


def generate_answer(question: str, max_new_tokens: int = 200) -> str:
    """Return the assistant's answer to a single support question."""
    model, tokenizer = load_model()
    messages = [{"role": "user", "content": question.strip()}]
    input_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)
    output = model.generate(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        repetition_penalty=1.1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )
    # decode only the newly generated tokens
    return tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True).strip()


def _interactive():
    print("Customer Support Assistant (DPO-aligned). Type 'exit' to quit.\n")
    while True:
        try:
            q = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if q.lower() in {"exit", "quit", "q"}:
            break
        if not q:
            continue
        print("Assistant:", generate_answer(q), "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Customer Support Assistant inference")
    ap.add_argument("--q", "--question", dest="q", type=str, default=None,
                    help="ask a single question and exit")
    ap.add_argument("--max_new_tokens", type=int, default=200)
    args = ap.parse_args()

    if args.q:
        # question = "How can I apply for reimbursement?"
        # answer = generate_answer(question)
        # print(answer)
        print(generate_answer(args.q, max_new_tokens=args.max_new_tokens))
    else:
        _interactive()
