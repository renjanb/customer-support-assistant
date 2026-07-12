"""
Self-contained Colab inference for the DPO-aligned Customer Support Assistant.

Run this in a BRAND-NEW Google Colab runtime (Runtime -> Change runtime type ->
T4 GPU). It does not need the repo cloned - it just needs your dpo_adapter.zip.

How to use in Colab (paste each block as its own cell, in order):

    # Cell 1 - install
    !pip install --upgrade unsloth unsloth_zoo

    # Cell 2 - upload + unpack your adapter zip
    from google.colab import files
    import shutil, os
    up = files.upload()                       # pick dpo_adapter.zip
    zip_name = next(iter(up))
    os.makedirs("dpo_adapter", exist_ok=True)
    shutil.unpack_archive(zip_name, "dpo_adapter")
    print(os.listdir("dpo_adapter"))

    # Cell 3 - run the code below
    exec(open("colab_inference.py").read())   # if you uploaded this file, or just paste it

    # Cell 4 - ask questions
    print(generate_answer("How can I apply for reimbursement?"))
"""
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

ADAPTER_DIR = "dpo_adapter"      # folder created by unzipping dpo_adapter.zip
MAX_SEQ_LENGTH = 2048

print(f"Loading adapter '{ADAPTER_DIR}' on Llama-3.2-3B-Instruct ...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=ADAPTER_DIR,      # loading the adapter folder rebuilds base + LoRA
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,
    load_in_4bit=True,
)
tokenizer = get_chat_template(tokenizer, chat_template="llama-3.1")
FastLanguageModel.for_inference(model)


def generate_answer(question: str, max_new_tokens: int = 200) -> str:
    """Answer a single customer-support question with the DPO-aligned model."""
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
    return tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True).strip()


print("Model ready. Call generate_answer('your question here').")

if __name__ == "__main__":
    for _q in ["How can I apply for reimbursement?",
               "How can I cancel my order?",
               "I received a damaged item. What should I do?"]:
        print("\nQ:", _q)
        print("A:", generate_answer(_q))
