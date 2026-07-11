"""
Central configuration for the Customer Support Assistant fine-tuning project.
Edit paths here if your layout differs.
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / "reports"
MODELS_DIR = PROJECT_ROOT / "models"

# Raw Kaggle dataset (suraj520/customer-support-ticket-dataset).
# Download from Kaggle and place the CSV here. The Kaggle file is usually named
# "customer_support_tickets.csv".
RAW_TICKETS_CSV = DATA_DIR / "customer_support_tickets.csv"

# Derived datasets (deliverables)
NON_INSTRUCTION_CSV = DATA_DIR / "non_instruction_data.csv"
INSTRUCTION_JSONL = DATA_DIR / "instruction_dataset.jsonl"
PREFERENCE_JSONL = DATA_DIR / "preference_dataset.jsonl"

# Adapter / model output dirs
NON_INSTRUCT_ADAPTER = MODELS_DIR / "non_instruct_adapter"
SFT_ADAPTER = MODELS_DIR / "sft_adapter"
DPO_ADAPTER = MODELS_DIR / "dpo_adapter"

# ---------------------------------------------------------------------------
# Model / training hyper-parameters
# ---------------------------------------------------------------------------
BASE_MODEL = "unsloth/Llama-3.2-3B-Instruct-bnb-4bit"   # Llama-3.2-3B-Instruct, 4-bit
MAX_SEQ_LENGTH = 2048
LOAD_IN_4BIT = True                        # QLoRA
CHAT_TEMPLATE = "llama-3.1"                # Unsloth get_chat_template name (works for 3.2)

# LoRA config (shared across stages)
LORA_R = 16
LORA_ALPHA = 16
LORA_DROPOUT = 0.0
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# Chat / prompt template (Alpaca-style, simple and TinyLlama-friendly)
PROMPT_TEMPLATE = (
    "### Instruction:\n{instruction}\n\n### Response:\n{response}"
)
INFERENCE_TEMPLATE = (
    "### Instruction:\n{instruction}\n\n### Response:\n"
)
