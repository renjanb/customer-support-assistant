"""
Stage 1 data prep: build the NON-INSTRUCTION (raw domain text) dataset.

Reads the raw Kaggle "customer-support-ticket-dataset" CSV and turns each ticket
row into a coherent free-text paragraph about customer support. This raw text is
used for continued-pretraining-style (non-instruction) fine-tuning so the base
model absorbs the domain vocabulary and style BEFORE we do instruction tuning.

Input : data/customer_support_tickets.csv   (download from Kaggle)
Output: data/non_instruction_data.csv        (columns: text)

Run:
    python src/prepare_non_instruction_data.py --n 60
"""
import argparse
import re
import sys
import pandas as pd

try:
    from config import RAW_TICKETS_CSV, NON_INSTRUCTION_CSV
except ImportError:  # allow running from project root
    from src.config import RAW_TICKETS_CSV, NON_INSTRUCTION_CSV


def clean(text) -> str:
    if pd.isna(text):
        return ""
    return re.sub(r"\s+", " ", str(text)).strip()


def fill_placeholders(desc: str, product: str) -> str:
    """The raw dataset stores template tokens like '{product_purchased}'."""
    if not desc:
        return desc
    desc = desc.replace("{product_purchased}", product or "the product")
    # strip any other leftover {tokens}
    desc = re.sub(r"\{[^}]*\}", "the product", desc)
    return desc


def row_to_paragraph(row) -> str:
    """Compose one natural-language paragraph from a ticket row."""
    product = clean(row.get("Product Purchased"))
    ttype = clean(row.get("Ticket Type"))
    subject = clean(row.get("Ticket Subject"))
    desc = fill_placeholders(clean(row.get("Ticket Description")), product)
    status = clean(row.get("Ticket Status"))
    priority = clean(row.get("Ticket Priority"))
    channel = clean(row.get("Ticket Channel"))
    resolution = clean(row.get("Resolution"))

    parts = []
    if ttype and subject:
        parts.append(
            f"A customer raised a {ttype.lower()} ticket about \"{subject}\""
            + (f" regarding their {product}." if product else ".")
        )
    elif subject:
        parts.append(f"A support ticket was opened about \"{subject}\".")

    if desc:
        parts.append(f"The customer explained: {desc}")

    meta = []
    if priority:
        meta.append(f"priority {priority.lower()}")
    if channel:
        meta.append(f"submitted via {channel.lower()}")
    if status:
        meta.append(f"current status {status.lower()}")
    if meta:
        parts.append("This ticket was " + ", ".join(meta) + ".")

    if resolution:
        parts.append(f"Resolution provided by the support team: {resolution}")
    elif status and status.lower() != "closed":
        parts.append(
            "The support team is still working with the customer to resolve the issue."
        )

    return " ".join(parts).strip()


def build(n: int) -> pd.DataFrame:
    if not RAW_TICKETS_CSV.exists():
        sys.exit(
            f"[ERROR] Raw dataset not found at {RAW_TICKETS_CSV}\n"
            "Download 'suraj520/customer-support-ticket-dataset' from Kaggle and "
            "place customer_support_tickets.csv in the data/ folder."
        )

    df = pd.read_csv(RAW_TICKETS_CSV)
    print(f"Loaded {len(df):,} raw tickets with columns: {list(df.columns)}")

    paragraphs, seen = [], set()
    for _, row in df.iterrows():
        para = row_to_paragraph(row)
        # keep only substantial, unique paragraphs
        if len(para) >= 120 and para not in seen:
            seen.add(para)
            paragraphs.append(para)
        if len(paragraphs) >= n:
            break

    if len(paragraphs) < n:
        print(f"[WARN] Only produced {len(paragraphs)} paragraphs (asked for {n}).")

    return pd.DataFrame({"text": paragraphs})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60,
                    help="number of raw paragraphs to produce (>= 50 required)")
    args = ap.parse_args()

    out = build(args.n)
    NON_INSTRUCTION_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(NON_INSTRUCTION_CSV, index=False)
    print(f"Wrote {len(out)} paragraphs -> {NON_INSTRUCTION_CSV}")
    print("\n--- sample paragraph ---")
    if len(out):
        print(out.iloc[0]["text"][:400])


if __name__ == "__main__":
    main()
