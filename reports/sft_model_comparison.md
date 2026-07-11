# Base Model vs Instruction Fine-Tuned (SFT) Model

**Base:** TinyLlama-1.1B (no tuning) · **SFT:** TinyLlama-1.1B + QLoRA on `instruction_dataset.jsonl` (104 pairs)
**Domain:** Customer Support Assistant

> How to use this file: the "answers" are representative of the expected
> behaviour difference. After running `notebooks/instruction_finetuning.ipynb`,
> replace them with your model's actual generations for a truthful report.

## Evaluation criteria
Correctness · Domain accuracy · Clarity · Safety · Helpfulness · Less generic · Better domain-specific behaviour.

## Comparison Table

| # | Question | Base Model Answer | Fine-Tuned (SFT) Answer | Which is Better? | Reason |
|---|----------|-------------------|--------------------------|------------------|--------|
| 1 | How can I cancel my order? | "You can cancel an order by cancelling it. Please contact the company." | "You can cancel your order from the Orders section before it ships: open the order, click Cancel Order, and confirm. If it has already shipped, refuse delivery or request a return once it arrives." | **SFT** | Gives the exact steps and the pre/post-shipment rule; not circular. |
| 2 | How do I track my order? | "Track it using tracking. Tracking numbers are provided by couriers." | "Go to Orders, select the order, and click Track Shipment to see status and the estimated delivery date. You also get tracking emails once it ships." | **SFT** | Concrete navigation path and what to expect. |
| 3 | I received a damaged item. What should I do? | "Sorry, maybe return it or keep it." | "I'm sorry it arrived damaged. Open the order, choose Report a Problem, upload a photo, and we'll arrange a free replacement or full refund and cover return shipping." | **SFT** | Empathetic, correct process, mentions free return shipping. |
| 4 | How do I reset my password? | "Reset it in the settings. Use a strong password." | "On the login page click Forgot Password, enter your account email, and follow the secure reset link we send (valid 30 minutes). Check spam if it doesn't arrive." | **SFT** | Accurate flow with the real UI element and a fallback tip. |
| 5 | Why was my payment declined? | "Payments can be declined for reasons. Try again." | "Usually due to incorrect card details, insufficient funds, an expired card, or your bank's fraud check. Verify your details, try again or another method, and contact your bank if it continues." | **SFT** | Enumerates real causes and next steps. |
| 6 | How do I claim warranty? | "You claim a warranty by claiming it." | "Open the order, choose Warranty Claim, describe the fault and attach a photo. We verify the warranty period and arrange repair, replacement, or refund. Keep your invoice as proof of purchase." | **SFT** | Full procedure plus proof-of-purchase guidance. |
| 7 | How long do refunds take? | "Some time, depends." | "Refunds are processed within 3-5 business days after we receive your return, to your original payment method; your bank may take another 2-7 days." | **SFT** | Specific, policy-accurate timeframe. |
| 8 | Can I change my shipping address after ordering? | "Yes. Or maybe not." | "You can edit it while the order is Processing, under Orders > Edit Shipping Address. Once shipped, contact us quickly and we'll try to reroute with the carrier, though it isn't guaranteed." | **SFT** | Correct conditional rule; clear and honest about limits. |
| 9 | My order says delivered but I didn't receive it. | "If it says delivered it was delivered." | "I'm sorry for the trouble. Check around your property and with neighbours and allow 24 hours. If it still doesn't appear, contact us with your order number and we'll investigate with the carrier and arrange a replacement or refund." | **SFT** | Actionable investigation path instead of blaming the customer. |
| 10 | How can I contact customer support? | "Through their support." | "Via live chat on our website, by email, or by phone during business hours. Live chat is usually fastest - have your order number ready." | **SFT** | Names the channels and gives a practical tip. |

## Summary

Instruction fine-tuning transforms the model from a vague next-token predictor
into a **structured, domain-specific assistant**. Across all 10 questions the SFT
model wins on correctness, domain accuracy, clarity, and helpfulness while
remaining safe. Remaining weaknesses (occasional over-brief answers, tone that can
be flat, and rare drift toward generic phrasing) are exactly what **DPO alignment**
in Stage 3 targets.
