# Final Evaluation — Base vs SFT vs DPO

**Base:** TinyLlama-1.1B (no tuning)
**SFT:** Base + QLoRA on `instruction_dataset.jsonl` (104 pairs)
**DPO:** SFT + DPO on `preference_dataset.jsonl` (53 triples)
**Domain:** Customer Support Assistant

> How to use this file: the answers illustrate the expected trajectory across the
> three stages. After running all three notebooks, paste your actual generations
> so the report is truthful for your run. Settings: `temperature=0.7, top_p=0.9`.

## Evaluation criteria
Correctness · Helpfulness · Domain accuracy · Safety · Tone · Clarity · Hallucination reduction · Professional quality.

## Final Comparison Table

| # | Question | Base Model Answer | SFT Model Answer | DPO Model Answer | Best | Reason |
|---|----------|-------------------|------------------|------------------|------|--------|
| 1 | How can I cancel my order? | "Cancel it by cancelling it. Contact the company." | "Cancel from the Orders section before it ships; if shipped, request a return." | "You can cancel your order from the Orders section before it ships - open the order, click Cancel Order, and confirm. If it has already shipped, you can refuse delivery or request a return once it arrives, and I'm happy to help with either." | **DPO** | Same correct steps as SFT but warmer, more complete, offers further help. |
| 2 | My order says delivered but I didn't receive it. | "If it says delivered, you got it." | "Check around and with neighbours, allow 24 hours, then contact us." | "I'm sorry for the worry. Please check around your property and with neighbours and allow 24 hours, as carriers sometimes scan early. If it still doesn't appear, share your order number and we'll open a carrier investigation and arrange a replacement or refund." | **DPO** | Most empathetic and reassuring while keeping the correct process; best tone. |
| 3 | Someone may have accessed my account. | "Change your password. Security matters." | "Reset your password and enable 2FA; review recent orders." | "Please reset your password now and turn on two-factor authentication. Review recent orders and saved payment methods, remove any devices you don't recognise under Security, and contact us if you spot unauthorised activity so we can secure the account." | **DPO** | Most thorough and safety-focused; covers all mitigation steps. |
| 4 | Why was my payment declined? | "Declined for reasons. Try again." | "Incorrect details, insufficient funds, expired card, or bank fraud check; retry or use another method." | "Payments are usually declined due to incorrect card details, insufficient funds, an expired card, or your bank's fraud check. Please double-check your details and try again or use another method, and if it keeps happening your bank can authorise the transaction." | **DPO** | Clear causes plus a calm, actionable resolution; professional tone. |
| 5 | How do I return a product? | "Return it somewhere." | "Within 30 days: Orders > Return Item, print the prepaid label, drop it off." | "You can return most items within 30 days. Go to Orders, select the item, choose Return Item, pick a reason, and print the prepaid label, then drop it at any partner location. Your refund is issued after we receive and inspect it." | **DPO** | Complete, correctly ordered steps with the refund condition. |
| 6 | How do I claim warranty? | "Claim it by claiming it." | "Orders > Warranty Claim, describe the fault, keep your invoice." | "Open the order, choose Warranty Claim, describe the fault, and attach a photo or video. We'll verify the warranty period and arrange a repair, replacement, or refund - just keep your invoice as proof of purchase." | **DPO** | Adds evidence step and reassurance; clearest professional guidance. |
| 7 | How long do refunds take? | "Some time." | "3-5 business days after we receive the return; bank adds 2-7 days." | "Refunds are processed within 3-5 business days after we receive your return, credited to your original payment method, and your bank may take a further 2-7 days to show it." | **SFT / DPO (tie)** | Both correct and specific; DPO slightly smoother but content is equal. |
| 8 | Can I speak to a human agent? | "Agents are people who help." | "Type 'agent' in chat to reach a human during hours." | "Of course. In live chat, type 'agent' or select Talk to a Human to be connected during business hours. If none are available, leave your details and we'll follow up by email." | **DPO** | Friendlier, adds the after-hours fallback; better helpfulness. |
| 9 | Do I have to pay for return shipping? | "Probably yes." | "Free if it's our error; a small fee for change-of-mind." | "Return shipping is free when the return is due to our error - a damaged, faulty, or wrong item. For change-of-mind returns a small fee may be deducted, and we always show it before you confirm." | **DPO** | Precise policy with transparency about fees; professional. |
| 10 | How do I cancel my subscription? | "Subscriptions are final." (wrong) | "Account Settings > Subscriptions > Cancel; active until period end." | "Go to Account Settings, then Subscriptions, select the plan, and click Cancel Subscription. It stays active until the end of your current billing period and won't renew - you can resubscribe any time." | **DPO** | Base is factually wrong (hallucination); DPO is correct, complete, and reassuring. |

## Stage-by-stage summary

- **Base -> SFT:** the biggest jump. SFT fixes correctness, domain accuracy, and
  clarity, and eliminates most hallucinations and circular non-answers.
- **SFT -> DPO:** a refinement, not a rewrite. DPO improves **tone, empathy,
  completeness, safety phrasing, and professionalism**, and further reduces
  generic or curt replies by having learned which answers customers prefer.
- **Hallucination:** worst in Base (e.g. "subscriptions are final"), largely
  removed by SFT, and most reliably grounded after DPO.

## Overall verdict

The DPO-aligned model is the best of the three on nearly every question. It keeps
the correctness gained from SFT and adds the warmth, thoroughness, and safety that
make it suitable as an internal customer-support assistant. Given TinyLlama-1.1B's
small size, answers can still occasionally be brief or slightly repetitive; a
larger base model or more preference data would push quality further (see the
README's "Future improvements").
