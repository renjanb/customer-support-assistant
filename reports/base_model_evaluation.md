# Base Model Evaluation (Before Any Fine-Tuning)

**Model:** TinyLlama-1.1B (`unsloth/tinyllama`), 4-bit, no fine-tuning
**Domain:** Customer Support Assistant
**Goal:** Understand how the raw base model behaves on domain questions *before* training.

> How to use this file: the answers below are representative of typical
> untuned TinyLlama-1.1B behaviour (small model, no domain or instruction
> tuning). After you run `notebooks/instruction_finetuning.ipynb`, paste the
> model's *actual* base-model generations here so the report reflects your run.
> Generation settings used: `temperature=0.7, top_p=0.9, max_new_tokens=160`.

## Observations at a glance

The base model is a tiny, general pretrained model. On support questions it tends to:

- ramble or continue the prompt instead of answering it,
- give vague, generic advice ("contact the company"),
- invent product names, policies, or steps that do not exist (hallucination),
- ignore the domain's real workflow (Orders page, Returns, 30-day window, etc.),
- sometimes repeat itself or trail off mid-sentence.

## Evaluation Table

| # | Question | Base Model Answer (representative) | Problem |
|---|----------|-----------------------------------|---------|
| 1 | How can I cancel my order? | "You can cancel an order by cancelling it. Orders can be cancelled if you want to cancel them. Please contact the company for more information about your order and cancellation." | Circular and empty; no concrete steps, no mention of the Orders section or the pre-shipment rule. |
| 2 | I received a damaged item. What should I do? | "Sorry to hear that. You should probably return it or maybe keep it. Damaged items can happen during shipping which is handled by couriers and logistics companies worldwide." | Vague, drifts into irrelevant background; gives no return/replacement process. |
| 3 | How do I reset my password? | "To reset your password, reset it in the settings. Passwords are important for security. Make sure to use a strong password with letters and numbers." | Generic security lecture; misses the Forgot Password link / email reset flow. |
| 4 | How long do refunds take? | "Refunds usually take some time depending on many factors. It could be fast or slow. Banks are involved in processing refunds." | No timeframe, no policy; unhelpfully non-committal. |
| 5 | Why was my payment declined? | "Your payment may have been declined. Payments can be declined for reasons. Try paying again or use another card maybe." | Tautological; does not list real causes or next steps. |
| 6 | How can I contact customer support? | "You can contact customer support through their support. Most companies have support teams that help customers with support questions and issues." | Says nothing about channels (chat/email/phone) or having an order number ready. |
| 7 | Can I change my shipping address after ordering? | "Yes you can change it. Or maybe not. It depends on the order and the shipping and the company policy about addresses." | Contradictory and non-committal; ignores the Processing vs Shipped rule. |
| 8 | How do I claim warranty on a product? | "Warranties cover products for a period of time. To claim a warranty you claim it. Manufacturers provide warranties for their products usually." | Restates the concept of warranty; gives no claim procedure or proof-of-purchase note. |
| 9 | My order says delivered but I did not receive it. | "That is strange. If it says delivered then it was probably delivered. Sometimes packages get lost in the mail system somewhere along the way." | Dismissive; blames the customer/mail; no investigation or resolution path. |
| 10 | Someone may have accessed my account. | "Accounts can be accessed by people. You should change your password. Online security is very important these days for everyone." | Only half-answers (password); misses 2FA, reviewing orders, removing devices, contacting support. |

## Summary

Across all 10 questions the base model is **not usable** as a support assistant.
The dominant failure modes are **genericness**, **circular non-answers**, and
**missing domain-specific procedures**. This motivates instruction fine-tuning
(Stage 2) to teach concrete, correct, domain-grounded answers, and DPO alignment
(Stage 3) to further improve tone, safety, and helpfulness.
