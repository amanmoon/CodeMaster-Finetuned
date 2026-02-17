# Model Evaluation: LeetCode Fine-Tuning Comparison

This document presents a comparative analysis of three model versions evaluated on classic LeetCode problems. The goal was to observe the transition from a base "completion" model to a specialized coding assistant using **Supervised Fine-Tuning (SFT)**.

## 🔗 Quick Links
* **Model Repository:** [Hugging Face](https://huggingface.co/amanmoon/leetcode_finetuned_Qwen2.5-Coder-0.5B-bnb-4bit)
* **Base Model:** `unsloth/Qwen2.5-Coder-0.5B-bnb-4bit`

## 🛠 Training Methodology
The models were trained using **Supervised Fine-Tuning (SFT)**. This technique involves training the base model on a high-quality dataset of instruction-output pairs (Problem Statement -> Python/C++ Solution), allowing the model to learn both the "assistant" conversational format and specific algorithmic logic.

---

## Performance Summary

| Feature | Base Model | QLoRA Fine-Tune | Full Fine-Tune |
| :--- | :--- | :--- | :--- |
| **Output Style** | Raw code / Completion | Conversational + Code | Concise + Structured |
| **Instruction Following** | Low (Loops/Repeats) | High (Follows prompts) | Very High |
| **Language Choice** | C++ (Default) | Python | Python |
| **Stability** | Prone to repetition loops | Moderate (Some suffix noise) | High stability |

---

## Detailed Results

### 1. Two Sum
**Problem:** Find indices of two numbers that add up to a target.

* **Base Model:** Provided a standard C++ nested loop solution. Functional but lacks explanation.
* **QLoRA Model:** Switched to Python. Implemented an optimized $O(n)$ hash map approach. 
* **Full Fine-Tune:** Provided a clean, professional Python implementation with focused comments on time complexity.



### 2. Longest Substring Without Repeating Characters
**Problem:** Find the length of the longest substring with unique characters.

* **Base Model:** **Failed.** Entered a repetition loop, repeating the problem description and Chinese characters indefinitely.
* **QLoRA Model:** **Success.** Implemented a sliding window approach.
* **Full Fine-Tune:** **Success.** Implemented an elegant sliding window solution with clear variable naming (`longest_streak`).



### 3. Median of Two Sorted Arrays
**Problem:** Find the median in $O(\log (m+n))$ time.

* **Base Model:** **Failed.** Repeated the "You are a competitive programmer" persona prompt without generating code.
* **QLoRA Model:** Used a `sorted()` approach. While functional, it operates at $O(N \log N)$.
* **Full Fine-Tune:** Attempted a more complex approach using `heapq` (min-heaps), showing a better grasp of advanced data structures.

---

## Observations & Analysis

### The "Repetition" Trap
The **Base Model** struggled significantly with the "assistant" format. Because it is a base-completion model, it often got stuck in loops. SFT successfully bridged this gap, teaching the model to recognize when to stop generating code.

### QLoRA vs. Full Fine-Tuning
* **QLoRA:** Showed a dramatic improvement in "personality." However, it exhibited some "suffix noise" (repeating `_equalTo` or `(egt`).
* **Full Fine-Tuning:** Produced the most stable results. The outputs were concise, free of repetitive suffixes, and the logic was generally more robust across all 5 samples.

---

## Acknowledgments & Licensing
- **Base Model:** This project is improved using [Qwen2.5-Coder-0.5B](https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B) by Alibaba Cloud.
- **Dataset:** Fine-tuned on the [newfacade/LeetCodeDataset](https://huggingface.co/datasets/newfacade/LeetCodeDataset).
- **License:** The original code and fine-tuning scripts in this repository are licensed under the **MIT License**. The model weights are subject to the [Qwen License Agreement](https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B/blob/main/LICENSE).

---
