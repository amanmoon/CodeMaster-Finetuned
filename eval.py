import json
import torch
import re
from tqdm import tqdm
from unsloth import FastLanguageModel
from datasets import load_dataset

base_model_name = "unsloth/Qwen2.5-Coder-0.5B-bnb-4bit"

lora_model_path = "./models/leetcode_lora/checkpoint-498"
full_ft_model_path = "./models/leetcode_full_finetune/checkpoint-168"

device = "cuda" if torch.cuda.is_available() else "cpu"

def extract_code(text):
    pattern = r"<\|im_start\|>assistant\n(.*?)(?:<\|im_end\|>|$)"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else text


def run_eval(model, tokenizer, dataset, tag):
    results = []

    for entry in tqdm(dataset, desc=f"Evaluating {tag}"):
        instruction = entry["problem_description"]

        prompt = (
            "<|im_start|>system\nYou are a competitive programmer.<|im_end|>\n"
            f"<|im_start|>user\n{instruction}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

        inputs = tokenizer([prompt], return_tensors="pt").to(device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            pad_token_id=tokenizer.eos_token_id,
        )

        decoded = tokenizer.batch_decode(outputs)[0]
        generated_code = extract_code(decoded)

        results.append({
            "problem": instruction[:200] + "...",
            "generated_code": generated_code,
        })

    return results


eval_dataset = load_dataset("newfacade/LeetCodeDataset", split="train").select(range(5))

base_model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=base_model_name,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(base_model)

lora_model, _ = FastLanguageModel.from_pretrained(
    model_name=lora_model_path,  
    load_in_4bit=True,
)
FastLanguageModel.for_inference(lora_model)

full_ft_model, _ = FastLanguageModel.from_pretrained(
    model_name=full_ft_model_path,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(full_ft_model)

base_results = run_eval(base_model, tokenizer, eval_dataset, "BASE_MODEL")
lora_results = run_eval(lora_model, tokenizer, eval_dataset, "LORA_TUNED_MODEL")
full_ft_results = run_eval(full_ft_model, tokenizer, eval_dataset, "FULL_FINETUNED_MODEL")

output_file = "eval_results_all_models.json"

final_data = {
    "total_samples": len(eval_dataset),
    "models": {
        "base_model": base_model_name,
        "lora_model_path": lora_model_path,
        "full_ft_model_path": full_ft_model_path,
    },
    "results": {
        "base_model_outputs": base_results,
        "lora_model_outputs": lora_results,
        "full_finetuned_outputs": full_ft_results,
    }
}

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(final_data, f, indent=4)
