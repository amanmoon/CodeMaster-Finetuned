import torch
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset

model_name = "unsloth/Qwen2.5-Coder-0.5B-bnb-4bit"
max_seq_length = None

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    # max_seq_length = max_seq_length,
    load_in_4bit = True,
    full_finetuning = False
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 32, 
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 64,
    lora_dropout = 0, 
    bias = "lora_only",    
    use_gradient_checkpointing = "unsloth",
)

def formatting_prompts_func(examples):
    instructions = examples["problem_description"]
    outputs      = examples["response"]
    texts = []
    for instruction, output in zip(instructions, outputs):
        text = (
            "<|im_start|>system\nYou are a competitive programmer. Solve the LeetCode problem provided.<|im_end|>\n"
            f"<|im_start|>user\n{instruction}<|im_end|>\n"
            f"<|im_start|>assistant\n{output}<|im_end|>"
        )
        texts.append(text)
    return { "text" : texts, }

dataset = load_dataset("newfacade/LeetCodeDataset", split = "train")
dataset = dataset.map(formatting_prompts_func, batched = True)

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    packing = True,
    # max_seq_length = max_seq_length,
    args = TrainingArguments(
        per_device_train_batch_size = 6,
        gradient_accumulation_steps = 16,
        warmup_steps = 10,
        num_train_epochs=6,
        max_steps = -1,
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 3,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "cosine",
        seed = 3407,
        output_dir = "leetcode_full_finetune",
        report_to="tensorboard",
        logging_dir="./logs",
    ),
)

trainer.train()
