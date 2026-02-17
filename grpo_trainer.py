import re
import signal
import torch
from datasets import load_dataset
from trl import GRPOTrainer, GRPOConfig
from unsloth import FastLanguageModel

MODEL_NAME = "unsloth/Qwen2.5-Coder-0.5B-bnb-4bit" 
DATASET_NAME = "newfacade/LeetCodeDataset"
OUTPUT_DIR = "grpo-leetcode-coder"

dataset = load_dataset(DATASET_NAME, split="train")

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("Execution timed out")

def extract_python_code(text):
    """
    Extracts content strictly between ```python and ``` tags.
    Returns None if no valid block is found.
    """
    pattern = r"```python\s+(.*?)\s+```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1)
    pattern_generic = r"```\s+(.*?)\s+```"
    match_generic = re.search(pattern_generic, text, re.DOTALL)
    if match_generic:
        return match_generic.group(1)
    return None

def run_test_case(generated_code, test_code, timeout=2):
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout)

    try:
        exec_globals = {}
        
        exec(generated_code, exec_globals)
        
        candidate_func = None
        for name, obj in exec_globals.items():
            if callable(obj) and name != '__builtins__':
                candidate_func = obj
        
        if not candidate_func:
            return 0.0 

        test_globals = {}
        exec(test_code, test_globals)
        
        if 'check' not in test_globals:
            return 0.0 
            
        check_fn = test_globals['check']

        check_fn(candidate_func)
        
        return 1.0
        
    except AssertionError:
        return 0.0
    except TimeoutException:
        return 0.0
    except Exception as e:
        return 0.0
    finally:
        signal.alarm(0)


def correctness_reward_func(prompts, completions, **kwargs):
    tests = kwargs["test"]
    rewards = []

    for completion, test_case in zip(completions, tests):
        code = extract_python_code(completion)
        print(code)
        if not code:
            rewards.append(0.0)
            continue
            
        reward = run_test_case(code, test_case)
        rewards.append(reward)

    return rewards

def format_reward_func(completions, **kwargs):
    """
    Soft reward to encourage correct formatting {Explanation ... Code}.
    Checks if ```python exists.
    """
    rewards = []
    for completion in completions:
        if "```python" in completion and "```" in completion.split("```python")[1]:
            rewards.append(0.1) 
        else:
            rewards.append(0.0)
    return rewards

def prompt_func(data):
    return [
        f"Problem Description:\n{q}\n\nPlease provide a solution in Python. Output an explanation followed by the code block wrapped in ```python ... ```."
        for q in data['problem_description'] 
    ]

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = MODEL_NAME,
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

training_args = GRPOConfig(
    output_dir=OUTPUT_DIR,
    learning_rate=1e-6,
    adam_beta1=0.9,
    adam_beta2=0.99,
    weight_decay=0.1,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    logging_steps=1,
    bf16=True, 
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_generations=4,
    max_prompt_length=512,
    max_completion_length=512,
    max_steps=500, 
    save_steps=100,
)

trainer = GRPOTrainer(
    model=model,
    reward_funcs=[correctness_reward_func, format_reward_func],
    args=training_args,
    train_dataset=dataset,
)

if __name__ == "__main__":
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
