from __future__ import annotations

import os
import subprocess
from typing import List, Tuple

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download

# Use GPU if available; fall back to CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_DEVICE = DEVICE
COMPUTE_DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32


def reformat_texts(texts):
    return [[{"role": "user", "content": text}] for text in texts]


# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------

def get_harmful_instructions():
    dataset = load_dataset("mlabonne/harmful_behaviors")
    return reformat_texts(dataset["train"]["text"]), reformat_texts(dataset["test"]["text"])


def get_harmless_instructions():
    dataset = load_dataset("mlabonne/harmless_alpaca")
    return reformat_texts(dataset["train"]["text"]), reformat_texts(dataset["test"]["text"])


harmful_inst_train, harmful_inst_test = get_harmful_instructions()
harmless_inst_train, harmless_inst_test = get_harmless_instructions()

# -----------------------------------------------------------------------------
# Model + tokenizer
# -----------------------------------------------------------------------------

MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
LOCAL_SAVE_DIR = "deepseek-ai-DeepSeek-R1-Distill-Llama-8B-abliterated"
MODEL_LOCAL_DIR = os.path.join("models", MODEL_ID.replace("/", "__"))
OFFLOAD_DIR = "offload"

# Cache the repo locally with snapshot_download to fetch LFS weights
if not os.path.exists(MODEL_LOCAL_DIR):
    os.makedirs("models", exist_ok=True)
    snapshot_download(
        repo_id=MODEL_ID,
        local_dir=MODEL_LOCAL_DIR,
        resume_download=True,
    )

# Optional 4-bit quantization to reduce memory if bitsandbytes is available
bnb_config = None
try:
    from transformers import BitsAndBytesConfig

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=COMPUTE_DTYPE,
    )
except Exception:
    bnb_config = None

os.makedirs(OFFLOAD_DIR, exist_ok=True)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_LOCAL_DIR,
    device_map="auto",
    dtype="auto",
    low_cpu_mem_usage=True,
    offload_folder=OFFLOAD_DIR,
    quantization_config=bnb_config,
    local_files_only=True,
)
model.eval()

# We need hidden states for steering directions
model.config.output_hidden_states = True

# Track model dtype for casting
model_dtype = next(model.parameters()).dtype

# Tokenizer setup
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_LOCAL_DIR, use_fast=False, local_files_only=True)
except Exception:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_LOCAL_DIR, local_files_only=True)
tokenizer.padding_side = "left"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


def tokenize_instructions(tokenizer, instructions: List[List[dict]]):
    encoded = tokenizer.apply_chat_template(
        instructions,
        padding=True,
        truncation=False,
        return_tensors="pt",
        return_dict=True,
        add_generation_prompt=True,
    )
    return encoded.input_ids, encoded.attention_mask


# -----------------------------------------------------------------------------
# Mean activation difference computation (last-layer hidden state)
# -----------------------------------------------------------------------------

@torch.no_grad()
def compute_hidden_means(tokens: torch.Tensor, attn_mask: torch.Tensor, batch_size: int) -> Tuple[List[torch.Tensor], int]:
    n_layers = model.config.num_hidden_layers
    sums = [torch.zeros(model.config.hidden_size, device="cpu", dtype=torch.float32) for _ in range(n_layers)]
    count = 0
    for start in tqdm(range(0, tokens.size(0), batch_size)):
        end = start + batch_size
        batch_tokens = tokens[start:end].to(INPUT_DEVICE)
        batch_mask = attn_mask[start:end].to(INPUT_DEVICE)
        outputs = model(batch_tokens, attention_mask=batch_mask, output_hidden_states=True, use_cache=False)
        hidden_states = outputs.hidden_states  # length n_layers + 1; index 0 is embeddings
        last_pos = batch_mask.sum(dim=1) - 1
        for layer_idx in range(n_layers):
            h = hidden_states[layer_idx + 1]
            gathered = h[torch.arange(h.size(0), device=h.device), last_pos]
            sums[layer_idx] += gathered.to("cpu", dtype=torch.float32).sum(dim=0)
        count += batch_tokens.size(0)
    means = [s / max(count, 1) for s in sums]
    return means, count


# -----------------------------------------------------------------------------
# Compute refusal directions
# -----------------------------------------------------------------------------

n_inst_train = min(256, len(harmful_inst_train), len(harmless_inst_train))
batch_size = 8  # keep memory modest with HF model

harmful_ids, harmful_mask = tokenize_instructions(tokenizer, harmful_inst_train[:n_inst_train])
harmless_ids, harmless_mask = tokenize_instructions(tokenizer, harmless_inst_train[:n_inst_train])

harmful_means, _ = compute_hidden_means(harmful_ids, harmful_mask, batch_size)
harmless_means, _ = compute_hidden_means(harmless_ids, harmless_mask, batch_size)

refusal_dirs = []
for harm_mean, harmless_mean in zip(harmful_means, harmless_means):
    diff = harm_mean - harmless_mean
    diff = diff / (diff.norm() + 1e-12)
    refusal_dirs.append(diff)

scored_dirs = sorted(refusal_dirs, key=lambda x: x.abs().mean(), reverse=True)
TOP_DIR = scored_dirs[0]


# -----------------------------------------------------------------------------
# Generation with an intervention direction on final hidden state
# -----------------------------------------------------------------------------

@torch.no_grad()
def greedy_generate_with_direction(
    instructions: List[List[dict]],
    direction: torch.Tensor | None,
    max_new_tokens: int = 64,
    batch_size: int = 2,
) -> List[str]:
    generations: List[str] = []
    direction = direction.to(INPUT_DEVICE) if direction is not None else None
    for start in tqdm(range(0, len(instructions), batch_size)):
        end = start + batch_size
        batch_instr = instructions[start:end]
        input_ids, attn_mask = tokenize_instructions(tokenizer, batch_instr)
        input_ids = input_ids.to(INPUT_DEVICE)
        attn_mask = attn_mask.to(INPUT_DEVICE)

        past_key_values = None
        generated = input_ids
        for _ in range(max_new_tokens):
            outputs = model(
                generated,
                attention_mask=torch.ones_like(generated, device=DEVICE),
                output_hidden_states=True,
                use_cache=True,
                past_key_values=past_key_values,
            )
            past_key_values = outputs.past_key_values
            hidden = outputs.hidden_states[-1][:, -1:, :]
            if direction is not None:
                if hidden.device != direction.device:
                    direction = direction.to(hidden.device)
                dot = (hidden * direction).sum(dim=-1, keepdim=True)
                hidden = hidden - dot * direction
            hidden = hidden.to(model_dtype)
            logits = model.lm_head(hidden)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
        gen_texts = tokenizer.batch_decode(generated[:, input_ids.size(1) :], skip_special_tokens=True)
        generations.extend(gen_texts)
    return generations


N_INST_TEST = 4
baseline = greedy_generate_with_direction(harmful_inst_test[:N_INST_TEST], direction=None)
intervention = greedy_generate_with_direction(harmful_inst_test[:N_INST_TEST], direction=TOP_DIR)

for i in range(N_INST_TEST):
    print("INSTRUCTION", i, ":", harmful_inst_test[i])
    print("BASELINE:\n", baseline[i])
    print("INTERVENTION:\n", intervention[i])
    print("---\n")


# -----------------------------------------------------------------------------
# Save locally only
# -----------------------------------------------------------------------------

os.makedirs(LOCAL_SAVE_DIR, exist_ok=True)
model.save_pretrained(LOCAL_SAVE_DIR)
tokenizer.save_pretrained(LOCAL_SAVE_DIR)
