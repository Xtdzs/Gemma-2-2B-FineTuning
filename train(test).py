# Loading the model from HF Hub
import os
import torch
import transformers
from trl import SFTTrainer
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
# multiple ways to fine-tune the model
from peft import LoraConfig, get_peft_model
from datetime import datetime
now = datetime.now()
time_str = now.strftime('%Y-%m-%d-%H-%M-%S')

secret_value_hf = ""
secret_value_wandb = ""
os.environ["HF_TOKEN"] = secret_value_hf
os.environ["wandb-key"] = secret_value_wandb
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
model_id = "gemma-2-2b"
device = "cuda:0"

# load the tokenizer first
tokenizer = AutoTokenizer.from_pretrained(model_id)
# Let's quantize the model to reduce its weight
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
)
# Let's load the final model
model = AutoModelForCausalLM.from_pretrained(
    model_id, quantization_config=bnb_config, device_map={"": 0}
)

#
# prompt = "My favourite color is"
# inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
# outputs = model.generate(inputs, max_new_tokens=20)
# text = tokenizer.decode(outputs[0], skip_special_tokens=True)
# print(text)

# Load the dataset
from datasets import load_dataset
data = load_dataset(path='english_quotes')
print("Example item:", data["train"][0])

# Tokenize the quotes
data = data.map(lambda samples: tokenizer(samples["quote"]), batched=True)

# Define tuning parameters
lora_config = LoraConfig(
    r=8,
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj",
        "o_proj",
        "k_proj",
        "v_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
)

model = get_peft_model(model, lora_config)

def formatting_func(example):
    text = f"Quote: {example['quote'][0]}\nAuthor: {example['author'][0]}<eos>"
    return [text]

# Create Trainer objects that takes care of the process
trainer = transformers.Trainer(
    model=model,
    train_dataset=data["train"],
    args=transformers.TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=32,
        warmup_steps=10,
        max_steps=1000,
        learning_rate=3e-4,
        fp16=True,
        logging_steps=1,
        output_dir="outputs/checkpoint-1" + time_str,
        optim="paged_adamw_8bit",
        save_strategy='steps',
        save_steps=200,
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

# Start the training
trainer.train()

# Save the model
trainer.save_model("outputs/")

