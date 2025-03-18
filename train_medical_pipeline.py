import os
import torch
import transformers
from trl import SFTTrainer
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig
from datasets import load_dataset
from datetime import datetime
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

# data formatting function
def formatting_func_medical(example):
    text = f"Question: {example['question'][0]}\nAnswer: {example['answer'][0]}<eos>"
    return [text]


# metrics function
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(logits, dim=-1)
    return {"accuracy": (predictions == labels).float().mean().item()}


# set up the environment
now = datetime.now()
time_str = now.strftime('%Y-%m-%d-%H-%M-%S')
secret_value_hf = ""
secret_value_wandb = ""
os.environ["HF_TOKEN"] = secret_value_hf
os.environ["wandb-key"] = secret_value_wandb
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
model_id = "gemma-2-2b"
device = "cuda"

# load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.padding_side = "right"

# load the data
medical_dataset = load_dataset("json", data_files="datasets/Huatuo26M-Lite/format_data.jsonl", split="train")
print("Raw datasets:")
print("Medical dataset:")
print(medical_dataset.column_names)
print("Example item:", medical_dataset[0])

# preprocess the data
medical_data = medical_dataset.map(lambda samples: tokenizer(samples["question"], samples["answer"], return_tensors="pt", padding=True), batched=True)
medical_data = medical_data.remove_columns(["label"])
medical_data = medical_data.train_test_split(test_size=0.2)

# save the data
medical_data.save_to_disk("datasets/Huatuo26M-Lite/medical_data_all")

# quantize the model to reduce its weight
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,  # 使用 4-bit 量化
    bnb_4bit_quant_type="nf4",  # 量化类型
    bnb_4bit_compute_dtype=torch.bfloat16,  # 计算时使用 bfloat16
    bnb_4bit_use_double_quant=True,  # 使用双重量化以进一步减少显存占用
)

# load the final model
model = AutoModelForCausalLM.from_pretrained(
    model_id, quantization_config=bnb_config, device_map={"": 0}
)

# define tuning parameters
lora_config = LoraConfig(
    r=8,  # 低秩矩阵的秩
    lora_alpha=32,  # 缩放因子
    target_modules=[
        "q_proj",
        "v_proj",
        "k_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],  # 目标模块
    lora_dropout=0.1,  # Dropout 概率
    bias="none",  # 是否微调偏置
    task_type="CAUSAL_LM",  # 任务类型
)

# train the model
args = transformers.TrainingArguments(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    # num_train_epochs=1,
    max_steps=100,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    output_dir="outputs/medical-fine-tuning-checkpoint" + time_str,
    optim="paged_adamw_8bit",
    save_steps=10,
    eval_steps=10,
)

trainer = SFTTrainer(
    model=model,
    train_dataset=medical_data["train"],
    args=args,
    peft_config=lora_config,
    formatting_func=formatting_func_medical,
    max_seq_length=512,
)

trainer.train()

# test the model
text = "Quote: Imagination is"
inputs = tokenizer(text, return_tensors="pt").to(device)
outputs = model.generate(**inputs, max_new_tokens=20)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

