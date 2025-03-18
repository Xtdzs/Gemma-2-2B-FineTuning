import os
import torch
import transformers
from trl import SFTTrainer
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
# multiple ways to fine-tune the model
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from datetime import datetime

# Standard Format
# {"messages": [{"role": "system", "content": "You are..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}

def formatting_func_medical(example):
    text = f"Question: {example['question'][0]}\nAnswer: {example['answer'][0]}<eos>"
    return [text]

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


system_message_medical = """You are a doctor assistant. Users will ask you questions about medical conditions and you will provide them with information about the symptoms, causes, and treatments of the condition."""
def create_conversation_medical(sample):
    return {
        "messages": [
            {"role": "system", "content": system_message_medical},
            {"role": "user", "content": sample["question"]},
            {"role": "assistant", "content": sample["answer"]}
        ]
    }

def preprocess_medical_dataset(dataset):
    dataset = dataset.map(create_conversation_medical, remove_columns=dataset.features, batched=False)
    dataset = dataset.train_test_split(test_size=0.2)
    print(dataset["train"][345]["messages"])
    return dataset

medical_dataset = load_dataset("json", data_files="datasets/Huatuo26M-Lite/format_data.jsonl", split="train")
financial_dataset = load_dataset(path='datasets/Sujet-Finance-Instruct-177k')
ruozhiba_dataset = load_dataset(path='datasets/Better-Ruozhiba')

print("Raw datasets:")
print("Medical dataset:")
print(medical_dataset.column_names)
print("Example item:", medical_dataset[0])
print("Financial dataset:")
print(financial_dataset.column_names)
print("Example item:", financial_dataset["train"][0])
print("Ruozhiba dataset:")
print(ruozhiba_dataset.column_names)
print("Example item:", ruozhiba_dataset["train"][0])

medical_data = medical_dataset.map(lambda samples: tokenizer(samples["question"], return_tensors="pt", padding=True), batched=True)
medical_data = medical_data.remove_columns(["label"])
medical_data = medical_data.train_test_split(test_size=0.2)





# Let's quantize the model to reduce its weight
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
)
# Let's load the final model
model = AutoModelForCausalLM.from_pretrained(
    model_id, quantization_config=bnb_config, device_map={"": 0}
)

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

args = transformers.TrainingArguments(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    warmup_steps=2,
    max_steps=10,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=1,
    output_dir="outputs/checkpoint-1" + time_str,
    optim="paged_adamw_8bit",
)

# Create Trainer objects that takes care of the process
trainer = SFTTrainer(
    model=model,
    train_dataset=medical_data["train"],
    test_dataset=medical_data["test"],
    args=args,
    peft_config=lora_config,
    formatting_func=formatting_func_medical,
)

# Start the training
trainer.train()


