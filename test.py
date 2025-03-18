from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from datasets import load_dataset
import torch

# device = "cuda" if torch.cuda.is_available() else "cpu"
# tokenizer = AutoTokenizer.from_pretrained("outputs/ruozhiba-fine-tuning-checkpoint-2025-03-18-23-19-23/checkpoint-147")
# model = AutoModelForCausalLM.from_pretrained("outputs/ruozhiba-fine-tuning-checkpoint-2025-03-18-23-19-23/checkpoint-147")

tokenizer = AutoTokenizer.from_pretrained("gemma-2-2b")
model = AutoModelForCausalLM.from_pretrained("gemma-2-2b")

ruozhiba_data = load_dataset("json", data_files="datasets/Better-Ruozhiba/ruozhiba_qa.json", split="train")
testset = ruozhiba_data
text = "一个人说: " + testset[-2]["instruction"] + "。\n面对这句话做出的回应应该是: "
inputs = tokenizer(text, return_tensors="pt")
model.to(torch.bfloat16)
outputs = model.generate(**inputs, max_new_tokens=64)
print("<Input text>:")
print(text)
print("<Output text>:")
print(tokenizer.decode(outputs[0], skip_special_tokens=True))