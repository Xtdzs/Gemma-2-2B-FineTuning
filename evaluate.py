import os
import torch
import math
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# 设置设备
device = "cuda" if torch.cuda.is_available() else "cpu"

# 评估模型的不同 checkpoint
model_path_list = [
    "outputs/ruozhiba-fine-tuning-checkpoint-2025-03-18-23-19-23/checkpoint-10",
    "outputs/ruozhiba-fine-tuning-checkpoint-2025-03-18-23-19-23/checkpoint-20",
    "outputs/ruozhiba-fine-tuning-checkpoint-2025-03-18-23-19-23/checkpoint-30",
    "outputs/ruozhiba-fine-tuning-checkpoint-2025-03-18-23-19-23/checkpoint-40",
    "outputs/ruozhiba-fine-tuning-checkpoint-2025-03-18-23-19-23/checkpoint-50",
    "outputs/ruozhiba-fine-tuning-checkpoint-2025-03-18-23-19-23/checkpoint-60",
    "outputs/ruozhiba-fine-tuning-checkpoint-2025-03-18-23-19-23/checkpoint-70",
    "outputs/ruozhiba-fine-tuning-checkpoint-2025-03-18-23-19-23/checkpoint-80",
    "outputs/ruozhiba-fine-tuning-checkpoint-2025-03-18-23-19-23/checkpoint-90",
    "outputs/ruozhiba-fine-tuning-checkpoint-2025-03-18-23-19-23/checkpoint-100",
    "outputs/ruozhiba-fine-tuning-checkpoint-2025-03-18-23-19-23/checkpoint-110",
    "outputs/ruozhiba-fine-tuning-checkpoint-2025-03-18-23-19-23/checkpoint-120",
    "outputs/ruozhiba-fine-tuning-checkpoint-2025-03-18-23-19-23/checkpoint-130",
    "outputs/ruozhiba-fine-tuning-checkpoint-2025-03-18-23-19-23/checkpoint-140",
    "outputs/ruozhiba-fine-tuning-checkpoint-2025-03-18-23-19-23/checkpoint-147",
]

# 加载测试集
ruozhiba_data = load_dataset("json", data_files="datasets/Better-Ruozhiba/ruozhiba_qa.json", split="train")

# 评估工具
rouge = Rouge()


# 计算 PPL（困惑度）
def compute_perplexity(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss
        ppl = math.exp(loss.item())  # 计算困惑度
    return ppl


# 遍历多个 checkpoint 进行评估
for model_path in model_path_list:
    print(f"\n===== 评估模型: {model_path} =====")

    # 加载模型和 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path).to(device)

    predictions = []
    references = []
    ppl_scores = []

    # 遍历测试集
    for i in range(5):
        item = ruozhiba_data[i]
        instruction = item["instruction"]
        reference = item["output"]

        # 让模型生成回答
        inputs = tokenizer("Instruction: " + instruction, return_tensors="pt").to(device)
        output_tokens = model.generate(**inputs, max_new_tokens=128)
        prediction = tokenizer.decode(output_tokens[0], skip_special_tokens=True)

        # 存储结果
        predictions.append(prediction)
        references.append(reference)

        # 计算 PPL
        ppl_score = compute_perplexity(model, tokenizer, prediction)
        ppl_scores.append(ppl_score)

    # 计算 BLEU
    bleu_scores = [sentence_bleu([ref.split()], pred.split()) for ref, pred in zip(references, predictions)]
    avg_bleu = sum(bleu_scores) / len(bleu_scores)

    # 计算 ROUGE
    rouge_scores = rouge.get_scores(predictions, references, avg=True)

    # 计算平均 PPL
    average_ppl = sum(ppl_scores) / len(ppl_scores)

    # 打印评估结果
    print(f"BLEU Score: {avg_bleu}")
    print(f"ROUGE Scores: {rouge_scores}")
    print(f"Average Perplexity Score: {average_ppl}")
