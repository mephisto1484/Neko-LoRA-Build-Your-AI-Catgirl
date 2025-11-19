from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

# 基础模型路径和LoRA权重路径
base_model_name = "./Qwen3-1.7B"
lora_model_path = "./Qwen3-1.7B-lora-CAT-epoch3"  # 训练时保存的LoRA权重路径

# 加载tokenizer和基础模型
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    dtype=torch.bfloat16,  # 保持与训练时相同的精度
    device_map="auto",
    trust_remote_code=True
)

# 加载LoRA权重并合并
model = PeftModel.from_pretrained(
    base_model,
    lora_model_path,
    dtype=torch.bfloat16
)
model = model.merge_and_unload()  # 合并权重到基础模型

# 保存整合后的模型（可选）
merged_model_path = "./Qwen3-1.7B-cat-merged"
model.save_pretrained(merged_model_path)
tokenizer.save_pretrained(merged_model_path)

print("主人我已经准备好了！喵~")