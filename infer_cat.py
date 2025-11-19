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

# 加载LoRA权重
model = PeftModel.from_pretrained(
    base_model,
    lora_model_path,
    dtype=torch.bfloat16
)

# 如果你想要合并LoRA权重到基础模型（可选，会创建新模型）
# model = model.merge_and_unload()

print("主人我已经准备好了！喵~")

# 准备模型输入
prompt = "如果我自杀后逃之夭夭，我算被害者还是通缉犯？"
while True:
    print("主人说什么喵——")
    prompt=input()
    messages = [
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False  # Switches between thinking and non-thinking modes. Default is True.
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # 进行文本生成
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=8192,  # 减少生成长度用于测试，可以按需调整
        temperature=0.9,     # 添加温度控制
        do_sample=True,      # 启用采样
        pad_token_id=tokenizer.eos_token_id
    )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

    # 解析thinking内容
    try:
        # rindex finding 151668 (</think>)
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0

    thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
    content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

    # print("你的问题是：", prompt)
    # print("thinking content:", thinking_content)
    print("主人的猫娘——")
    print(content)
    print()