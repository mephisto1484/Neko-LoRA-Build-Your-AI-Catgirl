from datasets import load_dataset, DatasetDict
import torch
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    TrainingArguments, Trainer,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model


def load_and_split_data(json_path, test_size=0.1, val_size=0.1, seed=42):
    """加载JSON格式数据并划分训练集、验证集、测试集"""
    print("正在加载和划分数据集...")

    # 加载JSON文件
    import json
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 转换为datasets格式
    from datasets import Dataset
    full_dataset = Dataset.from_list(data)

    print(f"原始数据集样本数: {len(full_dataset)}")

    # 划分数据集: 训练集80%，验证集10%，测试集10%
    train_testval = full_dataset.train_test_split(
        test_size=test_size + val_size,
        seed=seed
    )

    # 从测试+验证集中再划分验证集和测试集
    test_val_ratio = val_size / (test_size + val_size)
    test_val = train_testval['test'].train_test_split(
        test_size=test_val_ratio,
        seed=seed
    )

    # 创建DatasetDict
    dataset_dict = DatasetDict({
        'train': train_testval['train'],
        'validation': test_val['train'],  # 这是验证集
        'test': test_val['test']  # 这是测试集
    })

    print(f"数据集划分完成:")
    print(f"  训练集: {len(dataset_dict['train'])} 样本")
    print(f"  验证集: {len(dataset_dict['validation'])} 样本")
    print(f"  测试集: {len(dataset_dict['test'])} 样本")

    # 显示样本示例
    print("\n数据集示例:")
    for i in range(min(2, len(dataset_dict['train']))):
        sample = dataset_dict['train'][i]
        print(f"样本 {i}:")
        print(f"  instruction: {sample['instruction']}")
        print(f"  output: {sample['output'][:100]}...")
        print()

    return dataset_dict

def setup_model_and_tokenizer(model_name):
    """设置模型和tokenizer，启用梯度检查点"""
    print("正在加载模型和tokenizer...")

    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 加载模型，使用bfloat16，启用梯度检查点
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )

    # 启用梯度检查点（大幅减少显存）
    model.gradient_checkpointing_enable()
    print("已启用梯度检查点")

    print(f"模型加载完成，dtype: {model.dtype}")
    return model, tokenizer

def setup_lora_model(model):
    """设置LoRA模型（16bit训练）"""
    print("正在配置LoRA...")

    lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model


def preprocess_data(dataset_dict, tokenizer):
    """预处理所有数据集（训练集、验证集、测试集）"""
    print("正在预处理数据...")

    def format_instruction(example):
        """格式化指令数据 - 适配新数据格式"""
        instruction = example['instruction']
        output = example['output']

        # 新数据格式只有instruction和output，没有input字段
        text = f"<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n{output}<|im_end|>"
        return {"text": text}

    def tokenize_function(examples):
        """tokenize函数"""
        formatted_texts = []
        for i in range(len(examples['instruction'])):
            example = {
                'instruction': examples['instruction'][i],
                'output': examples['output'][i]
            }
            formatted_text = format_instruction(example)['text']
            formatted_texts.append(formatted_text)

        # 先tokenize输入
        tokenized = tokenizer(
            formatted_texts,
            truncation=True,
            padding=False,
            max_length=2048,
            return_tensors=None
        )

        # 设置labels，对于因果语言建模，labels就是input_ids
        tokenized["labels"] = tokenized["input_ids"].copy()

        return tokenized

    # 对所有分割应用预处理
    tokenized_datasets = {}
    for split in ['train', 'validation', 'test']:
        print(f"正在处理 {split} 集...")
        tokenized_datasets[split] = dataset_dict[split].map(
            tokenize_function,
            batched=True,
            remove_columns=dataset_dict[split].column_names
        )
        print(f"  {split}集预处理完成: {len(tokenized_datasets[split])} 样本")

    return tokenized_datasets

def setup_training_args(output_dir, test_mode=False):
    """设置训练参数"""
    print("正在配置训练参数...")

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1 if not test_mode else 1,  # 降低batch size
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8 if not test_mode else 4,  # 增加梯度累积
        learning_rate=2e-4,
        num_train_epochs=3,
        logging_steps=10,
        eval_steps=250,
        save_steps=500,
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        warmup_steps=100,
        lr_scheduler_type="cosine",
        logging_dir="./logs",
        report_to=None,
        remove_unused_columns=False,
        fp16=False,
        bf16=True,
        dataloader_pin_memory=False,
        # 添加显存优化参数
        dataloader_num_workers=0,  # 减少worker数量
        ddp_find_unused_parameters=False,
        # 启用梯度检查点
        gradient_checkpointing=True,
    )

    return training_args

def train_model(model, tokenizer, tokenized_datasets, training_args):
    """训练模型（使用验证集）"""
    print("开始训练模型...")

    # 创建Trainer，传入验证集
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],  # 使用验证集
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            pad_to_multiple_of=8,
            return_tensors="pt",
            padding=True
        ),
    )

    # 开始训练
    print("开始训练...")
    train_result = trainer.train()

    # 保存训练指标（修复：直接使用trainer的log_history）
    print("训练指标:", train_result.metrics)

    return trainer


def evaluate_on_test_set(trainer, tokenized_datasets, tokenizer):
    """在测试集上评估模型"""
    print("\n在测试集上评估模型...")

    # 使用训练好的模型在测试集上评估
    eval_results = trainer.evaluate(tokenized_datasets["test"])

    print("测试集评估结果:")
    for key, value in eval_results.items():
        print(f"  {key}: {value:.4f}")

    # 生成几个测试样本的预测示例
    print("\n测试集预测示例:")
    test_samples = tokenized_datasets["test"].select(range(min(3, len(tokenized_datasets["test"]))))

    for i, sample in enumerate(test_samples):
        input_ids = torch.tensor(sample["input_ids"]).unsqueeze(0).to(trainer.model.device)

        with torch.no_grad():
            outputs = trainer.model.generate(
                input_ids=input_ids,
                max_new_tokens=4096,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )

        # 解码生成结果
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"样本 {i + 1}:")
        print(f"  生成结果: {generated_text}")
        print()


def main():
    """主函数"""
    # 配置参数
    model_name = "E:\\Models\\Qwen3-1.7B"
    json_path = "NekoQA-10K.json"  # 替换为新的JSON文件路径
    output_dir = "./test"

    # 测试模式配置
    test_mode = False  # 先用测试模式验证
    max_train_samples = 10
    max_val_samples = 5
    max_test_samples = 5

    try:
        # 1. 加载并划分数据
        dataset_dict = load_and_split_data(
            json_path,  # 使用新的JSON路径
            test_size=0.1,
            val_size=0.1,
            seed=42
        )

        # 测试模式：只使用少量数据
        if test_mode:
            print("\n=== 测试模式 ===")
            print(f"使用少量数据进行测试:")
            print(f"  训练样本: {max_train_samples}")
            print(f"  验证样本: {max_val_samples}")
            print(f"  测试样本: {max_test_samples}")

            dataset_dict['train'] = dataset_dict['train'].select(
                range(min(max_train_samples, len(dataset_dict['train']))))
            dataset_dict['validation'] = dataset_dict['validation'].select(
                range(min(max_val_samples, len(dataset_dict['validation']))))
            dataset_dict['test'] = dataset_dict['test'].select(
                range(min(max_test_samples, len(dataset_dict['test']))))

            print(f"调整后数据集大小:")
            print(f"  训练集: {len(dataset_dict['train'])} 样本")
            print(f"  验证集: {len(dataset_dict['validation'])} 样本")
            print(f"  测试集: {len(dataset_dict['test'])} 样本")

        # 2. 设置模型和tokenizer
        model, tokenizer = setup_model_and_tokenizer(model_name)

        # 3. 设置LoRA
        model = setup_lora_model(model)

        # 4. 预处理所有数据集
        tokenized_datasets = preprocess_data(dataset_dict, tokenizer)

        # 5. 设置训练参数
        training_args = setup_training_args(output_dir, test_mode=test_mode)

        # 测试模式：调整训练参数
        if test_mode:
            training_args.num_train_epochs = 1
            training_args.logging_steps = 1
            training_args.eval_steps = 2
            training_args.save_steps = 3
            print(f"测试模式训练参数: {training_args.num_train_epochs} epoch")

        # 6. 训练模型
        trainer = train_model(model, tokenizer, tokenized_datasets, training_args)

        # 7. 保存模型
        print("正在保存模型...")
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        print(f"训练完成！模型保存在: {output_dir}")

        # 8. 在测试集上评估
        evaluate_on_test_set(trainer, tokenized_datasets, tokenizer)

    except Exception as e:
        print(f"训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()