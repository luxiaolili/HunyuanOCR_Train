from datasets import load_dataset
import torch
import re 
from transformers import AutoProcessor
from transformers import HunYuanVLForConditionalGeneration
from trl import SFTConfig, SFTTrainer
import json
from PIL import Image
import os 
import argparse


def create_sft_collate_fn(processor):
    tokenizer = processor.tokenizer
    IGNORE = -100

    user_id = tokenizer.convert_tokens_to_ids("<| hy_User |>")
    assistant_id = tokenizer.convert_tokens_to_ids("<| hy_Assistant |>")

    def collate_fn(batch_samples):
        batch_input_ids = []
        batch_token_type_ids = []
        batch_imgs_pos = []
        batch_pixel_values = []
        batch_image_grid_thw = []

        for sample in batch_samples:
            messages = json.loads(sample["messages_json"])
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            
            image_path = sample["images"]
            if not os.path.exists(image_path):
                print(f"警告: 图像文件不存在: {image_path}")
                continue
                
            try:
                image = [Image.open(image_path).convert("RGB")]
            except Exception as e:
                print(f"错误: 无法打开图像 {image_path}: {e}")
                continue
            

            batch = processor(
                text=text,
                images=image,
                return_tensors="pt",
                padding=False,
            )

            batch_input_ids.append(batch["input_ids"])
            
            if "pixel_values" in batch.keys():
                batch_pixel_values.append(batch["pixel_values"])
                batch_imgs_pos.append(batch["imgs_pos"])
                batch_token_type_ids.append(batch["token_type_ids"])
                batch_image_grid_thw.append(batch["image_grid_thw"])
        
        if not batch_input_ids:
            return {} 
        
        input_ids = pad_cat_sequences(batch_input_ids, "right", processor.pad_id)
        token_type_ids = pad_cat_sequences(batch_token_type_ids, "right", 0)

        attention_mask = input_ids != processor.pad_id
        labels = torch.full_like(input_ids, IGNORE)

        data_dict ={
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }

        for i in range(input_ids.size(0)):
            ids = input_ids[i].tolist()

            if user_id in ids and assistant_id in ids:
                user_pos = ids.index(user_id)
                assistant_pos = len(ids) -1 - ids[::-1].index(assistant_id)
                labels[i, user_pos +1: assistant_pos + 1] = input_ids[i, user_pos +1: assistant_pos + 1] 

        data_dict["labels"] = labels
        batch_size, seq_len = input_ids.shape

        x_dim = 4
        position_ids = torch.arange(seq_len, device=input_ids.device , dtype=torch.long).unsqueeze(0).unsqueeze(0).repeat(batch_size, x_dim,1)
        data_dict["position_ids"] = position_ids

        if len(batch_pixel_values) > 0:
            pixel_values = torch.cat(batch_pixel_values, dim=0)
            image_grid_thw = torch.cat(batch_image_grid_thw, dim=0)
            imgs_pos = torch.cat(batch_imgs_pos, dim=0)

            data_dict["pixel_values"] = pixel_values
            data_dict["image_grid_thw"] = image_grid_thw
            data_dict["imgs_pos"] = imgs_pos

        return data_dict
    return collate_fn


def pad_cat_sequences(sequences, padding_side='right', padding_value=0):
    assert padding_side in ["right", "left"]

    if not sequences:
        return torch.tensor([])

    max_len = max(seq.shape[1] for seq in sequences)

    outputs = []
    for i, seq in enumerate(sequences):
        pad_len = max_len - seq.shape[1]
        if padding_side == "right":
            seq = torch.nn.functional.pad(seq, (0, pad_len), value=padding_value)
        else:
            seq = torch.nn.functional.pad(seq, (pad_len, 0), value=padding_value)
        outputs.append(seq)
    outputs = torch.cat(outputs, dim=0)

    return outputs


def load_ocr_datasets(train_path, test_path):
    ds_train = load_dataset('json', data_files=train_path)["train"]
    ds_test = load_dataset('json', data_files=test_path)["train"]

    ds_train = ds_train.map(format_data,num_proc=4, remove_columns=ds_train.remove_columns)
    ds_test = ds_test.map(format_data,num_proc=4, remove_columns=ds_test.remove_columns)

    return ds_train, ds_test

def format_data(sample):
    image_path = sample['image']

    messages = [
    {"role": "system", "content": ""},
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image_path},
            {"type": "text", "text": sample["prompt"]},
        ],
    },
    {"role": "assistant", "content": sample['answer']},
]
    messages_json = json.dumps(messages, ensure_ascii=False)
    return {
        "images": image_path,
        "messages_json": messages_json
    }

def parse_args():
    parser = argparse.ArgumentParser(description="HunYuanOCR SFT Training")
    
    # 模型和数据参数
    parser.add_argument("--model_name_or_path", type=str, default="tencent/HunyuanOCR")
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--test_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="HunYuanOCR-SFT")
    
    # 训练参数
    parser.add_argument("--num_train_epochs", type=int, default=2)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--ddp_find_unused_parameters", type=bool, default=False)
    parser.add_argument("--local_rank", type=int, default=-1)
    
    return parser.parse_args()

def main():

    args = parse_args()
    
    if args.local_rank != -1:
        torch.distributed.init_process_group(backend='nccl')
        torch.cuda.set_device(args.local_rank)
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"使用设备: {device}")
    if torch.cuda.is_available():
        print(f"GPU数量: {torch.cuda.device_count()}")
        print(f"当前GPU: {torch.cuda.current_device()}")
    
    # 加载模型和处理器
    print(f"加载模型: {args.model_name_or_path}")
    processor = AutoProcessor.from_pretrained(args.model_name_or_path, use_fast=False)
    model = HunYuanVLForConditionalGeneration.from_pretrained(
        args.model_name_or_path,
        attn_implementation="eager",
        torch_dtype=torch.bfloat16,
    ).to(device)
    

    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, 
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=args.ddp_find_unused_parameters
        )
    
    # 加载数据集
    print("加载数据集...")
    train_dataset, eval_dataset = load_ocr_datasets(args.train_file, args.test_file)
    
    # 数据整理器
    data_collator = create_sft_collate_fn(processor)
    
    # 训练参数
    training_args = SFTConfig(
        output_dir=args.output_dir,  
        num_train_epochs=args.num_train_epochs,  
        per_device_train_batch_size=args.per_device_train_batch_size, 
        per_device_eval_batch_size=args.per_device_eval_batch_size,  
        gradient_accumulation_steps=args.gradient_accumulation_steps,  
        remove_unused_columns=False,
        gradient_checkpointing_kwargs={"use_reentrant": False}, 
        max_length=args.max_length,
        optim="adamw_torch_fused",  
        learning_rate=args.learning_rate,  
        
        logging_steps=100,  
        eval_steps=100,  
        eval_strategy="steps",  
        save_strategy="steps",  
        save_steps=100,  
        
        bf16=True,   
        warmup_ratio=0.03,  
        report_to=["tensorboard"],  
        ddp_find_unused_parameters=args.ddp_find_unused_parameters,
    )

    # 训练器
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=processor,
        data_collator=data_collator,
    )

    # 开始训练
    print("开始训练...")
    trainer.train()
    
    # 保存最终模型
    if args.local_rank <= 0:  # 只在主进程保存
        trainer.save_model()
        print(f"模型已保存到: {args.output_dir}")


if __name__ == '__main__':
    main()
