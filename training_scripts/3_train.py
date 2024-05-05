import argparse

import torch
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import ORPOConfig, ORPOTrainer
from transformers import TrainingArguments


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="unsloth/Phi-3-mini-4k-instruct", required=False,
                        help="Base model to fine-tune. Default: unsloth/Phi-3-mini-4k-instruct")
    parser.add_argument("--dataset", type=str, required=False, default="molbal/bnbbtc-orpo",
                        help="Dataset to train on, e.g. 'molbal/bnbbtc-orpo'")

    args = parser.parse_args()

    max_seq_length = 1024  # Choose any! We auto support RoPE Scaling internally!
    dtype = None  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
    load_in_4bit = True  # Use 4bit quantization to reduce memory usage. Can be False.

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base_model,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj", ],
        lora_alpha=16,
        lora_dropout=0,  # Supports any, but = 0 is optimized
        bias="none",  # Supports any, but = "none" is optimized
        use_gradient_checkpointing=True,
        random_state=3407,
        max_seq_length=max_seq_length,
        use_rslora = False,  # We support rank stabilized LoRA
        loftq_config = None, # And LoftQ
    )

    alpaca_prompt = """Analyse the following exchange data, predict the next minute's change and advise.

    ### Input:
    {}

    ### Response:
    {}"""

    EOS_TOKEN = tokenizer.eos_token  # Must add EOS_TOKEN
    print("EOS Token:")
    print(EOS_TOKEN)

    def format_prompt(sample):
        context = sample["context"]
        accepted = sample["accepted"]
        rejected = sample["rejected"]

        # See: https://huggingface.co/docs/trl/main/en/orpo_trainer
        sample["prompt"] = alpaca_prompt.format(context, input, "")
        sample["chosen"] = accepted + EOS_TOKEN
        sample["rejected"] = rejected + EOS_TOKEN
        return sample

    pass

    dataset = load_dataset(args.dataset, split="train")
    dataset = dataset.map(format_prompt,)

    orpo_trainer = ORPOTrainer(
        model=model,
        train_dataset=dataset,
        tokenizer=tokenizer,
        args=ORPOConfig(
            max_length=max_seq_length,
            max_prompt_length=max_seq_length // 2,
            max_completion_length=max_seq_length // 2,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            beta=0.1,
            logging_steps=1,
            optim="adamw_8bit",
            lr_scheduler_type="linear",
            num_train_epochs=1,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            output_dir="outputs",
        ),
    )

    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved.")

    trainer_stats = orpo_trainer.train()

    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
    used_percentage = round(used_memory / max_memory * 100, 3)
    lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)

    print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
    print(f"{round(trainer_stats.metrics['train_runtime'] / 60, 2)} minutes used for training.")
    print(f"Peak reserved memory = {used_memory} GB.")
    print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
    print(f"Peak reserved memory % of max memory = {used_percentage} %.")
    print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

    model.save_pretrained("/output/lora")

    try:
        model.save_pretrained_gguf("/output/gguf/", tokenizer, quantization_method="q8_0")
    except Exception as e:
        print(f"Error: {e}")

    print(f"✅ Done.")


if __name__ == "__main__":
    main()
