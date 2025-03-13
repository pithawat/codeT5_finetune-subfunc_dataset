import os
import json
import pprint
import argparse
from datasets import Dataset, load_from_disk
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, TrainingArguments, Trainer, TrainerCallback
import numpy as np

class CustomSaveCallback(TrainerCallback):
    def __init__(self, save_dir):
        self.save_dir = save_dir
    
    def on_epoch_end(self, args, state, control, **kwargs):
        if state.epoch % 10 == 0 and state.epoch > 0:
            checkpoint_dir = os.path.join(self.save_dir, f"epoch-{int(state.epoch)}")
            kwargs["model"].save_pretrained(checkpoint_dir)
            kwargs["tokenizer"].save_pretrained(checkpoint_dir)
            print(f"  ==> Saved model and tokenizer at epoch {int(state.epoch)} to {checkpoint_dir}")

def run_training(args, model, train_data, val_data, tokenizer):
    print(f"Starting main loop")

    training_args = TrainingArguments(
        report_to='tensorboard',
        output_dir=args.save_dir,
        overwrite_output_dir=False,
        do_train=True,
        do_eval=True,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size_per_replica,
        gradient_accumulation_steps=args.grad_acc_steps,
        learning_rate=args.lr,
        weight_decay=0.05,
        warmup_steps=args.lr_warmup_steps,
        logging_dir=args.logging_dir,
        logging_first_step=True,
        logging_steps=args.log_freq,
        dataloader_drop_last=False,
        dataloader_num_workers=4,
        local_rank=args.local_rank,
        deepspeed=args.deepspeed,
        fp16=args.fp16,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        callbacks=[CustomSaveCallback(args.save_dir)],
    )

    if args.local_rank in [0, -1]:
        trainer.train()
        final_checkpoint_dir = os.path.join(args.save_dir, "final_checkpoint")
        model.save_pretrained(final_checkpoint_dir)
        tokenizer.save_pretrained(final_checkpoint_dir)
        print(f'  ==> Finished training and saved model and tokenizer to {final_checkpoint_dir}')

def load_tokenize_data(args):
    train_cache_path = args.cache_data
    val_cache_path = args.cache_data + "_val"
    
    if os.path.exists(train_cache_path) and os.path.exists(val_cache_path):
        train_data = load_from_disk(train_cache_path)
        val_data = load_from_disk(val_cache_path)
        print(f'  ==> Loaded {len(train_data)} train samples and {len(val_data)} val samples')
        tokenizer = AutoTokenizer.from_pretrained(args.load)
        return train_data, val_data, tokenizer
    else:
        print("Cache missing or incomplete, regenerating datasets...")
        if os.path.exists(train_cache_path):
            os.system(f"rm -rf {train_cache_path}")
        if os.path.exists(val_cache_path):
            os.system(f"rm -rf {val_cache_path}")
        
        with open(args.json_data, 'r', encoding="utf-8") as f:
            raw_data = json.load(f)
        
        # Split into train and validation sets
        train_size = int(0.8 * len(raw_data))
        train_raw = raw_data[:train_size]
        val_raw = raw_data[train_size:]
        
        # Convert new dataset format to a single target string
        def format_output(example):
            html_css = example["output"]["html_css"]
            js_suggestions = example["output"]["javascript_suggestions"]
            js_str = "".join(
                f"<func>{func['code']} // {func['description']}</func>"
                for func in js_suggestions
            )
            target = f"<html_css>{html_css}</html_css><js_suggestions>{js_str}</js_suggestions>"
            return {"description": example["description"], "target": target}

        train_raw_formatted = [format_output(ex) for ex in train_raw]
        val_raw_formatted = [format_output(ex) for ex in val_raw]
        
        train_dataset = Dataset.from_list(train_raw_formatted)
        val_dataset = Dataset.from_list(val_raw_formatted)
        tokenizer = AutoTokenizer.from_pretrained(args.load)
        
        def preprocess_function(examples):
            source = examples["description"]
            target = examples["target"]
            model_inputs = tokenizer(
                source,
                max_length=args.max_source_len,
                padding="max_length",
                truncation=True
            )
            labels = tokenizer(
                target,
                max_length=args.max_target_len,
                padding="max_length",
                truncation=True
            )
            model_inputs["labels"] = labels["input_ids"].copy()
            model_inputs["labels"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label]
                for label in model_inputs["labels"]
            ]
            return model_inputs

        train_data = train_dataset.map(
            preprocess_function,
            batched=True,
            remove_columns=train_dataset.column_names,
            num_proc=4,
            load_from_cache_file=False,
        )
        val_data = val_dataset.map(
            preprocess_function,
            batched=True,
            remove_columns=val_dataset.column_names,
            num_proc=4,
            load_from_cache_file=False,
        )
        
        print(f'  ==> Loaded {len(train_data)} train samples and {len(val_data)} val samples')
        train_data.save_to_disk(train_cache_path)
        val_data.save_to_disk(val_cache_path)
        print(f'  ==> Saved train data to {train_cache_path} and val data to {val_cache_path}')
        return train_data, val_data, tokenizer

def main(args):
    argsdict = vars(args)
    print(pprint.pformat(argsdict))

    with open(os.path.join(args.save_dir, "command.txt"), 'w') as f:
        f.write(pprint.pformat(argsdict))

    train_data, val_data, tokenizer = load_tokenize_data(args)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.load, cache_dir=args.cache_dir)
    print(f"  ==> Loaded model from {args.load}, model size {model.num_parameters()}")

    run_training(args, model, train_data, val_data, tokenizer)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune CodeT5+ for generating HTML/CSS and JavaScript suggestions from descriptions.")
    parser.add_argument('--data-num', default=-1, type=int)
    parser.add_argument('--max-source-len', default=50, type=int)
    parser.add_argument('--max-target-len', default=520, type=int)  # Increase if needed for longer JS
    parser.add_argument('--cache-data', default='cache_data/T5-40epoch_subfunc', type=str)
    parser.add_argument('--cache-dir', default='huggingface_cache', type=str)
    parser.add_argument('--load', default='Salesforce/codet5p-220m', type=str)
    parser.add_argument('--json-data', default='./dataset_subfunction.json.json', type=str, help='Path to the JSON dataset file')
    parser.add_argument('--epochs', default=40, type=int)
    parser.add_argument('--lr', default=5e-5, type=float)
    parser.add_argument('--lr-warmup-steps', default=50, type=int)
    parser.add_argument('--batch-size-per-replica', default=1, type=int)
    parser.add_argument('--grad-acc-steps', default=32, type=int)
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--deepspeed', default=None, type=str)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--save-dir', default="saved_models/T5-40epoch_subfunc", type=str)
    parser.add_argument('--logging-dir', default="saved_models/T5-40epoch_subfunc/logs", type=str)
    parser.add_argument('--log-freq', default=10, type=int)
    parser.add_argument('--save-freq', default=500, type=int)

    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.logging_dir, exist_ok=True)

    main(args)