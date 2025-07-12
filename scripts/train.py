#!/usr/bin/env python3
"""
GPT-2 Fine-tuning Script

This script fine-tunes a GPT-2 model on custom text data.
"""

import os
import argparse
import logging
from datasets import Dataset
from transformers import (
    GPT2LMHeadModel,
    GPT2TokenizerFast,
)
from transformers.data.data_collator import DataCollatorForLanguageModeling
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_dataset(file_path, chunk_size=512):
    """Load and prepare the dataset from a text file."""
    logger.info(f"Loading dataset from {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Split text into chunks
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    
    logger.info(f"Split text into {len(chunks)} chunks of ~{chunk_size} chars")
    return Dataset.from_dict({'text': chunks})


def tokenize_function(examples, tokenizer, max_length):
    """Tokenize the text examples."""
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=max_length)


def main():
    parser = argparse.ArgumentParser(description='Fine-tune GPT-2 on custom text data')
    parser.add_argument('--model_name', type=str, default='gpt2', help='GPT-2 variant (gpt2, gpt2-medium, etc.)')
    parser.add_argument('--dataset_path', type=str, default='dataset/custom_data.txt', help='Path to training dataset')
    parser.add_argument('--output_dir', type=str, default='output/model', help='Directory to save fine-tuned model')
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Training batch size per device')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--max_length', type=int, default=512, help='Max sequence length for each chunk')
    parser.add_argument('--save_steps', type=int, default=500, help='Checkpoint saving frequency')
    parser.add_argument('--eval_steps', type=int, default=500, help='Evaluation frequency')
    parser.add_argument('--warmup_steps', type=int, default=100, help='Number of warmup steps')
    parser.add_argument('--logging_steps', type=int, default=100, help='Logging frequency')
    args = parser.parse_args()

    # Check dataset file
    if not os.path.exists(args.dataset_path):
        logger.error(f"Dataset file not found: {args.dataset_path}")
        return

    os.makedirs(args.output_dir, exist_ok=True)

    # Load tokenizer & model
    logger.info(f"Loading model and tokenizer: {args.model_name}")
    tokenizer = GPT2TokenizerFast.from_pretrained(args.model_name)
    model = GPT2LMHeadModel.from_pretrained(args.model_name)

    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.resize_token_embeddings(len(tokenizer))

    # Load and tokenize dataset
    raw_dataset = load_dataset(args.dataset_path, chunk_size=args.max_length)
    logger.info("Tokenizing dataset...")
    tokenized_dataset = raw_dataset.map(
        lambda x: tokenize_function(x, tokenizer, args.max_length),
        batched=True,
        remove_columns=raw_dataset.column_names
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

        # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        save_steps=args.save_steps,
        save_total_limit=2,
        prediction_loss_only=True,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        greater_is_better=False,
        report_to=None,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        eval_dataset=tokenized_dataset,
        data_collator=data_collator
    )

    logger.info("Starting training...")
    trainer.train()

    logger.info(f"Saving model to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    logger.info("Training completed!")


if __name__ == "__main__":
    main()
