#!/usr/bin/env python3
"""
GPT-2 Text Generation Script

This script generates text using a fine-tuned GPT-2 model.
"""

import os
import argparse
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_model_and_tokenizer(model_path):
    """Load the fine-tuned GPT-2 model and tokenizer."""
    logger.info(f"Loading model from {model_path}")

    if not os.path.exists(model_path):
        logger.error(f"Model path not found: {model_path}")
        return None, None

    try:
        tokenizer = GPT2TokenizerFast.from_pretrained(model_path)
        model = GPT2LMHeadModel.from_pretrained(model_path)

        # Ensure pad token is set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            model.resize_token_embeddings(len(tokenizer))

        logger.info("Model and tokenizer loaded successfully.")
        return model, tokenizer

    except Exception as e:
        logger.error(f"Error loading model or tokenizer: {e}")
        return None, None


def generate_text(model, tokenizer, prompt, max_length, temperature, top_k, top_p, do_sample=True):
    """Generate text using the fine-tuned GPT-2 model."""
    logger.info(f"Generating text for prompt: {prompt}")

    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(model.device)

    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            early_stopping=True
        )

    return tokenizer.decode(output[0], skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(description='Generate text using fine-tuned GPT-2')
    parser.add_argument('--model_path', type=str, default='output/model',
                        help='Path to the fine-tuned model directory')
    parser.add_argument('--prompt', type=str, default='Once upon a time',
                        help='Starting text prompt')
    parser.add_argument('--max_length', type=int, default=100,
                        help='Maximum number of tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.8,
                        help='Controls randomness (0.1-2.0)')
    parser.add_argument('--top_k', type=int, default=50,
                        help='Top-k sampling value')
    parser.add_argument('--top_p', type=float, default=0.9,
                        help='Top-p nucleus sampling value')
    parser.add_argument('--greedy', action='store_true',
                        help='Use greedy decoding instead of sampling')
    parser.add_argument('--num_samples', type=int, default=1,
                        help='Number of samples to generate')

    args = parser.parse_args()

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args.model_path)
    if model is None or tokenizer is None:
        return

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)  # type: ignore

    logger.info(f"Using device: {device}")
    for i in range(args.num_samples):
        logger.info(f"Generating sample {i + 1}/{args.num_samples}")
        generated_text = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=args.prompt,
            max_length=args.max_length,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            do_sample=not args.greedy
        )

        print(f"\n{'=' * 50}")
        print(f"Generated Text (Sample {i + 1}):")
        print(f"{'=' * 50}")
        print(generated_text)
        print(f"{'=' * 50}\n")


if __name__ == "__main__":
    main()
