# GPT-2 Text Generator

A Python project for fine-tuning GPT-2 models on custom text data and generating text using the fine-tuned model.

## Project Structure

```
gpt2-text-generator/
│
├── dataset/
│   └── custom_data.txt           # Your custom training text (e.g., Shakespeare, news, movie dialogues)
│
├── output/
│   └── model/                    # Fine-tuned model will be saved here
│
├── scripts/
│   ├── train.py                  # Script to fine-tune GPT-2
│   └── generate.py               # Script to generate text using fine-tuned GPT-2
│
├── requirements.txt             # Dependencies
└── README.md                    # Project instructions
```

## Setup

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare Your Dataset**
   - Place your custom text data in `dataset/custom_data.txt`
   - The text should be in plain text format
   - You can use any text: Shakespeare, news articles, movie dialogues, etc.

## Usage

### Training (Fine-tuning)

1. **Prepare your dataset** in `dataset/custom_data.txt`

2. **Run the training script**:
   ```bash
   python scripts/train.py
   ```

   Optional parameters:
   - `--model_name`: GPT-2 model variant (default: "gpt2")
   - `--epochs`: Number of training epochs (default: 3)
   - `--batch_size`: Batch size for training (default: 4)
   - `--learning_rate`: Learning rate (default: 5e-5)
   - `--max_length`: Maximum sequence length (default: 512)

   Example:
   ```bash
   python scripts/train.py --model_name gpt2-medium --epochs 5 --batch_size 2
   ```

### Text Generation

1. **Generate text using your fine-tuned model**:
   ```bash
   python scripts/generate.py
   ```

   Optional parameters:
   - `--model_path`: Path to your fine-tuned model (default: "output/model")
   - `--prompt`: Starting text for generation (default: "Once upon a time")
   - `--max_length`: Maximum length of generated text (default: 100)
   - `--temperature`: Sampling temperature (default: 0.8)
   - `--top_k`: Top-k sampling (default: 50)
   - `--top_p`: Top-p (nucleus) sampling (default: 0.9)

   Example:
   ```bash
   python scripts/generate.py --prompt "The future of AI" --max_length 200 --temperature 0.9
   ```

## Model Variants

You can use different GPT-2 model sizes:
- `gpt2`: 124M parameters (fastest, least memory)
- `gpt2-medium`: 355M parameters
- `gpt2-large`: 774M parameters
- `gpt2-xl`: 1.5B parameters (slowest, most memory)

## Tips

1. **Dataset Size**: For good results, aim for at least 1MB of text data
2. **Training Time**: Larger models take longer to train. Start with `gpt2` for quick experiments
3. **Memory**: Use smaller batch sizes if you run out of GPU memory
4. **Quality**: More training data and longer training generally produce better results

## Example Dataset

You can use any text file. Here's an example of what to put in `dataset/custom_data.txt`:

```
To be, or not to be, that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles
And by opposing end them. To die—to sleep,
No more; and by a sleep to say we end
The heart-ache and the thousand natural shocks
That flesh is heir to: 'tis a consummation
Devoutly to be wish'd. To die, to sleep;
To sleep, perchance to dream—ay, there's the rub:
For in that sleep of death what dreams may come,
When we have shuffled off this mortal coil,
Must give us pause—there's the respect
That makes calamity of so long life.
```

## Troubleshooting

- **CUDA out of memory**: Reduce batch size or use a smaller model
- **Training too slow**: Use a smaller model or reduce sequence length
- **Poor generation quality**: Increase training data, train for more epochs, or use a larger model 