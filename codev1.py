


# Import necessary libraries
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset

# Load tokenizer and model
model_name = "distilgpt2"  # Replace with your specific model if needed
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Ensure the tokenizer has a padding token (GPT models don't by default)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Function to load and tokenize the custom dataset
def load_custom_dataset(file_path, tokenizer, block_size=128):
    # Load the text dataset
    dataset = load_dataset("text", data_files={"train": file_path}, split="train")
    
    # Tokenize and add labels
    def tokenize_function(examples):
        outputs = tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=block_size
        )
        # Use input_ids as labels to compute the loss
        outputs["labels"] = outputs["input_ids"].copy()
        return outputs

    # Map the tokenization function to the dataset
    dataset = dataset.map(tokenize_function, batched=True)
    return dataset

# Load your custom dataset with labels
file_path = "db.txt"  # Replace with your actual file path
train_dataset = load_custom_dataset(file_path, tokenizer)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    save_steps=10_000,
    save_total_limit=2,
    logging_dir='./logs',
    logging_steps=500
)

# Initialize the Trainer with the dataset containing labels
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset
)

# Start training
trainer.train()

# Save the fine-tuned model and tokenizer
model.save_pretrained("./fine_tuned_distilgpt2")
tokenizer.save_pretrained("./fine_tuned_distilgpt2")

print("Training complete and model saved!")
