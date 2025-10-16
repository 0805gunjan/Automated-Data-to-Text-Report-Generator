import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
import torch

# ---------- CONFIG ----------
MODEL_NAME = "t5-small"          # small model, works with 1650 GPU
MAX_INPUT_LENGTH = 128
MAX_OUTPUT_LENGTH = 128
BATCH_SIZE = 4                   # lower batch size for GPU memory
EPOCHS = 3
SAVE_DIR = "./models/fine_tuned_t5"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ---------- LOAD DATA ----------
orders = pd.read_csv("data/orders.csv", parse_dates=["Order Date"])
details = pd.read_csv("data/details.csv")

# Merge CSVs
df = pd.merge(orders, details, on="Order ID")

# Fill missing dates
# Forward fill missing dates first
df['Order Date'] = df['Order Date'].ffill()

# Convert to datetime
df['Order Date'] = pd.to_datetime(df['Order Date'], errors='coerce')

# Optional: forward fill again if coercion introduced NaT
df['Order Date'] = df['Order Date'].ffill()


# Generate "input" text and "target" text
df['input'] = df.apply(
    lambda x: f"Order {x['Order ID']} on {x['Order Date'].strftime('%Y-%m-%d')} for {x['CustomerName']} in {x['City']}, {x['State']}", 
    axis=1
)

df['target'] = df.apply(lambda x: f"Amount: {x['Amount']}, Profit: {x['Profit']}, Quantity: {x['Quantity']}, Category: {x['Category']}, Payment: {x['PaymentMode']}", axis=1)

# Train-test split
train_df, val_df = train_test_split(df[['input', 'target']], test_size=0.1, random_state=42)

# Convert to Hugging Face Dataset
train_dataset = Dataset.from_pandas(train_df.reset_index(drop=True))
val_dataset = Dataset.from_pandas(val_df.reset_index(drop=True))
dataset = {"train": train_dataset, "validation": val_dataset}

# ---------- TOKENIZER ----------
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)

def preprocess(batch):
    # batch is a dict of lists
    model_inputs = tokenizer(
        batch["input"], max_length=MAX_INPUT_LENGTH, truncation=True, padding="max_length"
    )
    labels = tokenizer(
        batch["target"], max_length=MAX_OUTPUT_LENGTH, truncation=True, padding="max_length"
    )["input_ids"]
    model_inputs["labels"] = labels
    return model_inputs

tokenized_dataset = {
    split: dataset[split].map(preprocess, batched=True)
    for split in dataset
}

# ---------- MODEL ----------
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME).to(device)

# ---------- TRAINING ----------
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    save_steps=500,
    save_total_limit=2,
    logging_steps=100,
    learning_rate=5e-5,
    weight_decay=0.01,
    report_to="none",
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    tokenizer=tokenizer,
)

# Train!
trainer.train()

# Save final model
trainer.save_model(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)

print(f"Model fine-tuned and saved at {SAVE_DIR}")

