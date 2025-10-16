from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments
from datasets import Dataset

# Dummy dataset
data = {"input": ["Order 1 for Alice"], "target": ["Order confirmed"]}
dataset = Dataset.from_dict(data)

# Model and tokenizer
model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Tokenize
def preprocess(batch):
    inputs = tokenizer(batch["input"], padding=True, truncation=True, return_tensors="pt")
    targets = tokenizer(batch["target"], padding=True, truncation=True, return_tensors="pt")
    inputs["labels"] = targets["input_ids"]
    return inputs

tokenized_dataset = dataset.map(preprocess)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=2,
    num_train_epochs=1,
    evaluation_strategy="steps",  # ✅ valid in 4.57.1
    logging_steps=10,
    save_steps=10,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

trainer.train()


