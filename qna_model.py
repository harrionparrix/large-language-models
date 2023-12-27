# Import necessary libraries
import transformers
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, TrainingArguments, Trainer
from datasets import Dataset, load_from_disk
from transformers import DataCollatorForQA
from torch.utils.data import DataLoader

# Define model and tokenizer
model_name = "gpt2"  # Change this if you want a different model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

# Load Shakespeare dataset
dataset = load_from_disk("./data/shakespeare_qa")  # Assuming you have a preprocessed QA dataset

# Create Data Collator
data_collator = DataCollatorForQA(tokenizer=tokenizer)

# Define Training Arguments
training_args = TrainingArguments(
    output_dir="./model_outputs",
    overwrite_output_dir=True,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    save_steps=1000,
    eval_steps=1000,
)

# Create Trainer and Train the Model
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    data_collator=data_collator,
)
trainer.train()

# Save the model
model.save_pretrained("./model_outputs")
tokenizer.save_pretrained("./model_outputs")

print("Model and tokenizer saved successfully!")