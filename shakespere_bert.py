#pip install transformers
from transformers import AutoTokenizer
import torch

from torch.utils.data import TensorDataset, DataLoader
import os
import random


from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup

random.seed(42)
torch.manual_seed(42)

with open('./data/shakes.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# initialize GPT2 tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# tokenize text and convert to torch tensors
#input_ids = tokenizer.encode(text, return_tensors='pt').to(device)
input_ids = tokenizer.encode(text, return_tensors='pt', max_length=512, truncation=True).to(device)

# set training parameters
train_batch_size = 4
num_train_epochs = 3
learning_rate = 5e-5

# initialize optimizer and scheduler
optimizer = AdamW(model.parameters(), lr=learning_rate)
total_steps = len(input_ids) * num_train_epochs // train_batch_size
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

# train the model
model.train()
for epoch in range(num_train_epochs):
    epoch_loss = 0.0
    for i in range(0, len(input_ids)-1, train_batch_size):
        # slice the input ids tensor to get the current batch
        batch_input_ids = input_ids[i:i+train_batch_size]
        # create shifted labels for each input in the batch
        batch_labels = batch_input_ids.clone()
        batch_labels[:, :-1] = batch_labels[:, 1:]
        # set label ids to -100 for padded tokens
        batch_labels[batch_labels == tokenizer.pad_token_id] = -100
        # clear gradients
        optimizer.zero_grad()
        # forward pass
        outputs = model(input_ids=batch_input_ids, labels=batch_labels)
        loss = outputs[0]
        # backward pass
        loss.backward()
        epoch_loss += loss.item()
        # clip gradients to prevent exploding gradients problem
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        # update parameters
        optimizer.step()
        scheduler.step()
    print('Epoch: {}, Loss: {:.4f}'.format(epoch+1, epoch_loss/len(input_ids)))

# save the trained model
output_dir = './results/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)