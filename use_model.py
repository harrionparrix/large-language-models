from transformers import AutoTokenizer, GPT2LMHeadModel
import torch
# Load the model and tokenizer
model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

test_model = GPT2LMHeadModel.from_pretrained("./results/")
test_tokenizer = AutoTokenizer.from_pretrained("./results/")

# Encode a text inputs
text = "Abraham Lincoln was "

num_predictions = 20

# GPT2 LLM
indexed_tokens = tokenizer.encode(text)
tokens_tensor = torch.tensor([indexed_tokens])
for _ in range(num_predictions):
    with torch.no_grad():
        outputs = model(tokens_tensor)
        predictions = outputs.logits

    # Get the predicted next sub-word
    predicted_index = torch.argmax(predictions[0, -1, :]).item()
    indexed_tokens.append(predicted_index)
    tokens_tensor = torch.tensor([indexed_tokens])

    predicted_text = tokenizer.decode(indexed_tokens)

print("GPT2 : \n"+predicted_text)
print("\n -------------------------------------------------------------------------------------------------------")
# Test LLM
test_indexed_tokens = test_tokenizer.encode(text)
test_tokens_tensor = torch.tensor([test_indexed_tokens])
for _ in range(num_predictions):
    with torch.no_grad():
        outputs = test_model(test_tokens_tensor)
        predictions = outputs.logits

    # Get the predicted next sub-word
    predicted_index = torch.argmax(predictions[0, -1, :]).item()
    test_indexed_tokens.append(predicted_index)
    test_tokens_tensor = torch.tensor([indexed_tokens])

    predicted_text_test = tokenizer.decode(indexed_tokens)

print("Test : \n"+predicted_text_test)
