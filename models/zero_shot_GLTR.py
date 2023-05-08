from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "bigscience/bloomz-560m"

device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model.to(device)

def calculate_probabilities(text):
    encoding = tokenizer(text, return_tensors="pt")
    encoding = encoding['input_ids']
    encoding = torch.cat((torch.tensor([11]),encoding[0])).reshape(1, -1)
    encoding = encoding.to(device)

    out = model(encoding).logits.squeeze()
    predictions = torch.softmax(out, -1).cpu().detach().numpy()

    output_dict = {}
    for token, prob in zip(encoding[0,1:], predictions):
        decoded_token = tokenizer.decode(token)
        output_dict[decoded_token] = prob[token]
    output_dict
    
    return output_dict

user = "."
while user:
    user = input()
    preds = calculate_probabilities(user)
    print(preds)


