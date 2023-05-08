from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "bigscience/bloomz-560m"

device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model.to(device)

def calculate_probabilities(text):
    encoding = tokenizer(text, return_tensors="pt")
    encoding = encoding['input_ids'][0]
    encoding = torch.cat((torch.tensor([0]),encoding))

    preds = []
    for i in range(encoding.shape[0]-1):
        token = torch.tensor([[encoding[i]]]).to(device)
        next_token = encoding[i+1].item()

        predictions = torch.softmax(model(token).logits.squeeze(), -1)

        next_token_prediction = predictions[next_token].cpu().detach().numpy()# / torch.max(predictions).cpu().detach().numpy()
        preds.append(next_token_prediction)
    
    output_dict = {}
    for token, prob in zip(encoding[1:], preds):
        decoded_token = tokenizer.decode(token)
        output_dict[decoded_token] = prob.item()
    
    return output_dict

user = "."
while user:
    user = input()
    preds = calculate_probabilities(user)
    print(preds)


