from transformers import BertForSequenceClassification, BertTokenizerFast
import torch

model_path = './my_finetuned_bert'
model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizerFast.from_pretrained(model_path)

def classify_text(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=128)
    model.eval()

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(outputs.logits, dim=1)
        predicted_class = torch.argmax(logits, dim=1).item()

    if predicted_class == 0:
        return "Human", probs[0].tolist()[0]
    else:
        return "AI", probs[0].tolist()[1]