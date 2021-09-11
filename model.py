from pyvi.ViTokenizer import ViTokenizer
import torch
import numpy as np

from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer

def preprocess(text, tokenized = True, lowercased = True):
    text = ViTokenizer.tokenize(text) if tokenized else text
    # text = filter_stop_words(text, stopwords)
    # text = deEmojify(text)
    text = text.lower() if lowercased else text
    return text

class BuildDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def load_model():
    phobert = AutoModelForSequenceClassification.from_pretrained(
        "model/phobert", num_labels=3)
    tokenizer_phobert = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=False)

    bert4news = AutoModelForSequenceClassification.from_pretrained(
        "model/bert4news", num_labels=3)
    tokenizer_bert4news = AutoTokenizer.from_pretrained("NlpHUST/vibert4news-base-cased", use_fast=False)

    xlm_r_aug = AutoModelForSequenceClassification.from_pretrained(
        "model/xlm-r", num_labels=3)
    tokenizer_xlm_r_aug = AutoTokenizer.from_pretrained("xlm-roberta-base", use_fast=False)

    return (phobert, tokenizer_phobert, bert4news, tokenizer_bert4news, xlm_r_aug, tokenizer_xlm_r_aug)

def sigmoid_array(x):
    return 1 / (1 + np.exp(-x))

def predict_ensembles(text, lowercased=False, tokenized=False, list_modes=None):
    labels = {
        0: "CLEAN",
        1: "OFFENSIVE",
        2: "HATE"
    }
    p_text = preprocess(text, lowercased=lowercased, tokenized=tokenized)

    phobert, tokenizer_phobert, bert4news, tokenizer_bert4news, xlm_r_aug, tokenizer_xlm_r_aug = list_modes

    X1 = tokenizer_xlm_r_aug([p_text], truncation=True, padding=True, max_length=100)
    X1 = BuildDataset(X1, [0])
    y1 = Trainer(model=xlm_r_aug).predict(X1).predictions
    y1 = sigmoid_array(y1)

    X2 = tokenizer_phobert([p_text], truncation=True, padding=True, max_length=100)
    X2 = BuildDataset(X2, [0])
    y2 = Trainer(model=phobert).predict(X2).predictions
    y2 = sigmoid_array(y2)

    X3 = tokenizer_bert4news([p_text], truncation=True, padding=True, max_length=100)
    X3 = BuildDataset(X3, [0])
    y3 = Trainer(model=bert4news).predict(X3).predictions
    y3 = sigmoid_array(y3)

    y_pred = (y1 + y2 + y3) / 3

    model_results = {
        "XLM-R": [np.around(y1, decimals=2)[0], labels[np.argmax(y1, axis=-1)[0]]],
        "PhoBERT": [np.around(y2, decimals=2)[0], labels[np.argmax(y2, axis=-1)[0]]],
        "Bert4News": [np.around(y3, decimals=2)[0], labels[np.argmax(y3, axis=-1)[0]]]
    }

    return labels[np.argmax(y_pred, axis=-1)[0]], model_results

if __name__ == "__main__":
    print(predict_ensembles("Chời ơi cái quần què dì đây", list_modes=load_model()))
