import torch.nn as nn
from transformers import MarianMTModel, MarianTokenizer
from config import parsers
import torch


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.args = parsers()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = MarianMTModel.from_pretrained(self.args.marian_model)
        self.tokenizer = MarianTokenizer.from_pretrained(self.args.marian_model)
        for param in self.model.parameters():
            param.requires_grad = True

    def forward(self, x, y):
        inputs = self.tokenizer(x, text_target=y, return_tensors="pt", padding=True)
        pred = self.model(**inputs)
        return pred
