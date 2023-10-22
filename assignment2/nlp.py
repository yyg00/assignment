import fitz
import math
import sys
import random
import torch
from torch.utils.data import Dataset, random_split, DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
import os
import numpy as np
from torch import nn
from tqdm.auto import tqdm
from sacrebleu.metrics import BLEU
from datasets import load_metric


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]
    return preds, labels

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["score"]}
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result

class TranslationDataset(Dataset):
    def __init__(self, encoded):
        self.encoded = encoded

    def __len__(self):
        return len(self.encoded["input_ids"])

    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.encoded.items()}

    
if __name__=='__main__':
    src, tgt = sys.argv[1], sys.argv[2]
    metric = load_metric("sacrebleu")
    content_zh = []
    # It's hard to correctly split contents in each page to complete sentences merely by the pdf reader packages
    # More techniques should be used for text reading, and it will be easier to align zh page and en page content
    # What's more, since the text length of each page varies, the better approach might be training the model in sentence-level
    # which also needs matching of zh and en content
    doc_z = fitz.open(src)
    content_en = []
    doc_e = fitz.open(tgt)

    for page1, page2 in zip(doc_z, doc_e):
        page_ct_list_z = page1.get_text('text').split('\n')[:-1]
        page_ct_list_e = page2.get_text('text').split('\n')[:-1]
        page_content_z = ''.join(page_ct_list_z)
        page_content_e = ''.join(page_ct_list_e)
        if len(page_content_z) > 512 or len(page_content_e) > 512:
            page_ct_list_split = max(len(page_content_z), len(page_content_e)) // 512 + 1
            len_list_z = len(page_ct_list_z) // page_ct_list_split
            len_list_e = len(page_ct_list_e) // page_ct_list_split
            for i in range(page_ct_list_split):
                ct_z = page_ct_list_z[i * len_list_z: min((i+1) * len_list_z, len(page_ct_list_z))]
                ct_e = page_ct_list_e[i * len_list_e: min((i+1) * len_list_e, len(page_ct_list_e))]
                str_ct_z = ' '.join(ct_z)
                str_ct_e = ' '.join(ct_e)
                content_zh.append(str_ct_z)
                content_en.append(str_ct_e)
        else:
            content_en.append(page_content_e)
            content_zh.append(page_content_z)
      

    translation_data = []
    for i, j in zip(content_zh, content_en):
        if len(i) <= 2 or len(j) <= 2:
            continue
        translation_data.append((i,j))

    tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-zh-en")
    inputs = [x[0] for x in translation_data]
    targets = [x[1] for x in translation_data]
    
    encoded_inputs = tokenizer(inputs, padding='max_length', truncation=True,  max_length=512, return_tensors="pt")
    encoded_targets = tokenizer(targets, padding='max_length', truncation=True,  max_length=512, return_tensors="pt")
    encoded = {
        "input_ids": encoded_inputs["input_ids"],
        "attention_mask": encoded_inputs["attention_mask"],
        "decoder_input_ids": encoded_targets["input_ids"],
        "decoder_attention_mask": encoded_targets["attention_mask"],
        "labels": encoded_targets["input_ids"]
    }
    dataset = TranslationDataset(encoded)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using {device} device')
    model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-zh-en")
    model = model.to(device)
    batch_size = 8
    args = Seq2SeqTrainingArguments(
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        save_total_limit=1,
        num_train_epochs=1,
        logging_dir='./logs',
        logging_steps=10, 
        evaluation_strategy="steps",
        eval_steps=50,
        save_steps=50,
        output_dir='./results',
        predict_with_generate=True    
    )
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    trainer = Seq2SeqTrainer(
        model,
        args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
    trainer.train()