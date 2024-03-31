# coding: UTF-8
import time
import torch
import numpy as np

from configs.bert_cnn_config import *
from dataset.bert_cnn_dataset import *
from models.bert_cnn_model import *


def train():
    # 解决内存溢出：1、bs 小一点 2、固定 bert 参数
    train_dataset = Dataset('train')
    train_loader = data.DataLoader(train_dataset, batch_size=32, collate_fn=collate_fn, shuffle=True)

    dev_dataset = Dataset('dev')
    dev_loader = data.DataLoader(dev_dataset, batch_size=32, collate_fn=collate_fn, shuffle=True)

    model = TextCNN().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = torch.nn.CrossEntropyLoss()

    model.train()
    for e in range(EPOCH):
        for i, (input_ids, attention_mask, labels) in enumerate(train_loader):
            out = model(input_ids, attention_mask)
            loss = loss_fn(out, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if i % 5 == 0:
                out = out.argmax(dim=1)
                acc = (out == labels).sum().item() / len(labels)

                with torch.no_grad():
                    input_ids_, attention_mask_, labels_ = iter(dev_loader).__next__()
                    out_ = model(input_ids_, attention_mask_)
                    out_ = out_.argmax(dim=1)
                    dev_acc = (out_ == labels_).sum().item() / len(labels_)

                print(
                    ' epoch: ', e,
                    ' batch: ', i,
                    ' loss: ', round(loss.item(), 2),
                    ' train_acc: ', acc,
                    ' dev_acc: ', dev_acc,
                )

        torch.save(model, MODEL_DIR + f'{e}.pth')

        if e == 0:
            exit()


if __name__ == '__main__':
    train()

