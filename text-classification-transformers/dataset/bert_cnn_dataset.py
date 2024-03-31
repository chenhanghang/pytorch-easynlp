from configs.bert_cnn_config import *
import torch
from torch.utils import data
from transformers import logging
import warnings

logging.set_verbosity_error()
warnings.filterwarnings("ignore")


class Dataset(data.Dataset):
    def __init__(self, split='train'):
        super(Dataset, self).__init__()
        path = "./data/input/" + str(split) + ".txt"
        self.lines = open(path).readlines()

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, i):
        text, label = self.lines[i].strip().split('\t')
        return text, int(label)

def collate_fn(data):
    sents = [i[0] for i in data]
    labels = [i[1] for i in data]

    data = tokenizer.batch_encode_plus(
        sents,
        add_special_tokens=True,
        truncation=True,
        padding="max_length",
        max_length=TEXT_LEN,
        return_tensors="pt",
        return_token_type_ids=True,
        return_attention_mask=True
    )

    input_ids = data["input_ids"]
    attention_mask = data["attention_mask"]
    # token_type_ids = data["token_type_ids"]  # 只有一个句子，不需要传这个参数
    labels = torch.LongTensor(labels)

    return input_ids.to(DEVICE), attention_mask.to(DEVICE), labels.to(DEVICE)

def get_label():
    text = open(LABEL_PATH).read()
    # print(text)
    id2label = text.split()
    return id2label, {v: k for k, v in enumerate(id2label)}


if __name__ == '__main__':
    # dataset = Dataset()
    # loader = data.DataLoader(
    #     dataset,
    #     batch_size=16,
    #     collate_fn=collate_fn,
    #     shuffle=True,
    #     drop_last=True
    # )
    #
    # for i, data in enumerate(loader):
    #     break

    print(get_label())
