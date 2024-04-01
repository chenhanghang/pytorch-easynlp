from configs.bert_cnn_config import *
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel



class TextCNN(nn.Module):
    def __init__(self):
        super(TextCNN, self).__init__()
        #self.bert = AutoModel.from_pretrained("bert-base-chinese")
        self.bert = AutoModel.from_pretrained("./model_hub/bert-base-chinese")
        for param in self.bert.parameters():
            param.requires_grad_(False)

        # (1, NUM_FILTERS, (2, EMBEDDING_DIM))
        # 1：输入通道数，文本看作 1 通道
        # NUM_FILTERS：输出通道数，也就是卷积核数量 256
        # (2, EMBEDDING_DIM)：卷积核大小，2 是宽度，一次覆盖几个词
        # EMBEDDING_DIM 是长度，正好覆盖 bert 输出词向量长度 768
        self.conv1 = nn.Conv2d(1, NUM_FILTERS, (2, EMBEDDING_DIM))
        self.conv2 = nn.Conv2d(1, NUM_FILTERS, (3, EMBEDDING_DIM))
        self.conv3 = nn.Conv2d(1, NUM_FILTERS, (4, EMBEDDING_DIM))
        self.linear = nn.Linear(NUM_FILTERS * 3, NUM_CLASSES)

    def conv_and_pool(self, conv, input):
        out = conv(input)
        out = F.relu(out)

        # 最大池化是在 shape 的最后两维，例如卷积后的 shape：(bz, channel, 29, 1)
        # 那么池化核尺寸为 (29, 1)，也就是 (out.shape[2], out.shape[3])
        # 池化后维度 (bz, channel, 1, 1)，squeeze() 去掉大小为 1 的维度，结果为 (bz, channel)
        # out1\2\3 均为 shape：(bz, channel)
        return F.max_pool2d(out, (out.shape[2], out.shape[3])).squeeze()

    def forward(self, input_ids, attention_mask):
        # self.bert(input, mask)[0] 等价于 self.bert(input, mask).last_hidden_state
        # self.bert(input, mask)[0] 的 shape 是(bz, 30, 768)
        # 但是传入卷积网络要求维数是 4，手动扩维 unsqueeze(1)，给句子特征添加通道数
        # shape 变成 (bz, channel, seq_length, embedding_dim)
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)[0].unsqueeze(1)
        out1 = self.conv_and_pool(self.conv1, out)
        out2 = self.conv_and_pool(self.conv2, out)
        out3 = self.conv_and_pool(self.conv3, out)
        out = torch.cat([out1, out2, out3], dim=1)
        return self.linear(out).softmax(dim=1)


if __name__ == '__main__':
    model = TextCNN()
    input_ids = torch.randint(0, 3000, (2, TEXT_LEN))
    attention_mask = torch.ones_like(input_ids)
    print(model(input_ids, attention_mask).shape)
