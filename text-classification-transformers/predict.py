from configs.bert_cnn_config import *
from dataset.bert_cnn_dataset import *
from models.bert_cnn_model import *


def predict(sents):
    id2label, _ = get_label()
    model = torch.load(MODEL_DIR + '0.pth', map_location=DEVICE)
    tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')

    data = tokenizer.batch_encode_plus(
        sents,
        add_special_tokens=True,
        truncation=True,
        padding='max_length',
        max_length=30,
        return_tensors='pt',
        return_attention_mask=True,
    )

    input_ids = data['input_ids']
    attention_mask = data['attention_mask']

    model.eval()
    pre = model(input_ids, attention_mask)
    pre = pre.argmax(dim=1)
    pre_list = [id2label[i] for i in pre]

    print(pre_list)


if __name__ == '__main__':
    texts = [
        '小城不大，风景如画：边境小镇室韦的蝶变之路',
        '天问一号发射两周年，传回火卫一高清影像',
        '林志颖驾驶特斯拉自撞路墩起火，车头烧成废铁',
    ]

    predict(texts)
