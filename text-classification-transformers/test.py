from configs.bert_cnn_config import *
from dataset.bert_cnn_dataset import *
from models.bert_cnn_model import *


def test():
    test_dataset = Dataset('test')
    test_loader = data.DataLoader(test_dataset, collate_fn=collate_fn, batch_size=64, shuffle=False)

    model = torch.load(MODEL_DIR + '0.pth').to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()

    total = 0
    correct = 0

    model.eval()
    with torch.no_grad():
        for i, (input_ids, attention_mask, labels) in enumerate(test_loader):
            out = model(input_ids, attention_mask)
            loss = loss_fn(out, labels)

            out = out.argmax(dim=1)
            correct += (out == labels).sum().item()
            total += len(labels)

            print(' batch: ', i, ' loss ', loss.item())

    acc = correct / total
    print(' acc: ', round(acc, 2))


if __name__ == '__main__':
    test()
