from utils import *
from model import *
from torch.utils import data

if __name__ == '__main__':
    model = CasRel().to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    dataset = Dataset()
    for e in range(EPOCH):
        loader = data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=dataset.collate_fn)
        for b, (batch_mask, batch_x, batch_y) in enumerate(loader):
            batch_text, batch_sub_rnd = batch_x
            batch_sub, batch_obj_rel = batch_y

            # 整理input数据并预测
            input_mask = torch.tensor(batch_mask).to(DEVICE)
            input = (
                torch.tensor(batch_text['input_ids']).to(DEVICE),
                torch.tensor(batch_sub_rnd['head_seq']).to(DEVICE),
                torch.tensor(batch_sub_rnd['tail_seq']).to(DEVICE),
            )
            
            encoded_text, pred_y = model(input, input_mask)

            # 整理target数据并计算损失
            true_y = (
                torch.tensor(batch_sub['heads_seq']).to(DEVICE),
                torch.tensor(batch_sub['tails_seq']).to(DEVICE),
                torch.tensor(batch_obj_rel['heads_mx']).to(DEVICE),
                torch.tensor(batch_obj_rel['tails_mx']).to(DEVICE),
            )
            loss = model.loss_fn(true_y, pred_y, input_mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if b % 50 == 0:
                print('>> epoch:', e, 'batch:', b, 'loss:', loss.item())

            if b % 500 == 0:
                report(model, encoded_text, pred_y, batch_text, batch_mask)

        if e % 3 == 0:
            torch.save(model, MODEL_DIR + f'model_{e}.pth') 