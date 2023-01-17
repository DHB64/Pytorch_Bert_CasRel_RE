from utils import *
from model import *
from torch.utils import data

if __name__ == '__main__':
    model = torch.load(MODEL_DIR + f'model_27.pth', map_location=DEVICE)

    dataset = Dataset('dev')

    with torch.no_grad():

        loader = data.DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=dataset.collate_fn)
        
        correct_num, predict_num, gold_num = 0, 0, 0
        pred_triple_list = []
        true_triple_list = []
        
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

            print('>> batch:', b, 'loss:', loss.item())

            # 计算关系三元组，和统计指标
            pred_sub_head, pred_sub_tail, _, _ = pred_y
            true_triple_list += batch_text['triple_list']
            
            # 遍历batch
            for i in range(len(pred_sub_head)):
                text = batch_text['text'][i]
                true_triple_item = true_triple_list[i]
                mask = batch_mask[i]
                offset_mapping = batch_text['offset_mapping'][i]

                sub_head_ids = torch.where(pred_sub_head[i] > SUB_HEAD_BAR)[0]
                sub_tail_ids = torch.where(pred_sub_tail[i] > SUB_TAIL_BAR)[0]

                pred_triple_item = get_triple_list(sub_head_ids, sub_tail_ids, model, \
                    encoded_text[i], text, mask, offset_mapping)

                # 统计个数
                correct_num += len(set(true_triple_item) & set(pred_triple_item))
                predict_num += len(set(pred_triple_item))
                gold_num += len(set(true_triple_item))

                pred_triple_list.append(pred_triple_item)

        precision = correct_num / (predict_num + EPS)
        recall = correct_num / (gold_num + EPS)
        f1_score = 2 * precision * recall / (precision + recall + EPS)
        print('\tcorrect_num:', correct_num, 'predict_num:', predict_num, 'gold_num:', gold_num)
        print('\tprecision:%.3f' % precision, 'recall:%.3f' % recall, 'f1_score:%.3f' % f1_score)