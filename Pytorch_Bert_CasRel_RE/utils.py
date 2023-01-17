import torch.utils.data as data
import pandas as pd
import random
from config import *
import json
import numpy as np
from transformers import BertTokenizerFast

def get_rel():
    df = pd.read_csv(REL_PATH, names=['rel', 'id'])
    return df['rel'].tolist(), dict(df.values)


class Dataset(data.Dataset):
    def __init__(self, type='train'):
        super().__init__()
        _, self.rel2id = get_rel()
        # 加载文件
        if type == 'train':
            file_path = TRAIN_JSON_PATH
        elif type == 'test':
            file_path = TEST_JSON_PATH
        elif type == 'dev':
            file_path = DEV_JSON_PATH
        with open(file_path) as f:
            self.lines = f.readlines()
        # 加载bert
        self.tokenizer = BertTokenizerFast.from_pretrained(BERT_MODEL_NAME)

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, index):
        line = self.lines[index]
        info = json.loads(line)
        tokenized = self.tokenizer(info['text'], return_offsets_mapping=True)
        info['input_ids'] = tokenized['input_ids']
        info['offset_mapping'] = tokenized['offset_mapping']
        return self.parse_json(info)

    def parse_json(self, info):
        text = info['text']
        input_ids = info['input_ids']
        dct = {
            'text': text,
            'input_ids': input_ids,
            'offset_mapping': info['offset_mapping'],
            'sub_head_ids': [],
            'sub_tail_ids': [],
            'triple_list': [],
            'triple_id_list': []
        }
        for spo in info['spo_list']:
            subject = spo['subject']
            object = spo['object']['@value']
            predicate = spo['predicate']
            dct['triple_list'].append((subject, predicate, object))
            # 计算 subject 实体位置
            tokenized = self.tokenizer(subject, add_special_tokens=False)
            sub_token = tokenized['input_ids']
            sub_pos_id = self.get_pos_id(input_ids, sub_token)
            if not sub_pos_id:
                continue
            sub_head_id, sub_tail_id = sub_pos_id
            # 计算 object 实体位置
            tokenized = self.tokenizer(object, add_special_tokens=False)
            obj_token = tokenized['input_ids']
            obj_pos_id = self.get_pos_id(input_ids, obj_token)
            if not obj_pos_id:
                continue
            obj_head_id, obj_tail_id = obj_pos_id
            # 数据组装
            dct['sub_head_ids'].append(sub_head_id)
            dct['sub_tail_ids'].append(sub_tail_id)
            dct['triple_id_list'].append((
                [sub_head_id, sub_tail_id],
                self.rel2id[predicate],
                [obj_head_id, obj_tail_id],
            ))
        return dct

    def get_pos_id(self, source, elem):
        for head_id in range(len(source)):
            tail_id = head_id + len(elem)
            if source[head_id:tail_id] == elem:
                return head_id, tail_id - 1

    def collate_fn(self, batch):
        batch.sort(key=lambda x: len(x['input_ids']), reverse=True)
        max_len = len(batch[0]['input_ids'])
        batch_text = {
            'text': [],
            'input_ids': [],
            'offset_mapping': [],
            'triple_list': [],
        }
        batch_mask = []
        batch_sub = {
            'heads_seq': [],
            'tails_seq': [],
        }
        batch_sub_rnd = {
            'head_seq': [],
            'tail_seq': [],
        }
        batch_obj_rel = {
            'heads_mx': [],
            'tails_mx': [],
        }

        for item in batch:
            input_ids = item['input_ids']
            item_len = len(input_ids)
            pad_len = max_len - item_len
            input_ids = input_ids + [0] * pad_len
            mask = [1] * item_len + [0] * pad_len
            # 填充subject位置
            sub_heads_seq = multihot(max_len, item['sub_head_ids'])
            sub_tails_seq = multihot(max_len, item['sub_tail_ids'])
            # 随机选择一个subject
            if len(item['triple_id_list']) == 0:
                continue
            sub_rnd = random.choice(item['triple_id_list'])[0]
            sub_rnd_head_seq = multihot(max_len, [sub_rnd[0]])
            sub_rnd_tail_seq = multihot(max_len, [sub_rnd[1]])
            # 根据随机subject计算relations矩阵
            obj_head_mx = [[0] * REL_SIZE for _ in range(max_len)]
            obj_tail_mx = [[0] * REL_SIZE for _ in range(max_len)]
            for triple in item['triple_id_list']:
                rel_id = triple[1]
                head_id, tail_id = triple[2]
                if triple[0] == sub_rnd:
                    obj_head_mx[head_id][rel_id] = 1
                    obj_tail_mx[tail_id][rel_id] = 1
            # 重新组装
            batch_text['text'].append(item['text'])
            batch_text['input_ids'].append(input_ids)
            batch_text['offset_mapping'].append(item['offset_mapping'])
            batch_text['triple_list'].append(item['triple_list'])
            batch_mask.append(mask)
            batch_sub['heads_seq'].append(sub_heads_seq)
            batch_sub['tails_seq'].append(sub_tails_seq)
            batch_sub_rnd['head_seq'].append(sub_rnd_head_seq)
            batch_sub_rnd['tail_seq'].append(sub_rnd_tail_seq)
            batch_obj_rel['heads_mx'].append(obj_head_mx)
            batch_obj_rel['tails_mx'].append(obj_tail_mx)

        # 注意，结构太复杂，没有转tensor
        return batch_mask, (batch_text, batch_sub_rnd), (batch_sub, batch_obj_rel)


# 生成长度为length，hot_pos位置为1，其他位置为0的列表
def multihot(length, hot_pos):
    return [1 if i in hot_pos else 0 for i in range(length)]


def get_triple_list(sub_head_ids, sub_tail_ids, model, encoded_text, text, mask, offset_mapping):
    id2rel, _ = get_rel()
    triple_list = []
    for sub_head_id in sub_head_ids:
        sub_tail_ids = sub_tail_ids[sub_tail_ids >= sub_head_id]
        if len(sub_tail_ids) == 0:
            continue
        sub_tail_id = sub_tail_ids[0]
        if mask[sub_head_id] == 0 or mask[sub_tail_id] == 0:
            continue
        # 根据位置信息反推出 subject 文本内容
        sub_head_pos_id = offset_mapping[sub_head_id][0]
        sub_tail_pos_id = offset_mapping[sub_tail_id][1]
        subject_text = text[sub_head_pos_id:sub_tail_pos_id]

        # 根据 subject 计算出对应 object 和 relation
        sub_head_seq = torch.tensor(multihot(len(mask), sub_head_id)).to(DEVICE)
        sub_tail_seq = torch.tensor(multihot(len(mask), sub_tail_id)).to(DEVICE)

        pred_obj_head, pred_obj_tail = model.get_objs_for_specific_sub(\
            encoded_text.unsqueeze(0), sub_head_seq.unsqueeze(0), sub_tail_seq.unsqueeze(0))

        # 按分类找对应关系
        pred_obj_head = pred_obj_head[0].T
        pred_obj_tail = pred_obj_tail[0].T
        for j in range(len(pred_obj_head)):
            obj_head_ids = torch.where(pred_obj_head[j] > OBJ_HEAD_BAR)[0]
            obj_tail_ids = torch.where(pred_obj_tail[j] > OBJ_TAIL_BAR)[0]
            for obj_head_id in obj_head_ids:
                obj_tail_ids = obj_tail_ids[obj_tail_ids >= obj_head_id]
                if len(obj_tail_ids) == 0:
                    continue
                obj_tail_id = obj_tail_ids[0]
                if mask[obj_head_id] == 0 or mask[obj_tail_id] == 0:
                    continue
                # 根据位置信息反推出 object 文本内容，mapping中已经有移位，不需要再加1
                obj_head_pos_id = offset_mapping[obj_head_id][0]
                obj_tail_pos_id = offset_mapping[obj_tail_id][1]
                object_text = text[obj_head_pos_id:obj_tail_pos_id]
                triple_list.append((subject_text, id2rel[j], object_text))
    return list(set(triple_list))


def report(model, encoded_text, pred_y, batch_text, batch_mask):
    # 计算三元结构，和统计指标
    pred_sub_head, pred_sub_tail, _, _ = pred_y
    true_triple_list = batch_text['triple_list']
    pred_triple_list = []

    correct_num, predict_num, gold_num = 0, 0, 0

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


if __name__ == '__main__':
    dataset = Dataset()
    loader = data.DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=dataset.collate_fn)
    print(iter(loader).next())