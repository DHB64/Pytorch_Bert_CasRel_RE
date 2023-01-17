from config import *
from utils import *
from transformers import BertTokenizerFast
from model import *

if __name__ == '__main__':
    # text = '俞敏洪，出生于1962年9月4日的江苏省江阴市，大学毕业于北京大学西语系。'
    text = '周杰伦将于7月15日发布2022新专辑《最伟大的作品》。'
    tokenizer = BertTokenizerFast.from_pretrained(BERT_MODEL_NAME)
    tokenized = tokenizer(text, return_offsets_mapping=True)
    info = {}
    info['input_ids'] = tokenized['input_ids']
    info['offset_mapping'] = tokenized['offset_mapping']
    info['mask'] = tokenized['attention_mask']

    input_ids = torch.tensor([info['input_ids']]).to(DEVICE)
    batch_mask = torch.tensor([info['mask']]).to(DEVICE)

    model = torch.load(MODEL_DIR + 'model_27.pth', map_location=DEVICE)

    encoded_text = model.get_encoded_text(input_ids, batch_mask)
    pred_sub_head, pred_sub_tail = model.get_subs(encoded_text)

    sub_head_ids = torch.where(pred_sub_head[0] > SUB_HEAD_BAR)[0]
    sub_tail_ids = torch.where(pred_sub_tail[0] > SUB_TAIL_BAR)[0]
    mask = batch_mask[0]
    encoded_text = encoded_text[0]

    offset_mapping = info['offset_mapping']

    pred_triple_item = get_triple_list(sub_head_ids, sub_tail_ids, model, \
            encoded_text, text, mask, offset_mapping)
    
    print(text)
    print(pred_triple_item)