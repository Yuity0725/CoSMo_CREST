# utf-8
"""
This code is to create the new caption dataset
without missing data
"""
import json
import os

_DEFAULT_FASHION_IQ_DATASET_ROOT = '../content/drive/MyDrive/CREST'
_DEFAULT_FASHION_IQ_VOCAB_PATH = '../content/drive/MyDrive/CREST/fashion_iq_vocab.pkl'


def get_img_caption_json(dataset_root, clothing_type, split):
    with open(os.path.join(dataset_root, 'captions/raw_cap/new_cap', 'cap.{}.{}.json'.format(clothing_type, split))) as json_file:
        img_caption_data = json.load(json_file)
    fname = "prefile.txt"
    with open(fname) as f:
        l = f.readlines()
        fname_list = [i.replace("\n", "") for i in l]

    new_data = []
    for i in range(len(img_caption_data)):
        # img_caption_data[i]['bert_feature'] = 'ln_bert_' + \
        #     str(i + 1).zfill(5) + '.npy'
        print(img_caption_data[i])
        target_path = os.path.join(
            dataset_root, "all_imgs/" + clothing_type + '/' + img_caption_data[i]["target"] + ".jpg")
        ref_path = os.path.join(
            dataset_root, "all_imgs/" + clothing_type + '/' + img_caption_data[i]["candidate"] + ".jpg")
        # if os.path.isfile(target_path) == True and os.path.isfile(ref_path) == True:
        img_caption_data[i]["dense_pose"] = str(int(fname_list.index(img_caption_data[i]["candidate"] + ".jpg")) + 1)
        new_data.append(img_caption_data[i])

    with open(os.path.join(dataset_root, 'captions/raw_cap/add_densepose_cap/', 'cap.{}.{}.json'.format(clothing_type, split)), 'a') as f:
        json.dump(new_data, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    dataset_root = _DEFAULT_FASHION_IQ_DATASET_ROOT
    clothing_types = ['toptee']
    split = 'train'
    for clothing_type in clothing_types:
        get_img_caption_json(dataset_root, clothing_type, split)
