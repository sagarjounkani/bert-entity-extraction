import numpy as np

import joblib
import torch

import config
import dataset
import engine
from tqdm import tqdm
from model import EntityModel
from train import load_data_bio


def convert_to_original_length(tokens, tags):
    r = []
    r_tags = []
    for index, token in enumerate(tokens):
        if token.startswith("##"):
            if r:
                r[-1] = f"{r[-1]}{token[2:]}"
        else:
            r.append(token)
            r_tags.append(tags[index])
    return r_tags


if __name__ == "__main__":

    meta_data = joblib.load("meta.bin")
    enc_tag = meta_data["enc_tag"]

    num_tag = len(list(enc_tag.classes_))

    sentences, tags = load_data_bio(config.TEST_FILE)
    tags = [enc_tag.transform(sublist) for sublist in tags]

    test_dataset = dataset.EntityDataset(
        texts=sentences,
        tags=tags
    )

    model = EntityModel(num_tag=num_tag)
    if config.DEVICE == 'cpu':
        model.load_state_dict(torch.load(config.MODEL_PATH, map_location=torch.device('cpu')))
    else:
        model.load_state_dict(torch.load(config.MODEL_PATH))
    model.to(config.DEVICE)

    with torch.no_grad():

        fout = open(config.DECODED_FILE, 'w')

        for idx, data in tqdm(enumerate(test_dataset), total=len(test_dataset), desc='Predicting tags for each address ...'):
            for k, v in data.items():
                data[k] = v.to(config.DEVICE).unsqueeze(0)
            tag, _ = model(**data)

            sentence = test_dataset.texts[idx]
            tokenized_sentence = config.TOKENIZER.tokenize(' '.join(sentence))

            decodedTags = enc_tag.inverse_transform(
                tag.argmax(2).cpu().numpy().reshape(-1)
            )[1:len(tokenized_sentence)+1]

            decodedTags_compressed = convert_to_original_length(tokenized_sentence, decodedTags)
            for token, _tag in zip(sentence, decodedTags_compressed):
                fout.write(token + " " + _tag + "\n")
            fout.write('\n')
        fout.close()
