import numpy as np

import joblib
import torch

import dataset
import engine
import transformers
from tqdm import tqdm
from model_def import EntityModel
from train import process_data
from common_utils.address_cleaner import AddressCleaner
from utils_bert import createOutputContainer
from typing import List, Dict
from collections import defaultdict


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


class BertTagger:
    def __init__(self, base_model='bert-base-uncased'):
        self.model = None
        self.enc_tag = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.tokenizer = transformers.BertTokenizer.from_pretrained(base_model, do_lower_case=True)

    def loadModel(self, model_location, meta_location):
        meta_data = joblib.load(meta_location)
        self.enc_tag = meta_data["enc_tag"]

        self.model = EntityModel(num_tag=len(self.enc_tag.classes_))
        if self.device == 'cpu':
            self.model.load_state_dict(torch.load(model_location, map_location=torch.device('cpu')))
        else:
            self.model.load_state_dict(torch.load(model_location))
        self.model.to(self.device)

    def getPred(self, input: str) -> Dict[str, List]:
        cleanedAddressTokens = self.addressCleaner([input])[0].split()

        test_dataset = dataset.EntityDataset(
            texts=[cleanedAddressTokens],
            tags=[self.enc_tag.transform(['O'] * len(cleanedAddressTokens))]
        )

        data = test_dataset[0]
        for k, v in data.items():
            data[k] = v.to(self.device).unsqueeze(0)
        tag, _ = self.model(**data)

        tokenized_address = self.tokenizer.tokenize(' '.join(cleanedAddressTokens))

        decodedTags = self.enc_tag.inverse_transform(
            tag.argmax(2).cpu().numpy().reshape(-1)
        )[1:len(tokenized_address) + 1]

        decodedTags_compressed = convert_to_original_length(tokenized_address, decodedTags)

        return createOutputContainer(cleanedAddressTokens, decodedTags_compressed)

    @staticmethod
    def addressCleaner(samples: List[str]) -> List[str]:
        ac = AddressCleaner(samples)
        return ac.clean(tqdm_disable=True)


if __name__ == "__main__":
    bertTagger = BertTagger()
    bertTagger.loadModel(model_location='../model/model.bin', meta_location='../model/meta.bin')
    print(bertTagger.getPred(
        "Wilson Manor Apartments, Wilson Garden 13th cross ,Bengaluru ,(Safa Medicure Hospital) Bangalore - ,Karnataka"))