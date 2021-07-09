from typing import List, Dict

import joblib
import torch
import transformers
from tqdm import tqdm

from .dataset import EntityDataset
from common_utils.address_cleaner import AddressCleaner
from .model import EntityModel
from .utils import createOutputContainer


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
        self.model.eval()

    def getPred(self, input: str) -> Dict[str, List]:
        cleanedAddressTokens = self.addressCleaner([input])[0].split()

        test_dataset = EntityDataset(
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

    def getBatchPred(self, input:List[str], batchSize=2) -> List[Dict[str, List]]:
        out = []
        cleanedAddressTokens = [address.split() for address in self.addressCleaner(input)]

        test_dataset = EntityDataset(
            texts=cleanedAddressTokens,
            tags=[self.enc_tag.transform(['O'] * len(address)) for address in cleanedAddressTokens]
        )

        test_data_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=batchSize, num_workers=1
        )

        for idx, data in tqdm(enumerate(test_data_loader), total=len(test_data_loader),
                              desc='Predicting tags for each address ...'):
            for k, v in data.items():
                data[k] = v.to(self.device)
            tag, _ = self.model(**data)

            tokenizedAddressBatch = [self.tokenizer.convert_ids_to_tokens(ids, skip_special_tokens=True)
                                     for ids in data['ids']]

            decodedTagIds = tag.argmax(2).cpu().numpy()
            decodedTags = [self.enc_tag.inverse_transform(ids)[1:len(address)+1]
                           for address, ids in zip(tokenizedAddressBatch, decodedTagIds)]

            decodedTags_compressed = [convert_to_original_length(address, tags)
                                      for address, tags in zip(tokenizedAddressBatch, decodedTags)]

            addressBatch = cleanedAddressTokens[idx * batchSize: (idx + 1) * batchSize]

            out.extend([createOutputContainer(toks, tags)
                        for toks, tags in zip(addressBatch, decodedTags_compressed)])

        return out


if __name__ == "__main__":
    bertTagger = BertTagger()
    bertTagger.loadModel(model_location='../model/model.bin', meta_location='../model/meta.bin')

    print(bertTagger.getBatchPred([
        "Wilson Manor Apartments, Wilson Garden 13th cross ,Bengaluru ,(Safa Medicure Hospital) Bangalore - ,Karnataka",
        "Wilson Manor Apartments, Wilson Garden 13th cross ,Bengaluru ,(Safa Medicure Hospital) Bangalore - ,Karnataka"
    ]))

    print(bertTagger.getPred(
        "Wilson Manor Apartments, Wilson Garden 13th cross ,Bengaluru ,(Safa Medicure Hospital) Bangalore - ,Karnataka"
    ))