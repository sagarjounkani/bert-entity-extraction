import transformers
import torch

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MAX_LEN = 128
TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 8
EPOCHS = 10
BASE_MODEL_PATH = "bert-base-uncased"
MODEL_PATH = "model.bin"
TRAINING_FILE = "../data/sampleAddressTaggedBIO.train"
DEV_FILE = "../data/sampleAddressTaggedBIO.dev"
TEST_FILE = "../data/sampleAddressTaggedBIO.test"
DECODED_FILE = "../data/decode.out"
TOKENIZER = transformers.BertTokenizer.from_pretrained(
    BASE_MODEL_PATH,
    do_lower_case=True
)
