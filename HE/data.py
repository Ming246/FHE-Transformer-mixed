from transformers import AutoTokenizer, DataCollatorWithPadding
import datasets
import torch
from torch.utils.data import DataLoader
import numpy as np
from liberate.fhe.data_struct import DataStruct

from ckks import CkksEngine

class ThorDataEncryptor:
    def __init__(self, dataset_type: str,  dataset:str="", 
                 embedding_model:torch.nn.Module=None, ckks_engine: CkksEngine=None, test=False):
        self.checkpoint = "bert-base-uncased" 
        
        dataset_type = dataset_type.lower()
        if dataset_type not in {'mrpc', 'rte', 'sst2', 'stsb'}:
            raise ValueError("Invalid dataset type")
        self.dataset_type = dataset_type
        
        if dataset == "":
            print("Downloading dataset")
            dataset = datasets.load_dataset("glue", self.dataset_type)
        else:
            print("Loading dataset from disk")
            dataset = datasets.load_from_disk(dataset)
            
        self.dataset = dataset
        del self.dataset['train']
        del self.dataset['test']
        if test:
            self.dataset['validation'] = self.dataset['validation'].select(range(1))

        self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint, model_max_length = 128)
        if self.dataset_type == 'stsb':
            for key in self.dataset.keys():
                self.dataset[key] = self.dataset[key].map(lambda x: {**x, 'label': x['label'] / 5})
        self.embedding_model = embedding_model
        self.engine = ckks_engine 
        self.tokenized_dataset = self.tokenize()
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        self.eval_dataloader = DataLoader(
            self.tokenized_dataset["validation"], batch_size=1, collate_fn=data_collator
        )       
        
    def encode_attention_mask(self, attention_mask:np.ndarray, level:int=15) -> np.ndarray[DataStruct]:
        """
        Return an array of size (8,) which contains 8 plaintexts. 
        """
        if attention_mask.shape != (128,):
            raise ValueError("Shape of attention mask should be (128,)")
        n_tokens = np.count_nonzero(attention_mask)
        attention_mask = np.full((8,), None, dtype=object)
        for i in range(8):
            msg = np.zeros((2**15,), dtype=float)
            for j in range(16):
                temp = j *(2**11)
                diag_index = i * 16 + j
                for t in range(128):
                    col_index = (diag_index + t) % 128
                    is_token = 1 if col_index < n_tokens else 0
                    for head in range(12):
                        msg[temp + t*16 + head] = is_token
            attention_mask[i] = self.engine.encode(msg, level)
        return attention_mask

    def embed_data(self, input_data:dict) -> np.ndarray:
        with torch.no_grad():
            embedding = self.embedding_model(**input_data)
        return embedding.numpy().squeeze()

    def tokenize(self) -> datasets.DatasetDict:
        tokenized_dataset = self.dataset.map(self._tokenize, batched=True)
        if self.dataset_type in {'mrpc', 'rte', 'stsb'}:
            tokenized_dataset = tokenized_dataset.remove_columns(['sentence1', 'sentence2', 'idx']).rename_column('label', 'labels')
        elif self.dataset_type == 'sst2':
            tokenized_dataset = tokenized_dataset.remove_columns(['sentence', 'idx']).rename_column('label', 'labels')
        tokenized_dataset.set_format(type='torch')
        return tokenized_dataset
    
    def _tokenize(self, data):
        if self.dataset_type in {'mrpc', 'rte', 'stsb'}:
            return self.tokenizer(data['sentence1'], data['sentence2'],max_length=128,padding='max_length',truncation=True)
        elif self.dataset_type == 'sst2':
            return self.tokenizer(data['sentence'],padding='max_length',max_length=128,truncation=True)

def _encode_input_lower8(embedding: np.ndarray) -> list[np.ndarray]:
    """BERT-base embedding (128, 768) → 8 real slot packs.

    Same layout as the old lower-diagonal encoder: 16 diagonals per pack,
    head axis padded to 16, hidden split into 6 column blocks.
    """
    x = np.asarray(embedding, dtype=np.float64)
    if x.shape != (128, 768):
        raise ValueError(f"expected embedding (128, 768), got {x.shape}")
    x_blocks = np.vsplit(x.T, 6)
    packs: list[np.ndarray] = []
    for pack_i in range(8):
        vec = np.zeros(32768, dtype=np.float64)
        for j in range(16):
            diag = pack_i * 16 + j
            for t in range(128):
                row = (diag + t) % 128
                for h in range(12):
                    vec[j * 2048 + t * 16 + h] = x_blocks[h % 6][row, t]
        packs.append(vec)
    return packs


def encrypt_embedding_real8(engine, embedding: np.ndarray, pk, *, level: int):
    """Encrypt the layer-entry embedding as 8 real packs."""
    msgs = _encode_input_lower8(embedding)
    if len(msgs) != 8:
        raise ValueError(f"expected 8 lower packs, got {len(msgs)}")
    cts = np.full((8,), None, dtype=object)
    for i, msg in enumerate(msgs):
        cts[i] = engine.encode_and_encrypt(msg, pk, level=level)
    return cts.astype(object)
