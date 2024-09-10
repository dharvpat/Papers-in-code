import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer

class CustomDataset(Dataset):
    def __init__(self, input_data, target_data, tokenizer):
        self.input_data = input_data
        self.target_data = target_data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, idx):
        input_seq = self.tokenizer(self.input_data[idx], return_tensors='pt', padding='max_length', truncation=True)
        target_seq = self.tokenizer(self.target_data[idx], return_tensors='pt', padding='max_length', truncation=True)
        return input_seq['input_ids'].squeeze(0), target_seq['input_ids'].squeeze(0)

def load_data(input_file_path, target_file_path, tokenizer):
    with open(input_file_path, 'r') as f:
        input_data = f.readlines()
    with open(target_file_path, 'r') as f:
        target_data = f.readlines()

    return input_data, target_data

def get_data_loaders(input_data, target_data, tokenizer, batch_size=32, shuffle=True):
    dataset = CustomDataset(input_data, target_data, tokenizer)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return data_loader

def collate_fn(batch):
    input_seqs, target_seqs = zip(*batch)
    input_seqs = torch.nn.utils.rnn.pad_sequence(input_seqs, padding_value=0, batch_first=True)
    target_seqs = torch.nn.utils.rnn.pad_sequence(target_seqs, padding_value=0, batch_first=True)
    return input_seqs, target_seqs