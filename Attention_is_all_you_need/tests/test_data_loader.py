import torch
import pytest
from utils.data_loader import get_data_loaders, collate_fn

@pytest.fixture
def data_loader():
    # Dummy data
    input_data = ["hello world", "transformer model"]
    target_data = ["hola mundo", "modelo transformer"]
    tokenizer = lambda x: torch.tensor([ord(c) for c in x])  # Simple tokenizer
    return get_data_loaders(input_data, target_data, tokenizer)

def test_data_loader_batch(data_loader):
    for input_seq, target_seq in data_loader:
        assert input_seq.size(0) == 64  # Batch size
        assert target_seq.size(0) == 64  # Batch size
        assert input_seq.size(1) > 0  # Sequence length
        assert target_seq.size(1) > 0  # Sequence length
        break  # Test only one batch

def test_collate_fn():
    batch = [
        (torch.tensor([1, 2, 3]), torch.tensor([4, 5])),
        (torch.tensor([6, 7]), torch.tensor([8, 9, 10])),
    ]
    input_seqs, target_seqs = collate_fn(batch)
    
    assert input_seqs.size(0) == 2  # Number of sequences in batch
    assert input_seqs.size(1) == 3  # Max sequence length (padded)
    assert input_seqs.size(2) == 0  # Padding value (if any)
    assert target_seqs.size(0) == 2  # Number of sequences in batch
    assert target_seqs.size(1) == 3  # Max sequence length (padded)
    assert target_seqs.size(2) == 0  # Padding value (if any)
