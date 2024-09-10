import torch
import pytest
from utils.loss import compute_loss
from utils.optimizer import get_std_opt
from model.transformer import Transformer

@pytest.fixture
def model():
    return Transformer(d_model=512, nhead=8, num_layers=6, d_ff=2048, dropout=0.1)

@pytest.fixture
def dummy_data():
    input_seq = torch.randint(0, 100, (10, 20))
    target_seq = torch.randint(0, 100, (10, 20))
    return input_seq, target_seq

def test_compute_loss(dummy_data):
    input_seq, target_seq = dummy_data
    model = Transformer(d_model=512, nhead=8, num_layers=6, d_ff=2048, dropout=0.1)
    outputs = model(input_seq)
    loss = compute_loss(outputs, target_seq)
    
    assert loss.item() >= 0  # Loss should be non-negative

def test_get_std_opt(model):
    optimizer = get_std_opt(model)
    
    assert optimizer is not None
    assert len(optimizer.param_groups) > 0
    assert optimizer.defaults['lr'] == 0.001  # Check learning rate from defaults