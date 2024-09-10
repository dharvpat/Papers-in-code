import torch
import pytest
from model.transformer import Transformer

@pytest.fixture
def model():
    return Transformer(d_model=512, nhead=8, num_layers=6, d_ff=2048, dropout=0.1)

def test_model_forward(model):
    input_seq = torch.randint(0, 100, (10, 20))  # (batch_size, seq_length)
    output = model(input_seq)
    
    assert output.size() == (10, 20, 512)  # (batch_size, seq_length, d_model)
    assert output is not None

def test_model_initialization(model):
    params = sum(p.numel() for p in model.parameters())
    expected_params = 0  # Set this to the expected number of parameters for your model
    assert params > expected_params, "Model has not been initialized correctly"