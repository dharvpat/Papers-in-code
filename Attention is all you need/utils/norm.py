import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.layer_norm = nn.LayerNorm(d_model, eps=eps)

    def forward(self, x):
        return self.layer_norm(x)

def test_layer_norm():
    d_model = 512
    layer_norm = LayerNorm(d_model)
    input_tensor = torch.randn(10, 20, d_model)  # (batch_size, seq_length, d_model)
    output_tensor = layer_norm(input_tensor)

    assert output_tensor.size() == input_tensor.size(), "LayerNorm output size mismatch"
    assert torch.allclose(output_tensor.mean(dim=-1), torch.zeros(10, 20)), "LayerNorm mean not close to zero"
    assert torch.allclose(output_tensor.std(dim=-1), torch.ones(10, 20)), "LayerNorm std not close to one"