


import torch
import torch.nn as nn
import openvino as ov
from pathlib import Path

class SoftmaxModel(nn.Module):
    def __init__(self):
        super(SoftmaxModel, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        return self.softmax(x)



model = SoftmaxModel()


datatype_map = {
    'bf16': torch.bfloat16,
    'fp32': torch.float32,
}
for label, datatype in datatype_map.items():
    example_input = torch.randn(1, 16, 1024, 1024, dtype=datatype)
    model.to(datatype)
    ov_model = ov.convert_model(model, example_input=example_input, input=example_input.shape)

    ovir_dir = Path(f"ovir_softmax_{label}")
    ov.save_model(ov_model, ovir_dir / "softmax.xml")

# NOTE: for bf16, we need to hack the xml, for some reason, it was output as f16 or FP16, do search and replace to bf16, BF16