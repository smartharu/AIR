import torch
from torch import nn
import copy


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def get_model_name(self) -> str:
        raise NotImplementedError()

    def to_inference_model(self) -> nn.Module:
        net = copy.deepcopy(self)
        net.eval()
        return net

    def load_model(self, strict: bool = True):
        checkpoint = torch.load(self.get_model_name() + ".pth")
        self.load_state_dict(checkpoint['model_state_dict'], strict=strict)
        print(checkpoint['epoch'])
        return

    def convert_to_onnx(self) -> None:
        x = torch.randn((1, 3, 128, 128))

        model = self.to_inference_model()

        dynamic_axes = {'input': {0: 'batch_size', 2: 'height', 3: 'width'},
                        'output': {0: 'batch_size', 2: 'height', 3: 'width'}}

        torch.onnx.export(model, x, self.get_model_name() + ".onnx", export_params=True, opset_version=17,
                          input_names=['input'],
                          output_names=['output'], dynamic_axes=dynamic_axes)


