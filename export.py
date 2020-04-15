from example import *
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch_model = Res_Deeplab()
x = torch.randn(1, 3, 1024, 2048, requires_grad=True)
torch_model.eval()
torch_out = torch_model(x)


torch.onnx.export(torch_model, x, 'output_deeplabv3.onnx', verbose=True, export_params=True, opset_version=11, do_constant_folding=True, input_names = ['input'], output_names = ['output'], dynamic_axes={'input' : {0 : 'batch_size'}, 'output' : {0 : 'batch_size'}})
'''
import onnx
onnx_model = onnx.load('./output_deeplabv3.onnx')
onnx.checker.check_model(onnx_model)
print('passed')
'''
