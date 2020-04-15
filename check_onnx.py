import onnx
onnx_model = onnx.load('./output_deeplabv3.onnx')
onnx.checker.check_model(onnx_model)
print('passed')
