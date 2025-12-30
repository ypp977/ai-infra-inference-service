# scripts/export_resnet50.py
import torch, torchvision

model = torchvision.models.resnet50(pretrained=True)
model.eval()

dummy = torch.randn(1, 3, 224, 224)
torch.onnx.export(model, dummy, "onnx/resnet50.onnx",
                  input_names=["input"], output_names=["output"],
                  dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}})

print("ResNet50 导出完成: onnx/resnet50.onnx")
