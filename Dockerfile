FROM nvcr.io/nvidia/pytorch:23.05-py3

# 安装 ONNX / TensorRT / Triton Client
RUN pip install --upgrade pip \ && pip install onnx onnxruntime-gpu tritonclient[all] tensorrt

WORKDIR /workspace
COPY . /workspace

CMD ["/bin/bash"]

