#!/bin/bash
set -e

# =============== 路径配置 ===============
ROOT_DIR=$(dirname $(readlink -f "$0"))/..
ONNX_DIR=$ROOT_DIR/onnx
TRT_DIR=$ROOT_DIR/tensorrt
TRITON_DIR=$ROOT_DIR/triton

mkdir -p $ONNX_DIR $TRT_DIR $TRITON_DIR

echo "📂 项目根目录: $ROOT_DIR"
echo "📂 ONNX 目录: $ONNX_DIR"
echo "📂 TensorRT Engine 目录: $TRT_DIR"
echo "📂 Triton 模型仓库: $TRITON_DIR"

# =============== 1. 训练 MNIST ===============
echo "📝 训练 MNIST 模型..."
python $ROOT_DIR/scripts/train_mnist.py

# =============== 2. 导出 MNIST ONNX ===============
echo "📦 导出 MNIST ONNX..."
python $ROOT_DIR/scripts/export_mnist.py

# =============== 3. 导出 ResNet50 ONNX ===============
echo "📦 导出 ResNet50 ONNX..."
python $ROOT_DIR/scripts/export_resnet50.py

# =============== 4. 转换 TensorRT Engine（支持动态 batch） ===============
echo "⚡ 转换 TensorRT Engine (FP16 + 动态 Batch)..."

# MNIST Engine: batch 1~8
trtexec \
  --onnx=$ONNX_DIR/mnist.onnx \
  --saveEngine=$TRT_DIR/mnist_fp16.engine \
  --fp16 \
  --minShapes=input:1x1x28x28 \
  --optShapes=input:4x1x28x28 \
  --maxShapes=input:8x1x28x28 \
  --explicitBatch \
  > $TRT_DIR/mnist_build.log 2>&1

# ResNet50 Engine: batch 1~16
trtexec \
  --onnx=$ONNX_DIR/resnet50.onnx \
  --saveEngine=$TRT_DIR/resnet50_fp16.engine \
  --fp16 \
  --minShapes=input:1x3x224x224 \
  --optShapes=input:8x3x224x224 \
  --maxShapes=input:16x3x224x224 \
  --explicitBatch \
  > $TRT_DIR/resnet50_build.log 2>&1

echo "✅ TensorRT Engine 已生成: $TRT_DIR/*.engine"

# =============== 5. 构建 Triton 模型仓库 ===============
echo "🚀 构建 Triton 模型仓库..."

# MNIST (ONNX)
mkdir -p $TRITON_DIR/mnist_mlp/1
cp -f $ONNX_DIR/mnist.onnx $TRITON_DIR/mnist_mlp/1/model.onnx
cat > $TRITON_DIR/mnist_mlp/config.pbtxt <<EOF
name: "mnist_mlp"
platform: "onnxruntime_onnx"
max_batch_size: 8
input [
  { name: "input", data_type: TYPE_FP32, dims: [1,28,28] }
]
output [
  { name: "output", data_type: TYPE_FP32, dims: [10] }
]
EOF

# ResNet50 (TensorRT)
mkdir -p $TRITON_DIR/resnet50/1
cp -f $TRT_DIR/resnet50_fp16.engine $TRITON_DIR/resnet50/1/model.plan
cat > $TRITON_DIR/resnet50/config.pbtxt <<EOF
name: "resnet50"
platform: "tensorrt_plan"
max_batch_size: 16
input [
  { name: "input", data_type: TYPE_FP32, dims: [3,224,224] }
]
output [
  { name: "output", data_type: TYPE_FP32, dims: [1000] }
]
EOF

echo "✅ Triton 模型仓库已生成: $TRITON_DIR"
