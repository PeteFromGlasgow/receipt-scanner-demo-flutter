#!/usr/bin/env python3
"""
Exports a YOLOv8n-OBB model as a single-class document detector to TFLite.

The exported model has:
  Input:  [1, 3, 640, 640] - RGB image normalized to 0-1 (NCHW)
  Output: [1, 6, 8400]     - raw OBB detections

Each of the 8400 anchor predictions contains 6 values:
  [x_center, y_center, width, height, class_score, angle]
  - x,y,w,h are in pixel space (0-640)
  - class_score is a raw logit (apply sigmoid for probability)
  - angle is rotation in radians

Post-processing (NMS, xywhr -> corners) is done in the Dart app.

Usage:
    python3 tools/export_doc_detector.py

    To fine-tune on document data first:
    python3 tools/export_doc_detector.py --train --data path/to/doc_dataset.yaml

Requirements:
    pip install ultralytics torch onnx onnxscript onnxruntime
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from ultralytics import YOLO

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
ASSETS_DIR = os.path.join(PROJECT_DIR, "assets", "models")
ONNX_PATH = os.path.join(SCRIPT_DIR, "yolov8_doc_detector.onnx")
TFLITE_PATH = os.path.join(ASSETS_DIR, "yolov8_doc_detector.tflite")

INPUT_SIZE = 640


class SingleClassOBBDetector(nn.Module):
    """Wraps YOLOv8-OBB with a 1-class head for document detection.

    Input:  [1, 3, 640, 640] - RGB image, 0-1 normalized
    Output: [1, 6, 8400]     - raw detections (x,y,w,h,cls,angle)
    """

    def __init__(self, base_model):
        super().__init__()
        self.model = base_model

        # Modify detection head from 15 classes (DOTA) to 1 class (document)
        detect = self.model.model[-1]
        nc_new = 1

        for i in range(len(detect.cv3)):
            seq = detect.cv3[i]
            old_conv = seq[-1]
            new_conv = nn.Conv2d(
                old_conv.in_channels,
                nc_new,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=old_conv.bias is not None,
            )
            # Initialize with pretrained weights projected to single class
            # Average the 15-class weights to create a general "object" detector
            with torch.no_grad():
                new_conv.weight.copy_(old_conv.weight[:nc_new].mean(dim=0, keepdim=True))
                if new_conv.bias is not None and old_conv.bias is not None:
                    new_conv.bias.copy_(old_conv.bias[:nc_new].mean(dim=0, keepdim=True))
            seq[-1] = new_conv

        detect.nc = nc_new
        detect.no = nc_new + 5  # 4 bbox + 1 angle + 1 class

    def forward(self, x):
        raw = self.model(x)
        if isinstance(raw, (list, tuple)):
            raw = raw[0]
        return raw  # [1, 6, 8400]


def export_onnx(wrapper):
    """Export the model to ONNX."""
    wrapper.eval()
    dummy = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE)

    with torch.no_grad():
        output = wrapper(dummy)
    print(f"  Output shape: {output.shape}")

    torch.onnx.export(
        wrapper,
        dummy,
        ONNX_PATH,
        input_names=["images"],
        output_names=["detections"],
        opset_version=17,
    )
    print(f"  ONNX saved to: {ONNX_PATH}")


def convert_onnx_to_tflite():
    """Convert ONNX model to TFLite."""
    import onnxruntime as ort

    # Validate ONNX
    sess = ort.InferenceSession(ONNX_PATH)
    input_info = sess.get_inputs()[0]
    output_info = sess.get_outputs()[0]
    print(f"  ONNX input:  {input_info.name} {input_info.shape}")
    print(f"  ONNX output: {output_info.name} {output_info.shape}")

    # Test inference
    dummy = np.random.randn(1, 3, INPUT_SIZE, INPUT_SIZE).astype(np.float32)
    result = sess.run(None, {input_info.name: dummy})
    print(f"  Test output shape: {result[0].shape}")

    try:
        import onnx2tf
        saved_model_dir = os.path.join(SCRIPT_DIR, "saved_model")
        onnx2tf.convert(
            input_onnx_file_path=ONNX_PATH,
            output_folder_path=saved_model_dir,
            not_use_onnxsim=True,
        )
        # Find generated tflite
        import shutil
        for f in os.listdir(saved_model_dir):
            if f.endswith(".tflite"):
                shutil.copy(os.path.join(saved_model_dir, f), TFLITE_PATH)
                print(f"  TFLite saved to: {TFLITE_PATH}")
                return True
    except Exception as e:
        print(f"  onnx2tf failed: {e}")

    try:
        import tensorflow as tf
        converter = tf.lite.TFLiteConverter.from_saved_model(
            os.path.join(SCRIPT_DIR, "saved_model")
        )
        tflite_model = converter.convert()
        with open(TFLITE_PATH, "wb") as f:
            f.write(tflite_model)
        print(f"  TFLite saved to: {TFLITE_PATH}")
        return True
    except Exception as e:
        print(f"  TF conversion failed: {e}")

    return False


def main():
    parser = argparse.ArgumentParser(
        description="Export YOLOv8 document corner detector"
    )
    parser.add_argument(
        "--train", action="store_true",
        help="Fine-tune on document dataset before export",
    )
    parser.add_argument("--data", type=str, help="Path to training data YAML")
    parser.add_argument(
        "--epochs", type=int, default=50, help="Training epochs"
    )
    parser.add_argument(
        "--convert-only", action="store_true",
        help="Only convert existing ONNX to TFLite",
    )
    args = parser.parse_args()

    os.makedirs(ASSETS_DIR, exist_ok=True)

    if args.convert_only:
        if not os.path.exists(ONNX_PATH):
            print("Error: ONNX not found. Run without --convert-only first.")
            sys.exit(1)
        print("Converting ONNX to TFLite...")
        convert_onnx_to_tflite()
        return

    print("Loading YOLOv8n-OBB base model...")
    base = YOLO("yolov8n-obb.pt")

    if args.train:
        if not args.data:
            print("Error: --data required with --train")
            sys.exit(1)
        print(f"Fine-tuning on {args.data} for {args.epochs} epochs...")
        base.train(data=args.data, epochs=args.epochs, imgsz=INPUT_SIZE)

    print("Creating single-class document detector...")
    wrapper = SingleClassOBBDetector(base.model)

    print("Exporting to ONNX...")
    export_onnx(wrapper)

    print("Converting to TFLite...")
    success = convert_onnx_to_tflite()

    if success:
        print(f"\nDone! Model at: {TFLITE_PATH}")
    else:
        print(f"\nONNX model at: {ONNX_PATH}")
        print("TFLite conversion requires tensorflow:")
        print("  pip install tensorflow onnx2tf")
        print("  python3 tools/export_doc_detector.py --convert-only")


if __name__ == "__main__":
    main()
