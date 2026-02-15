import 'dart:typed_data';
import 'dart:ui';
import 'package:image/image.dart' as img;
import 'package:tflite_flutter/tflite_flutter.dart';
import '../models/detection_result.dart';

/// Runs YOLOv8 document detection on a camera frame.
///
/// Expects a YOLOv8 model trained for document/receipt detection, exported to
/// TFLite format. The model should output bounding-box corners (4 points) plus
/// a confidence score.
///
/// Place your model at: assets/models/yolov8_doc_detector.tflite
class YoloDetector {
  static const String _modelPath = 'assets/models/yolov8_doc_detector.tflite';
  static const int inputSize = 640;

  Interpreter? _interpreter;
  bool _isReady = false;

  bool get isReady => _isReady;

  Future<void> loadModel() async {
    try {
      _interpreter = await Interpreter.fromAsset(_modelPath);
      _isReady = true;
      print('YOLOv8 document detector loaded successfully');
    } catch (e) {
      print('Failed to load YOLOv8 model: $e');
      print('Make sure yolov8_doc_detector.tflite is in assets/models/');
      _isReady = false;
    }
  }

  /// Runs detection on an [img.Image] and returns the best document detection.
  DetectionResult? detect(img.Image image) {
    if (!_isReady || _interpreter == null) return null;

    final inputImage = img.copyResize(image, width: inputSize, height: inputSize);
    final input = _imageToFloat32(inputImage);

    // YOLOv8 output shape varies by model config. A typical document detection
    // model outputs [1, N, 9] where each detection has:
    //   [x1, y1, x2, y2, x3, y3, x4, y4, confidence]
    // Adjust the output shape to match your specific model.
    final outputShape = _interpreter!.getOutputTensor(0).shape;
    final outputSize = outputShape.reduce((a, b) => a * b);
    final output = List.filled(outputSize, 0.0).reshape(outputShape);

    _interpreter!.run(input, output);

    return _parseOutput(output);
  }

  Float32List _imageToFloat32(img.Image image) {
    final buffer = Float32List(1 * inputSize * inputSize * 3);
    int idx = 0;
    for (int y = 0; y < inputSize; y++) {
      for (int x = 0; x < inputSize; x++) {
        final pixel = image.getPixel(x, y);
        buffer[idx++] = pixel.r / 255.0;
        buffer[idx++] = pixel.g / 255.0;
        buffer[idx++] = pixel.b / 255.0;
      }
    }
    return buffer;
  }

  /// Parses model output. Coordinates are kept normalized (0-1) so they can
  /// be scaled to any image resolution (stream preview or captured photo).
  DetectionResult? _parseOutput(dynamic output) {
    // This parsing logic should be adapted to your specific YOLOv8 model output
    // format. Below is a common layout for a 4-corner document detector.
    //
    // The model outputs normalized coordinates (0-1).
    // We store them as normalized so they can be applied to any resolution.
    // We find the detection with the highest confidence.

    try {
      final detections = output[0] as List;
      double bestConf = 0;
      DetectionResult? bestResult;

      for (final det in detections) {
        final d = det as List<double>;
        if (d.length < 9) continue;

        final confidence = d[8];
        if (confidence <= bestConf) continue;

        bestConf = confidence;
        bestResult = DetectionResult(
          topLeft: Offset(d[0], d[1]),
          topRight: Offset(d[2], d[3]),
          bottomRight: Offset(d[4], d[5]),
          bottomLeft: Offset(d[6], d[7]),
          confidence: confidence,
        );
      }

      return bestResult;
    } catch (e) {
      print('Error parsing YOLOv8 output: $e');
      return null;
    }
  }

  void dispose() {
    _interpreter?.close();
    _interpreter = null;
    _isReady = false;
  }
}
