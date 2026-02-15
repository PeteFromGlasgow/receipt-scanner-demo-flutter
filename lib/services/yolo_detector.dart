import 'dart:math';
import 'dart:typed_data';
import 'dart:ui';
import 'package:image/image.dart' as img;
import 'package:tflite_flutter/tflite_flutter.dart';
import '../models/detection_result.dart';

/// Runs YOLOv8-OBB document detection on a camera frame.
///
/// Uses a YOLOv8n-OBB (Oriented Bounding Box) model exported to TFLite.
///
/// Model I/O:
///   Input:  [1, 640, 640, 3] - NHWC float32, RGB normalized to 0-1
///   Output: [1, 20, 8400]    - raw OBB detections
///
/// Each of the 8400 anchors has 20 values:
///   [0:4]   x_center, y_center, width, height (in pixel space, 0-640)
///   [4:19]  15 class scores (raw logits — apply sigmoid)
///   [19]    rotation angle (radians)
///
/// Post-processing converts (x, y, w, h, angle) to 4 corner points
/// via rotation, then normalizes to 0-1 for resolution independence.
class YoloDetector {
  static const String _modelPath = 'assets/models/yolov8_doc_detector.tflite';
  static const int inputSize = 640;

  /// Minimum sigmoid(class_score) to consider a detection.
  static const double _confThreshold = 0.25;

  Interpreter? _interpreter;
  bool _isReady = false;

  bool get isReady => _isReady;

  Future<void> loadModel() async {
    try {
      _interpreter = await Interpreter.fromAsset(_modelPath);
      _isReady = true;
      print('YOLOv8-OBB document detector loaded successfully');
      final inputShape = _interpreter!.getInputTensor(0).shape;
      final outputShape = _interpreter!.getOutputTensor(0).shape;
      print('  Input shape:  $inputShape');
      print('  Output shape: $outputShape');
    } catch (e) {
      print('Failed to load YOLOv8 model: $e');
      print('Make sure yolov8_doc_detector.tflite is in assets/models/');
      _isReady = false;
    }
  }

  /// Runs detection on an [img.Image] and returns the best document detection.
  DetectionResult? detect(img.Image image) {
    if (!_isReady || _interpreter == null) return null;

    final inputImage =
        img.copyResize(image, width: inputSize, height: inputSize);
    final input = _imageToFloat32NHWC(inputImage);

    // Output: [1, 20, 8400]
    final outputShape = _interpreter!.getOutputTensor(0).shape;
    final outputSize = outputShape.reduce((a, b) => a * b);
    final output = List.filled(outputSize, 0.0).reshape(outputShape);

    _interpreter!.run(input, output);

    return _parseOBBOutput(output);
  }

  /// Converts image to Float32 in NHWC format [1, H, W, 3] normalized to 0-1.
  Float32List _imageToFloat32NHWC(img.Image image) {
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

  /// Parses [1, 20, 8400] OBB output into corner-based DetectionResult.
  ///
  /// For each anchor, extracts (x, y, w, h, best_class_score, angle),
  /// converts to 4 rotated corner points, and returns the highest-confidence
  /// detection with normalized (0-1) coordinates.
  DetectionResult? _parseOBBOutput(dynamic output) {
    try {
      // output[0] is [20, 8400]: 20 channels, 8400 anchors
      final channels = output[0] as List;
      final numAnchors = (channels[0] as List).length;
      final numChannels = channels.length;

      double bestConf = 0;
      DetectionResult? bestResult;

      for (int a = 0; a < numAnchors; a++) {
        // Extract bbox params (in pixel space, 0-640)
        final cx = (channels[0] as List)[a] as double;
        final cy = (channels[1] as List)[a] as double;
        final w = (channels[2] as List)[a] as double;
        final h = (channels[3] as List)[a] as double;

        // Find best class score among channels 4..18 (15 DOTA classes)
        double bestClassScore = -1e9;
        for (int c = 4; c < numChannels - 1; c++) {
          final score = (channels[c] as List)[a] as double;
          if (score > bestClassScore) bestClassScore = score;
        }

        // Apply sigmoid to get probability
        final confidence = _sigmoid(bestClassScore);
        if (confidence < _confThreshold) continue;

        // Rotation angle (last channel)
        final angle = (channels[numChannels - 1] as List)[a] as double;

        // Convert (cx, cy, w, h, angle) to 4 corner points
        final corners = _xywhrToCorners(cx, cy, w, h, angle);

        // Normalize corners to 0-1
        final norm = corners.map((p) =>
            Offset(p.dx / inputSize, p.dy / inputSize)).toList();

        if (confidence > bestConf) {
          bestConf = confidence;
          bestResult = DetectionResult(
            topLeft: norm[0],
            topRight: norm[1],
            bottomRight: norm[2],
            bottomLeft: norm[3],
            confidence: confidence,
          );
        }
      }

      return bestResult;
    } catch (e) {
      print('Error parsing YOLOv8-OBB output: $e');
      return null;
    }
  }

  /// Converts an oriented bounding box (cx, cy, w, h, angle) to 4 corner points.
  ///
  /// Returns corners in order: top-left, top-right, bottom-right, bottom-left
  /// (relative to the rotated rectangle).
  static List<Offset> _xywhrToCorners(
      double cx, double cy, double w, double h, double angle) {
    final cosA = cos(angle);
    final sinA = sin(angle);

    // Half-dimensions
    final hw = w / 2;
    final hh = h / 2;

    // Corner offsets before rotation (relative to center)
    final offsets = [
      [-hw, -hh], // top-left
      [hw, -hh],  // top-right
      [hw, hh],   // bottom-right
      [-hw, hh],  // bottom-left
    ];

    return offsets.map((o) {
      final rx = o[0] * cosA - o[1] * sinA;
      final ry = o[0] * sinA + o[1] * cosA;
      return Offset(cx + rx, cy + ry);
    }).toList();
  }

  static double _sigmoid(double x) => 1.0 / (1.0 + exp(-x));

  void dispose() {
    _interpreter?.close();
    _interpreter = null;
    _isReady = false;
  }
}
