import 'dart:async';
import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:image/image.dart' as img;
import 'package:permission_handler/permission_handler.dart';
import '../models/detection_result.dart';
import '../services/yolo_detector.dart';
import '../utils/cubic_polynomial_cropper.dart';
import '../widgets/edge_overlay_painter.dart';
import 'result_screen.dart';

class CameraScreen extends StatefulWidget {
  const CameraScreen({super.key});

  @override
  State<CameraScreen> createState() => _CameraScreenState();
}

class _CameraScreenState extends State<CameraScreen> {
  CameraController? _cameraController;
  final YoloDetector _detector = YoloDetector();
  DetectionResult? _currentDetection;
  bool _isProcessing = false;
  bool _isCameraReady = false;
  String _statusMessage = 'Initializing...';

  @override
  void initState() {
    super.initState();
    _initialize();
  }

  Future<void> _initialize() async {
    await _requestPermissions();
    await _detector.loadModel();
    await _initCamera();
  }

  Future<void> _requestPermissions() async {
    final status = await Permission.camera.request();
    if (!status.isGranted) {
      setState(() => _statusMessage = 'Camera permission denied');
    }
  }

  Future<void> _initCamera() async {
    final cameras = await availableCameras();
    if (cameras.isEmpty) {
      setState(() => _statusMessage = 'No cameras available');
      return;
    }

    _cameraController = CameraController(
      cameras.first,
      ResolutionPreset.high,
      enableAudio: false,
      imageFormatGroup: ImageFormatGroup.yuv420,
    );

    try {
      await _cameraController!.initialize();
      _cameraController!.startImageStream(_processFrame);
      setState(() {
        _isCameraReady = true;
        _statusMessage = _detector.isReady
            ? 'Point at a receipt or document'
            : 'Model not loaded — using manual capture';
      });
    } catch (e) {
      setState(() => _statusMessage = 'Camera error: $e');
    }
  }

  void _processFrame(CameraImage cameraImage) {
    if (_isProcessing || !_detector.isReady) return;
    _isProcessing = true;

    // Convert camera image to img.Image for detection.
    final image = _convertCameraImage(cameraImage);
    if (image == null) {
      _isProcessing = false;
      return;
    }

    final result = _detector.detect(image);

    if (mounted) {
      setState(() {
        _currentDetection = result;
        if (result != null && result.isValid) {
          _statusMessage =
              'Document detected (${(result.confidence * 100).toStringAsFixed(0)}%)';
        } else {
          _statusMessage = 'Searching for document...';
        }
      });
    }

    _isProcessing = false;
  }

  img.Image? _convertCameraImage(CameraImage cameraImage) {
    try {
      final width = cameraImage.width;
      final height = cameraImage.height;
      final yPlane = cameraImage.planes[0];
      final uPlane = cameraImage.planes[1];
      final vPlane = cameraImage.planes[2];

      final image = img.Image(width: width, height: height);

      for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
          final yIndex = y * yPlane.bytesPerRow + x;
          final uvIndex =
              (y ~/ 2) * uPlane.bytesPerRow + (x ~/ 2);

          final yVal = yPlane.bytes[yIndex];
          final uVal = uPlane.bytes[uvIndex];
          final vVal = vPlane.bytes[uvIndex];

          final r = (yVal + 1.370705 * (vVal - 128)).round().clamp(0, 255);
          final g = (yVal - 0.337633 * (uVal - 128) - 0.698001 * (vVal - 128))
              .round()
              .clamp(0, 255);
          final b = (yVal + 1.732446 * (uVal - 128)).round().clamp(0, 255);

          image.setPixelRgb(x, y, r, g, b);
        }
      }
      return image;
    } catch (e) {
      print('Error converting camera image: $e');
      return null;
    }
  }

  Future<void> _captureAndCrop() async {
    if (_cameraController == null || !_cameraController!.value.isInitialized) {
      return;
    }

    setState(() => _statusMessage = 'Capturing...');
    final debugLog = StringBuffer();

    try {
      // Step 1: Stop the image stream before taking a picture.
      debugLog.writeln('[Step 1] Stopping image stream');
      await _cameraController!.stopImageStream();

      // Step 2: Capture photo.
      debugLog.writeln('[Step 2] Taking picture...');
      final xFile = await _cameraController!.takePicture();
      final bytes = await xFile.readAsBytes();
      debugLog.writeln('[Step 2] Captured ${bytes.length} bytes');

      // Step 3: Decode captured image.
      debugLog.writeln('[Step 3] Decoding image...');
      final capturedImage = img.decodeImage(bytes);

      if (capturedImage == null) {
        debugLog.writeln('[Step 3] FAILED: decodeImage returned null');
        setState(() => _statusMessage = 'Failed to decode captured image');
        _cameraController!.startImageStream(_processFrame);
        return;
      }
      debugLog.writeln(
          '[Step 3] Decoded: ${capturedImage.width}x${capturedImage.height}');

      img.Image croppedImage;

      // Step 4: Crop or pass through.
      if (_currentDetection != null && _currentDetection!.isValid) {
        debugLog.writeln('[Step 4] Detection available:');
        debugLog.writeln('  Confidence: '
            '${(_currentDetection!.confidence * 100).toStringAsFixed(1)}%');
        debugLog.writeln('  Corners (normalized): '
            '${_currentDetection!.corners}');

        final scaled = _currentDetection!.scaleToImage(
          capturedImage.width.toDouble(),
          capturedImage.height.toDouble(),
        );
        debugLog.writeln('  Corners (scaled to capture): ${scaled.corners}');

        debugLog.writeln('[Step 4] Running cubic polynomial crop...');
        croppedImage = CubicPolynomialCropper.crop(
          capturedImage,
          _currentDetection!,
        );
        debugLog.writeln(
            '[Step 4] Cropped result: ${croppedImage.width}x${croppedImage.height}');
      } else {
        debugLog.writeln('[Step 4] No valid detection — using full image');
        debugLog.writeln('  Detection: $_currentDetection');
        croppedImage = capturedImage;
      }

      debugLog.writeln('[Step 5] Navigating to result screen');
      print(debugLog.toString());

      if (mounted) {
        Navigator.of(context).push(
          MaterialPageRoute(
            builder: (_) => ResultScreen(
              original: capturedImage,
              cropped: croppedImage,
              detection: _currentDetection,
              debugLog: debugLog.toString(),
            ),
          ),
        ).then((_) {
          // Restart camera stream when returning.
          if (_cameraController != null &&
              _cameraController!.value.isInitialized) {
            _cameraController!.startImageStream(_processFrame);
            setState(
                () => _statusMessage = 'Point at a receipt or document');
          }
        });
      }
    } catch (e, stack) {
      debugLog.writeln('[ERROR] Capture error: $e');
      debugLog.writeln(stack.toString());
      print(debugLog.toString());
      setState(() => _statusMessage = 'Capture error: $e');
      if (_cameraController != null &&
          _cameraController!.value.isInitialized) {
        _cameraController!.startImageStream(_processFrame);
      }
    }
  }

  @override
  void dispose() {
    _cameraController?.dispose();
    _detector.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Receipt Scanner'),
        centerTitle: true,
      ),
      body: Stack(
        fit: StackFit.expand,
        children: [
          if (_isCameraReady && _cameraController != null)
            CameraPreview(_cameraController!)
          else
            const Center(child: CircularProgressIndicator()),
          if (_isCameraReady && _cameraController != null)
            CustomPaint(
              painter: EdgeOverlayPainter(
                detection: _currentDetection,
                imageSize: Size(
                  _cameraController!.value.previewSize?.height ?? 1,
                  _cameraController!.value.previewSize?.width ?? 1,
                ),
              ),
            ),
          Positioned(
            bottom: 100,
            left: 0,
            right: 0,
            child: Center(
              child: Container(
                padding:
                    const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
                decoration: BoxDecoration(
                  color: Colors.black54,
                  borderRadius: BorderRadius.circular(20),
                ),
                child: Text(
                  _statusMessage,
                  style: const TextStyle(color: Colors.white, fontSize: 14),
                ),
              ),
            ),
          ),
        ],
      ),
      floatingActionButton: _isCameraReady
          ? FloatingActionButton.large(
              onPressed: _captureAndCrop,
              child: const Icon(Icons.camera_alt, size: 36),
            )
          : null,
      floatingActionButtonLocation: FloatingActionButtonLocation.centerFloat,
    );
  }
}
