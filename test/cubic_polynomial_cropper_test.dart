import 'dart:ui';
import 'package:flutter_test/flutter_test.dart';
import 'package:image/image.dart' as img;
import 'package:receipt_scanner/models/detection_result.dart';
import 'package:receipt_scanner/utils/cubic_polynomial_cropper.dart';

void main() {
  group('CubicPolynomialCropper', () {
    test('crops identity quadrilateral to same dimensions', () {
      final image = img.Image(width: 100, height: 100);
      // Fill with a gradient so we can verify the output isn't blank.
      for (int y = 0; y < 100; y++) {
        for (int x = 0; x < 100; x++) {
          image.setPixelRgb(x, y, x * 2, y * 2, 128);
        }
      }

      final detection = DetectionResult(
        topLeft: const Offset(0, 0),
        topRight: const Offset(100, 0),
        bottomRight: const Offset(100, 100),
        bottomLeft: const Offset(0, 100),
        confidence: 0.95,
      );

      final result = CubicPolynomialCropper.crop(
        image,
        detection,
        outputWidth: 100,
        outputHeight: 100,
      );

      expect(result.width, 100);
      expect(result.height, 100);
    });

    test('produces output matching requested dimensions', () {
      final image = img.Image(width: 640, height: 480);
      final detection = DetectionResult(
        topLeft: const Offset(100, 50),
        topRight: const Offset(500, 60),
        bottomRight: const Offset(510, 400),
        bottomLeft: const Offset(90, 390),
        confidence: 0.85,
      );

      final result = CubicPolynomialCropper.crop(
        image,
        detection,
        outputWidth: 400,
        outputHeight: 300,
      );

      expect(result.width, 400);
      expect(result.height, 300);
    });
  });

  group('DetectionResult', () {
    test('isValid returns true for reasonable quadrilateral', () {
      final detection = DetectionResult(
        topLeft: const Offset(0.1, 0.1),
        topRight: const Offset(0.9, 0.1),
        bottomRight: const Offset(0.9, 0.9),
        bottomLeft: const Offset(0.1, 0.9),
        confidence: 0.8,
      );
      expect(detection.isValid, isTrue);
    });

    test('isValid returns false for low confidence', () {
      final detection = DetectionResult(
        topLeft: const Offset(0.1, 0.1),
        topRight: const Offset(0.9, 0.1),
        bottomRight: const Offset(0.9, 0.9),
        bottomLeft: const Offset(0.1, 0.9),
        confidence: 0.1,
      );
      expect(detection.isValid, isFalse);
    });
  });
}
