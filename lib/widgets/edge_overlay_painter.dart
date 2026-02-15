import 'package:flutter/material.dart';
import '../models/detection_result.dart';

/// Custom painter that draws the detected document edges over the camera preview.
class EdgeOverlayPainter extends CustomPainter {
  final DetectionResult? detection;
  final Size imageSize;

  EdgeOverlayPainter({required this.detection, required this.imageSize});

  @override
  void paint(Canvas canvas, Size size) {
    if (detection == null || !detection!.isValid) return;

    // Detection corners are normalized (0-1). Scale to canvas size,
    // then adjust for the imageSize-to-canvas aspect ratio.
    final scaleX = size.width / imageSize.width;
    final scaleY = size.height / imageSize.height;

    // Corners are normalized (0-1), so multiply by imageSize first, then by scale.
    Offset scale(Offset pt) =>
        Offset(pt.dx * imageSize.width * scaleX, pt.dy * imageSize.height * scaleY);

    final corners = detection!.corners.map(scale).toList();

    // Draw semi-transparent overlay outside the detected region.
    final overlayPaint = Paint()
      ..color = Colors.black.withAlpha(100)
      ..style = PaintingStyle.fill;

    final fullRect = Rect.fromLTWH(0, 0, size.width, size.height);
    final docPath = Path()
      ..moveTo(corners[0].dx, corners[0].dy)
      ..lineTo(corners[1].dx, corners[1].dy)
      ..lineTo(corners[2].dx, corners[2].dy)
      ..lineTo(corners[3].dx, corners[3].dy)
      ..close();

    // Clip out the document area and fill the rest.
    canvas.save();
    canvas.clipPath(docPath, doAntiAlias: true);
    canvas.restore();

    // Draw the outside overlay using path difference.
    final outerPath = Path()
      ..addRect(fullRect)
      ..addPath(docPath, Offset.zero);
    outerPath.fillType = PathFillType.evenOdd;
    canvas.drawPath(outerPath, overlayPaint);

    // Draw the document border.
    final borderPaint = Paint()
      ..color = Colors.tealAccent
      ..style = PaintingStyle.stroke
      ..strokeWidth = 3.0
      ..strokeJoin = StrokeJoin.round;

    canvas.drawPath(docPath, borderPaint);

    // Draw corner handles.
    final cornerPaint = Paint()
      ..color = Colors.tealAccent
      ..style = PaintingStyle.fill;

    for (final corner in corners) {
      canvas.drawCircle(corner, 8, cornerPaint);
    }

    // Draw confidence label.
    final textPainter = TextPainter(
      text: TextSpan(
        text: '${(detection!.confidence * 100).toStringAsFixed(0)}%',
        style: const TextStyle(
          color: Colors.tealAccent,
          fontSize: 16,
          fontWeight: FontWeight.bold,
        ),
      ),
      textDirection: TextDirection.ltr,
    );
    textPainter.layout();
    textPainter.paint(canvas, corners[0] + const Offset(12, -24));
  }

  @override
  bool shouldRepaint(covariant EdgeOverlayPainter old) {
    return old.detection != detection;
  }
}
