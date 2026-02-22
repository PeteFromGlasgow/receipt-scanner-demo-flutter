import 'dart:ui';

/// Represents the four corner points of a detected document.
class DetectionResult {
  final Offset topLeft;
  final Offset topRight;
  final Offset bottomRight;
  final Offset bottomLeft;
  final double confidence;

  const DetectionResult({
    required this.topLeft,
    required this.topRight,
    required this.bottomRight,
    required this.bottomLeft,
    required this.confidence,
  });

  List<Offset> get corners => [topLeft, topRight, bottomRight, bottomLeft];

  /// Returns a new DetectionResult with corners scaled to [width] x [height].
  DetectionResult scaleToImage(double width, double height) {
    return DetectionResult(
      topLeft: Offset(topLeft.dx * width, topLeft.dy * height),
      topRight: Offset(topRight.dx * width, topRight.dy * height),
      bottomRight: Offset(bottomRight.dx * width, bottomRight.dy * height),
      bottomLeft: Offset(bottomLeft.dx * width, bottomLeft.dy * height),
      confidence: confidence,
    );
  }

  /// Returns true if the detection seems valid (non-degenerate quadrilateral).
  /// Corners are expected to be in normalized (0-1) coordinates.
  bool get isValid {
    if (confidence < 0.3) return false;
    // Check that corners form a reasonable quadrilateral (in normalized space)
    final area = _shoelaceArea(corners);
    return area > 0.01; // at least 1% of normalized image area
  }

  static double _shoelaceArea(List<Offset> pts) {
    double area = 0;
    for (int i = 0; i < pts.length; i++) {
      final j = (i + 1) % pts.length;
      area += pts[i].dx * pts[j].dy;
      area -= pts[j].dx * pts[i].dy;
    }
    return (area / 2).abs();
  }

  @override
  String toString() =>
      'DetectionResult(confidence: ${confidence.toStringAsFixed(2)}, '
      'tl: $topLeft, tr: $topRight, br: $bottomRight, bl: $bottomLeft)';
}
