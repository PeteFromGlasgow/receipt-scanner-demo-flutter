import 'dart:math';
import 'dart:ui';
import 'package:image/image.dart' as img;
import '../models/detection_result.dart';

/// Fallback document detector that uses image processing (edge detection +
/// projection analysis) instead of a ML model. Used when the YOLO model is
/// not available.
class EdgeDetector {
  static const int _processingSize = 320;

  /// Detects a document in [image] using edge-based analysis.
  ///
  /// Returns a [DetectionResult] with normalized (0-1) corners, or null if
  /// no document-like region is found.
  DetectionResult? detect(img.Image image) {
    // Downscale for faster processing.
    final small = img.copyResize(image,
        width: _processingSize, height: _processingSize);

    // Convert to grayscale and apply edge detection.
    final gray = img.grayscale(small);
    final blurred = img.gaussianBlur(gray, radius: 2);
    final edges = img.sobel(blurred);

    // Build horizontal and vertical projection profiles from edge magnitudes.
    final w = edges.width;
    final h = edges.height;
    final hProfile = List<double>.filled(h, 0);
    final vProfile = List<double>.filled(w, 0);

    for (int y = 0; y < h; y++) {
      for (int x = 0; x < w; x++) {
        final lum = edges.getPixel(x, y).luminance.toDouble();
        hProfile[y] += lum;
        vProfile[x] += lum;
      }
    }

    // Normalize profiles.
    final hMax = hProfile.reduce(max);
    final vMax = vProfile.reduce(max);
    if (hMax == 0 || vMax == 0) return null;
    for (int i = 0; i < h; i++) {
      hProfile[i] /= hMax;
    }
    for (int i = 0; i < w; i++) {
      vProfile[i] /= vMax;
    }

    // Find document edges by scanning inward from each side for the first
    // strong edge (profile value above threshold).
    const threshold = 0.3;
    const margin = 0.05; // ignore outer 5% (often contains UI chrome)

    final top = _findEdgeForward(hProfile, threshold, margin);
    final bottom = _findEdgeBackward(hProfile, threshold, margin);
    final left = _findEdgeForward(vProfile, threshold, margin);
    final right = _findEdgeBackward(vProfile, threshold, margin);

    if (top == null || bottom == null || left == null || right == null) {
      return null;
    }

    // Convert to normalized (0-1) coordinates.
    final nTop = top / h;
    final nBottom = bottom / h;
    final nLeft = left / w;
    final nRight = right / w;

    // Validate the detected region is reasonable.
    final regionWidth = nRight - nLeft;
    final regionHeight = nBottom - nTop;
    if (regionWidth < 0.15 || regionHeight < 0.15) return null;
    if (regionWidth > 0.98 && regionHeight > 0.98) return null;

    // Compute a confidence score based on how distinct the edges are.
    final edgeStrength = _measureEdgeStrength(
        hProfile, vProfile, top, bottom, left, right);

    if (edgeStrength < 0.3) return null;

    return DetectionResult(
      topLeft: Offset(nLeft, nTop),
      topRight: Offset(nRight, nTop),
      bottomRight: Offset(nRight, nBottom),
      bottomLeft: Offset(nLeft, nBottom),
      confidence: edgeStrength.clamp(0.0, 1.0),
    );
  }

  /// Scans forward through [profile] to find the first index where the value
  /// exceeds [threshold], starting from [margin] fraction into the profile.
  static int? _findEdgeForward(
      List<double> profile, double threshold, double margin) {
    final start = (profile.length * margin).round();
    final end = profile.length ~/ 2;
    for (int i = start; i < end; i++) {
      if (profile[i] > threshold) return i;
    }
    return null;
  }

  /// Scans backward through [profile] to find the first index where the value
  /// exceeds [threshold], starting from [margin] fraction from the end.
  static int? _findEdgeBackward(
      List<double> profile, double threshold, double margin) {
    final start = profile.length - (profile.length * margin).round() - 1;
    final end = profile.length ~/ 2;
    for (int i = start; i > end; i--) {
      if (profile[i] > threshold) return i;
    }
    return null;
  }

  /// Measures how strong the detected edges are relative to the interior.
  static double _measureEdgeStrength(List<double> hProfile,
      List<double> vProfile, int top, int bottom, int left, int right) {
    // Average edge strength at the detected boundaries.
    double edgeSum = hProfile[top] + hProfile[bottom] +
        vProfile[left] + vProfile[right];
    double edgeAvg = edgeSum / 4;

    // Average profile value in the interior (should be lower for a document).
    double interiorSum = 0;
    int count = 0;
    final hMid = (top + bottom) ~/ 2;
    final vMid = (left + right) ~/ 2;
    final hRange = max(1, (bottom - top) ~/ 4);
    final vRange = max(1, (right - left) ~/ 4);

    for (int i = hMid - hRange; i <= hMid + hRange; i++) {
      if (i >= 0 && i < hProfile.length) {
        interiorSum += hProfile[i];
        count++;
      }
    }
    for (int i = vMid - vRange; i <= vMid + vRange; i++) {
      if (i >= 0 && i < vProfile.length) {
        interiorSum += vProfile[i];
        count++;
      }
    }
    final interiorAvg = count > 0 ? interiorSum / count : 0.5;

    // Confidence is higher when edges are strong and interior is relatively quiet.
    if (interiorAvg >= edgeAvg) return 0.3;
    return (edgeAvg - interiorAvg + 0.3).clamp(0.0, 1.0);
  }
}
