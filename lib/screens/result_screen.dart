import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:image/image.dart' as img;
import 'package:path_provider/path_provider.dart';
import 'package:share_plus/share_plus.dart';
import 'dart:io';
import 'package:path/path.dart' as p;
import '../models/detection_result.dart';

class ResultScreen extends StatelessWidget {
  final img.Image original;
  final img.Image cropped;
  final DetectionResult? detection;

  const ResultScreen({
    super.key,
    required this.original,
    required this.cropped,
    this.detection,
  });

  Uint8List _encodeImage(img.Image image) {
    return Uint8List.fromList(img.encodePng(image));
  }

  Future<void> _saveImage(BuildContext context) async {
    try {
      final dir = await getApplicationDocumentsDirectory();
      final timestamp = DateTime.now().millisecondsSinceEpoch;
      final path = p.join(dir.path, 'receipt_$timestamp.png');
      final file = File(path);
      await file.writeAsBytes(_encodeImage(cropped));

      if (context.mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Saved to $path')),
        );
      }
    } catch (e) {
      if (context.mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Save failed: $e')),
        );
      }
    }
  }

  Future<void> _shareImage() async {
    final dir = await getApplicationDocumentsDirectory();
    final timestamp = DateTime.now().millisecondsSinceEpoch;
    final path = p.join(dir.path, 'receipt_share_$timestamp.png');
    final file = File(path);
    await file.writeAsBytes(_encodeImage(cropped));

    await Share.shareXFiles([XFile(path)], text: 'Scanned receipt');
  }

  @override
  Widget build(BuildContext context) {
    final croppedBytes = _encodeImage(cropped);
    final originalBytes = _encodeImage(original);

    return Scaffold(
      appBar: AppBar(
        title: const Text('Scan Result'),
        actions: [
          IconButton(
            icon: const Icon(Icons.save),
            onPressed: () => _saveImage(context),
            tooltip: 'Save',
          ),
          IconButton(
            icon: const Icon(Icons.share),
            onPressed: _shareImage,
            tooltip: 'Share',
          ),
        ],
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            if (detection != null) ...[
              Card(
                child: Padding(
                  padding: const EdgeInsets.all(12),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        'Detection Info',
                        style: Theme.of(context).textTheme.titleMedium,
                      ),
                      const SizedBox(height: 8),
                      Text(
                          'Confidence: ${(detection!.confidence * 100).toStringAsFixed(1)}%'),
                      Text('Method: YOLOv8 + Cubic Polynomial'),
                    ],
                  ),
                ),
              ),
              const SizedBox(height: 16),
            ],
            Text(
              'Cropped Result',
              style: Theme.of(context).textTheme.titleMedium,
            ),
            const SizedBox(height: 8),
            ClipRRect(
              borderRadius: BorderRadius.circular(8),
              child: Image.memory(
                croppedBytes,
                fit: BoxFit.contain,
              ),
            ),
            const SizedBox(height: 24),
            Text(
              'Original',
              style: Theme.of(context).textTheme.titleMedium,
            ),
            const SizedBox(height: 8),
            ClipRRect(
              borderRadius: BorderRadius.circular(8),
              child: Image.memory(
                originalBytes,
                fit: BoxFit.contain,
              ),
            ),
          ],
        ),
      ),
    );
  }
}
