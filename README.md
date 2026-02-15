# Receipt Scanner Demo

A Flutter Android app for testing receipt/document/invoice edge detection and cropping using **YOLOv8 + Cubic Polynomial** interpolation.

## Architecture

- **YOLOv8 Document Detection** — Runs a TFLite model to detect document corners in camera frames
- **Cubic Polynomial Cropping** — Uses Hermite-basis cubic interpolation (Coons patch) for perspective correction, producing smoother results than bilinear warping on curved document edges
- **Real-time Camera Preview** — Live edge overlay showing detected document boundaries

## Setup

### Prerequisites
- Flutter SDK 3.1+
- Android SDK with API 24+

### Adding Your YOLOv8 Model

1. Train a YOLOv8 model for document/receipt detection (e.g. using [Ultralytics](https://docs.ultralytics.com/))
2. Export to TFLite: `yolo export model=best.pt format=tflite`
3. Place the `.tflite` file at `assets/models/yolov8_doc_detector.tflite`

The model should output 4-corner coordinates (8 values) + confidence per detection.

### Running

```bash
flutter pub get
flutter run
```

Without a model loaded, the app still functions — you can capture photos manually and they'll pass through without cropping.

## Project Structure

```
lib/
├── main.dart                          # App entry point
├── models/
│   └── detection_result.dart          # Detection data model
├── screens/
│   ├── camera_screen.dart             # Camera preview + detection
│   └── result_screen.dart             # Cropped result viewer
├── services/
│   └── yolo_detector.dart             # YOLOv8 TFLite inference
├── utils/
│   └── cubic_polynomial_cropper.dart  # Cubic polynomial perspective correction
└── widgets/
    └── edge_overlay_painter.dart      # Detection overlay rendering
```

## CI/CD

GitHub Actions builds the debug APK on every push to `main` and `claude/**` branches. The APK artifact is uploaded and retained for 14 days.
