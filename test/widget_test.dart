import 'package:flutter_test/flutter_test.dart';
import 'package:receipt_scanner/main.dart';

void main() {
  testWidgets('App renders without crashing', (WidgetTester tester) async {
    await tester.pumpWidget(const ReceiptScannerApp());
    expect(find.text('Receipt Scanner'), findsOneWidget);
  });
}
