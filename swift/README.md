# wakewords Swift

Swift inference package for wakeword ONNX models exported by the Python `wakewords` tooling.

## What It Does

- Loads `model.onnx` and `labels.json` from an exported bundle.
- Reuses the same MFCC preprocessing pipeline as the JavaScript library.
- Runs local inference through ONNX Runtime.

## Add The Package

```swift
.package(path: "../swift")
```

Or depend on this subdirectory package from the repository.

This package also depends on Microsoft's ONNX Runtime Swift package:

```swift
.package(url: "https://github.com/microsoft/onnxruntime-swift-package-manager.git", exact: "1.20.0")
```

## Basic Usage

```swift
import Wakewords

let wakewords = try Wakewords.load(modelPath: "/path/to/models/model.onnx")

let result = try wakewords.predict(samples: samples, sampleRate: 16_000)
print(result.label, result.probability)
```

By default, `Wakewords.load(modelPath:)` looks for `labels.json` next to the model file and uses MFCC preprocessing before inference.

## API

```swift
public final class Wakewords {
    public static func load(
        modelPath: String,
        labelsPath: String? = nil,
        labels: [String]? = nil,
        inputFormat: Wakewords.InputFormat = .mfcc
    ) throws -> Wakewords

    public func predict(samples: [Float], sampleRate: Int) throws -> Prediction
}
```

## Notes

- The current Swift package targets exported models that expect MFCC inputs plus an optional audio length tensor.
- Tests in this package reuse the repository's existing MFCC fixture and ONNX test model for parity checks.
