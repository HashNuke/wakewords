import Foundation
@preconcurrency import OnnxRuntimeBindings

public final class Wakewords {
    public enum InputFormat: Sendable {
        case mfcc
    }

    private static let topProbabilityCount = 5

    public let modelPath: String
    public let labels: [String]

    private let inputFormat: InputFormat
    private let session: ORTSession
    private let inputNames: [String]
    private let outputNames: [String]

    public static func load(
        modelPath: String,
        labelsPath: String? = nil,
        labels: [String]? = nil,
        inputFormat: InputFormat = .mfcc,
        sessionOptions: ORTSessionOptions? = nil
    ) throws -> Wakewords {
        let resolvedLabels = try resolveLabels(modelPath: modelPath, labelsPath: labelsPath, labels: labels)
        return try Wakewords(
            modelPath: modelPath,
            labels: resolvedLabels,
            inputFormat: inputFormat,
            sessionOptions: sessionOptions
        )
    }

    public init(
        modelPath: String,
        labels: [String] = [],
        inputFormat: InputFormat = .mfcc,
        sessionOptions: ORTSessionOptions? = nil,
        env: ORTEnv? = nil
    ) throws {
        let ortEnv = try env ?? ORTEnv(loggingLevel: .warning)

        self.modelPath = modelPath
        self.labels = labels
        self.inputFormat = inputFormat
        self.session = try ORTSession(env: ortEnv, modelPath: modelPath, sessionOptions: sessionOptions)
        self.inputNames = try session.inputNames()
        self.outputNames = try session.outputNames()

        guard !self.outputNames.isEmpty else {
            throw Error.invalidModel("model has no outputs")
        }
    }

    public func predict(samples: [Float], sampleRate: Int) throws -> Prediction {
        guard sampleRate > 0 else {
            throw Error.invalidInput("sampleRate must be positive")
        }

        let normalizedSamples = sampleRate == Preprocessing.modelSampleRate
            ? samples
            : Preprocessing.resample(samples: samples, sourceRate: sampleRate)

        let feeds = try buildFeeds(samples: normalizedSamples)
        let outputs = try session.run(withInputs: feeds, outputNames: Set(outputNames), runOptions: nil)

        guard let output = outputs[outputNames[0]] else {
            throw Error.invalidModel("missing output tensor \(outputNames[0])")
        }

        let probabilities = try normalizeProbabilities(readOutputTensor(output))
        let resultLabels = labels.count == probabilities.count
            ? labels
            : (0..<probabilities.count).map { "class_\($0)" }

        let best = probabilities.enumerated().max(by: { $0.element < $1.element })
        guard let best else {
            throw Error.invalidModel("model returned an empty output tensor")
        }

        let byLabel = Dictionary(uniqueKeysWithValues: zip(resultLabels, probabilities))
        let top = zip(resultLabels, probabilities)
            .map { Prediction.ScoredLabel(label: $0.0, probability: $0.1) }
            .sorted { $0.probability > $1.probability }
            .prefix(Self.topProbabilityCount)

        return Prediction(
            label: resultLabels[best.offset],
            probability: best.element,
            probabilities: byLabel,
            topProbabilities: Array(top)
        )
    }
}

extension Wakewords {
    public enum Error: Swift.Error, LocalizedError {
        case invalidInput(String)
        case invalidModel(String)
        case unsupportedTensorType(String)
        case runtimeUnavailable

        public var errorDescription: String? {
            switch self {
            case let .invalidInput(message):
                return message
            case let .invalidModel(message):
                return message
            case let .unsupportedTensorType(message):
                return message
            case .runtimeUnavailable:
                return "failed to create the ONNX Runtime environment"
            }
        }
    }
}

private extension Wakewords {
    static func resolveLabels(modelPath: String, labelsPath: String?, labels: [String]?) throws -> [String] {
        if let labels {
            return labels
        }

        let path = labelsPath ?? URL(fileURLWithPath: modelPath).deletingLastPathComponent().appendingPathComponent("labels.json").path
        guard FileManager.default.fileExists(atPath: path) else {
            return []
        }

        let data = try Data(contentsOf: URL(fileURLWithPath: path))
        let decoded = try JSONDecoder().decode([String].self, from: data)
        return decoded
    }

    func buildFeeds(samples: [Float]) throws -> [String: ORTValue] {
        let mfcc = Preprocessing.mfccFeatures(samples: samples)
        var feeds: [String: ORTValue] = [:]

        for name in inputNames {
            let normalized = name.lowercased()
            if normalized.contains("length") {
                feeds[name] = try int64Tensor([Int64(mfcc.frames)], shape: [1])
                continue
            }

            if normalized.contains("audio") || normalized.contains("signal") || inputNames.count == 1 {
                switch inputFormat {
                case .mfcc:
                    feeds[name] = try floatTensor(mfcc.data, shape: [1, mfcc.features, mfcc.frames])
                }
                continue
            }

            throw Error.invalidModel("unsupported ONNX input: \(name)")
        }

        return feeds
    }

    func readOutputTensor(_ value: ORTValue) throws -> [Float] {
        let info = try value.tensorTypeAndShapeInfo()
        let data = try value.tensorData() as Data

        switch info.elementType {
        case .float:
            return data.toArray(type: Float.self)
        default:
            throw Error.unsupportedTensorType("unsupported output tensor type: \(info.elementType.rawValue)")
        }
    }

    func normalizeProbabilities(_ values: [Float]) throws -> [Float] {
        guard !values.isEmpty else {
            throw Error.invalidModel("model returned an empty output tensor")
        }

        let sum = values.reduce(0, +)
        let alreadyProbabilities = values.allSatisfy { $0 >= 0 && $0 <= 1 } && sum >= 0.99 && sum <= 1.01
        if alreadyProbabilities {
            return values.map { $0 / sum }
        }

        guard let maxValue = values.max() else {
            throw Error.invalidModel("model returned an empty output tensor")
        }

        let exp = values.map { Foundation.exp($0 - maxValue) }
        let total = exp.reduce(0, +)
        return exp.map { $0 / total }
    }

    func floatTensor(_ values: [Float], shape: [Int]) throws -> ORTValue {
        try ORTValue(
            tensorData: NSMutableData(data: Data(copyingBufferOf: values)),
            elementType: .float,
            shape: shape.map(NSNumber.init(value:))
        )
    }

    func int64Tensor(_ values: [Int64], shape: [Int]) throws -> ORTValue {
        try ORTValue(
            tensorData: NSMutableData(data: Data(copyingBufferOf: values)),
            elementType: .int64,
            shape: shape.map(NSNumber.init(value:))
        )
    }
}

private extension Data {
    init<T>(copyingBufferOf values: [T]) {
        self = values.withUnsafeBufferPointer { Data(buffer: $0) }
    }

    func toArray<T>(type: T.Type) -> [T] {
        guard count.isMultiple(of: MemoryLayout<T>.stride) else { return [] }
        return withUnsafeBytes { rawBuffer in
            Array(rawBuffer.bindMemory(to: T.self))
        }
    }
}
