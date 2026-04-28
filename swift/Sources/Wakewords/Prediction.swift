import Foundation

public struct Prediction: Sendable {
    public struct ScoredLabel: Sendable, Equatable {
        public let label: String
        public let probability: Float

        public init(label: String, probability: Float) {
            self.label = label
            self.probability = probability
        }
    }

    public let label: String
    public let probability: Float
    public let probabilities: [String: Float]
    public let topProbabilities: [ScoredLabel]

    public init(label: String, probability: Float, probabilities: [String: Float], topProbabilities: [ScoredLabel]) {
        self.label = label
        self.probability = probability
        self.probabilities = probabilities
        self.topProbabilities = topProbabilities
    }
}
