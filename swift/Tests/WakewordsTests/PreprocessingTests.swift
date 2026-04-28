import Foundation
import Testing
@testable import Wakewords

struct PreprocessingTests {
    @Test func mfccMatchesFixture() throws {
        let fixtureURL = repositoryRoot()
            .appendingPathComponent("tests/fixtures/mfcc-preprocessor.json")
        let fixtureData = try Data(contentsOf: fixtureURL)
        let fixture = try JSONDecoder().decode(MFCCFixture.self, from: fixtureData)
        let tolerance: Float = 3e-4

        for testCase in fixture.cases {
            let actual = Preprocessing.mfccFeatures(samples: testCase.samples)
            #expect(actual.features == testCase.mfccShape[0])
            #expect(actual.frames == testCase.mfccShape[1])
            #expect(actual.data.count == testCase.mfcc.count)

            var maxDelta: Float = 0
            for index in testCase.mfcc.indices {
                maxDelta = max(maxDelta, abs(actual.data[index] - testCase.mfcc[index]))
            }

            #expect(maxDelta <= tolerance, "\(testCase.name): max MFCC delta \(maxDelta) exceeds \(tolerance)")
        }
    }
}

private struct MFCCFixture: Decodable {
    let cases: [MFCCCase]
}

private struct MFCCCase: Decodable {
    let name: String
    let samples: [Float]
    let mfcc: [Float]
    let mfccShape: [Int]

    enum CodingKeys: String, CodingKey {
        case name
        case samples
        case mfcc
        case mfccShape = "mfcc_shape"
    }
}

private func repositoryRoot() -> URL {
    URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .deletingLastPathComponent()
}
