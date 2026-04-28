import Foundation
import Testing
@testable import Wakewords

struct InferenceTests {
    @Test func predictsFixtureWords() throws {
        let root = repositoryRoot()
        let modelPath = root.appendingPathComponent("tests/fixtures/model.onnx").path
        let labelsPath = root.appendingPathComponent("tests/fixtures/labels.json").path
        let wakewords = try Wakewords.load(modelPath: modelPath, labelsPath: labelsPath)

        let cases = [
            (label: "backward", file: "tests/fixtures/speech-commands/backward/017c4098_nohash_0.wav"),
            (label: "backward", file: "tests/fixtures/speech-commands/backward/017c4098_nohash_1.wav"),
            (label: "forward", file: "tests/fixtures/speech-commands/forward/012187a4_nohash_0.wav"),
            (label: "forward", file: "tests/fixtures/speech-commands/forward/017c4098_nohash_1.wav"),
            (label: "happy", file: "tests/fixtures/speech-commands/happy/00970ce1_nohash_1.wav"),
            (label: "happy", file: "tests/fixtures/speech-commands/happy/012187a4_nohash_0.wav"),
        ]

        for testCase in cases {
            let wav = try readWav(root.appendingPathComponent(testCase.file))
            let result = try wakewords.predict(samples: wav.samples, sampleRate: wav.sampleRate)
            #expect(result.label == testCase.label, "\(testCase.file): expected \(testCase.label), got \(result.label)")
            #expect(result.probability > 0.4, "\(testCase.file): expected confidence over 40%, got \(result.probability)")
        }
    }
}

private struct WAVInput {
    let samples: [Float]
    let sampleRate: Int
}

private func readWav(_ url: URL) throws -> WAVInput {
    let data = try Data(contentsOf: url)
    guard String(data: data.prefix(4), encoding: .ascii) == "RIFF",
          String(data: data.subdata(in: 8..<12), encoding: .ascii) == "WAVE" else {
        throw WAVError.invalidHeader(url.path)
    }

    var offset = 12
    var channels = 0
    var sampleRate = 0
    var bitsPerSample = 0
    var dataStart = 0
    var dataLength = 0

    while offset + 8 <= data.count {
        let chunkID = data.string(at: offset, length: 4)
        let chunkSize = Int(data.uint32LE(at: offset + 4))
        let chunkStart = offset + 8

        if chunkID == "fmt " {
            let audioFormat = Int(data.uint16LE(at: chunkStart))
            channels = Int(data.uint16LE(at: chunkStart + 2))
            sampleRate = Int(data.uint32LE(at: chunkStart + 4))
            bitsPerSample = Int(data.uint16LE(at: chunkStart + 14))
            guard audioFormat == 1 else {
                throw WAVError.unsupportedFormat(url.path)
            }
        } else if chunkID == "data" {
            dataStart = chunkStart
            dataLength = chunkSize
        }

        offset = chunkStart + chunkSize + (chunkSize % 2)
    }

    guard channels > 0, sampleRate > 0, bitsPerSample == 16, dataStart > 0, dataLength > 0 else {
        throw WAVError.unsupportedFormat(url.path)
    }

    let frameCount = dataLength / 2 / channels
    var samples = Array(repeating: Float.zero, count: frameCount)
    for frame in 0..<frameCount {
        var sample: Float = 0
        for channel in 0..<channels {
            let byteOffset = dataStart + ((frame * channels + channel) * 2)
            sample += Float(data.int16LE(at: byteOffset)) / 32_768
        }
        samples[frame] = sample / Float(channels)
    }

    return WAVInput(samples: samples, sampleRate: sampleRate)
}

private enum WAVError: Error {
    case invalidHeader(String)
    case unsupportedFormat(String)
}

private extension Data {
    func string(at offset: Int, length: Int) -> String {
        String(data: subdata(in: offset..<(offset + length)), encoding: .ascii) ?? ""
    }

    func uint16LE(at offset: Int) -> UInt16 {
        withUnsafeBytes { rawBuffer in
            rawBuffer.load(fromByteOffset: offset, as: UInt16.self).littleEndian
        }
    }

    func uint32LE(at offset: Int) -> UInt32 {
        withUnsafeBytes { rawBuffer in
            rawBuffer.load(fromByteOffset: offset, as: UInt32.self).littleEndian
        }
    }

    func int16LE(at offset: Int) -> Int16 {
        withUnsafeBytes { rawBuffer in
            rawBuffer.load(fromByteOffset: offset, as: Int16.self).littleEndian
        }
    }
}

private func repositoryRoot() -> URL {
    URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .deletingLastPathComponent()
}
