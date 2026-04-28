import Foundation

public enum Preprocessing {
    public static let modelSampleRate = 16_000
    static let nFFT = 512
    static let winLength = 400
    static let hopLength = 160
    public static let nMels = 64
    public static let nMFCC = 64
    static let logMelEpsilon: Float = 1e-6

    public struct MFCCFeatures: Sendable {
        public let data: [Float]
        public let features: Int
        public let frames: Int

        public init(data: [Float], features: Int, frames: Int) {
            self.data = data
            self.features = features
            self.frames = frames
        }
    }

    public static func mfccFeatures(samples: [Float]) -> MFCCFeatures {
        let cache = MFCCHelper.cache
        let padded = reflectPad(samples: samples, pad: nFFT / 2)
        let frames = ((padded.count - nFFT) / hopLength) + 1
        var mfcc = Array(repeating: Float.zero, count: nMFCC * frames)
        var spectrum = Array(repeating: Float.zero, count: nFFT / 2 + 1)
        var mel = Array(repeating: Float.zero, count: nMels)
        var frameSamples = Array(repeating: Float.zero, count: nFFT)

        for frame in 0..<frames {
            let offset = frame * hopLength
            for index in 0..<nFFT {
                frameSamples[index] = padded[offset + index] * cache.window[index]
            }

            for bin in 0...(nFFT / 2) {
                var real: Float = 0
                var imag: Float = 0
                let cosRow = cache.cos[bin]
                let sinRow = cache.sin[bin]
                for index in 0..<nFFT {
                    let value = frameSamples[index]
                    real += value * cosRow[index]
                    imag -= value * sinRow[index]
                }
                spectrum[bin] = real * real + imag * imag
            }

            for melIndex in 0..<nMels {
                var energy: Float = 0
                for filter in cache.melFilters[melIndex] {
                    energy += spectrum[filter.bin] * filter.weight
                }
                mel[melIndex] = log(energy + logMelEpsilon)
            }

            for coeff in 0..<nMFCC {
                var value: Float = 0
                for melIndex in 0..<nMels {
                    value += mel[melIndex] * cache.dct[melIndex][coeff]
                }
                mfcc[(coeff * frames) + frame] = value
            }
        }

        return MFCCFeatures(data: mfcc, features: nMFCC, frames: frames)
    }

    public static func resample(samples: [Float], sourceRate: Int, targetRate: Int = modelSampleRate) -> [Float] {
        guard sourceRate != targetRate else { return samples }
        let targetLength = max(1, Int(round(Double(samples.count * targetRate) / Double(sourceRate))))
        var result = Array(repeating: Float.zero, count: targetLength)
        let scale = Float(samples.count - 1) / Float(max(1, targetLength - 1))

        for index in 0..<targetLength {
            let position = Float(index) * scale
            let left = Int(floor(position))
            let right = min(left + 1, samples.count - 1)
            let weight = position - Float(left)
            result[index] = (samples[left] * (1 - weight)) + (samples[right] * weight)
        }

        return result
    }
}

private enum MFCCHelper {
    static let cache = Cache(
        window: paddedHannWindow(),
        cos: dftTable(using: Foundation.cos),
        sin: dftTable(using: Foundation.sin),
        melFilters: melFilterbank(),
        dct: dctMatrix()
    )

    struct Filter: Sendable {
        let bin: Int
        let weight: Float
    }

    struct Cache: Sendable {
        let window: [Float]
        let cos: [[Float]]
        let sin: [[Float]]
        let melFilters: [[Filter]]
        let dct: [[Float]]
    }

    static func paddedHannWindow() -> [Float] {
        var window = Array(repeating: Float.zero, count: Preprocessing.nFFT)
        let start = (Preprocessing.nFFT - Preprocessing.winLength) / 2
        for index in 0..<Preprocessing.winLength {
            let value = 0.5 - (0.5 * Foundation.cos((2 * Double.pi * Double(index)) / Double(Preprocessing.winLength)))
            window[start + index] = Float(value)
        }
        return window
    }

    static func dftTable(using function: (Double) -> Double) -> [[Float]] {
        (0...(Preprocessing.nFFT / 2)).map { bin in
            (0..<Preprocessing.nFFT).map { index in
                Float(function((2 * Double.pi * Double(bin * index)) / Double(Preprocessing.nFFT)))
            }
        }
    }

    static func melFilterbank() -> [[Filter]] {
        let melMin = hzToMel(0)
        let melMax = hzToMel(Float(Preprocessing.modelSampleRate) / 2)
        let points = (0..<(Preprocessing.nMels + 2)).map { index in
            let mel = melMin + ((melMax - melMin) * Float(index) / Float(Preprocessing.nMels + 1))
            return melToHz(mel)
        }
        let fftFreqs = (0...(Preprocessing.nFFT / 2)).map { bin in
            Float(bin * Preprocessing.modelSampleRate) / Float(Preprocessing.nFFT)
        }

        return (0..<Preprocessing.nMels).map { melIndex in
            let lower = points[melIndex]
            let center = points[melIndex + 1]
            let upper = points[melIndex + 2]

            return fftFreqs.enumerated().compactMap { bin, frequency in
                let weight: Float
                if frequency >= lower && frequency <= center {
                    weight = (frequency - lower) / (center - lower)
                } else if frequency > center && frequency <= upper {
                    weight = (upper - frequency) / (upper - center)
                } else {
                    weight = 0
                }
                return weight > 0 ? Filter(bin: bin, weight: weight) : nil
            }
        }
    }

    static func hzToMel(_ frequency: Float) -> Float {
        2595 * log10(1 + (frequency / 700))
    }

    static func melToHz(_ mel: Float) -> Float {
        700 * (pow(10, mel / 2595) - 1)
    }

    static func dctMatrix() -> [[Float]] {
        (0..<Preprocessing.nMels).map { mel in
            (0..<Preprocessing.nMFCC).map { coeff in
                var value = Foundation.cos((Double.pi / Double(Preprocessing.nMels)) * (Double(mel) + 0.5) * Double(coeff))
                if coeff == 0 {
                    value *= 1 / sqrt(2.0)
                }
                return Float(value * sqrt(2.0 / Double(Preprocessing.nMels)))
            }
        }
    }

    static func reflectPad(samples: [Float], pad: Int) -> [Float] {
        var result = Array(repeating: Float.zero, count: samples.count + (pad * 2))
        for index in 0..<pad {
            result[index] = samples[pad - index]
            result[pad + samples.count + index] = samples[samples.count - 2 - index]
        }
        result.replaceSubrange(pad..<(pad + samples.count), with: samples)
        return result
    }
}

private func paddedHannWindow() -> [Float] {
    MFCCHelper.paddedHannWindow()
}

private func dftTable(using function: (Double) -> Double) -> [[Float]] {
    MFCCHelper.dftTable(using: function)
}

private func melFilterbank() -> [[MFCCHelper.Filter]] {
    MFCCHelper.melFilterbank()
}

private func dctMatrix() -> [[Float]] {
    MFCCHelper.dctMatrix()
}

private func reflectPad(samples: [Float], pad: Int) -> [Float] {
    MFCCHelper.reflectPad(samples: samples, pad: pad)
}
