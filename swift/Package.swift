// swift-tools-version: 6.0

import PackageDescription

let package = Package(
    name: "wakewords",
    platforms: [
        .iOS(.v15),
        .macOS(.v14),
        .macCatalyst(.v15),
    ],
    products: [
        .library(name: "wakewords", targets: ["Wakewords"]),
    ],
    dependencies: [
        .package(url: "https://github.com/microsoft/onnxruntime-swift-package-manager.git", exact: "1.20.0"),
    ],
    targets: [
        .target(
            name: "Wakewords",
            dependencies: [
                .product(name: "onnxruntime", package: "onnxruntime-swift-package-manager"),
            ]
        ),
        .testTarget(
            name: "WakewordsTests",
            dependencies: ["Wakewords"]
        ),
    ]
)
