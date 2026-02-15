// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "SmartStackUI",
    platforms: [
        .macOS(.v13)
    ],
    products: [
        .executable(name: "SmartStackUI", targets: ["SmartStackUI"])
    ],
    targets: [
        .executableTarget(
            name: "SmartStackUI",
            path: "Sources/SmartStackUI"
        )
    ]
)
