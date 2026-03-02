import AppKit
import Carbon.HIToolbox
import Foundation
import SwiftUI
import UniformTypeIdentifiers

private let stackRoot = "/Users/pranjal/garage/smart_stack"
private let guardedIngestScript = "\(stackRoot)/run_guarded_ingest.sh"
private let venvPython = "\(stackRoot)/.venv/bin/python"
private let mmCliScript = "\(stackRoot)/mm_cli.py"

enum SearchMode: String, CaseIterable, Identifiable {
    case semantic = "Semantic"
    case keyword = "Keyword"

    var id: String { rawValue }
}

enum TimelineGranularity: String, CaseIterable, Identifiable {
    case year
    case month
    case day

    var id: String { rawValue }

    var title: String {
        switch self {
        case .year: return "Year"
        case .month: return "Month"
        case .day: return "Day"
        }
    }

    var zoomIn: TimelineGranularity? {
        switch self {
        case .year: return .month
        case .month: return .day
        case .day: return nil
        }
    }

    var zoomOut: TimelineGranularity? {
        switch self {
        case .year: return nil
        case .month: return .year
        case .day: return .month
        }
    }
}

enum SourceFilter: String, CaseIterable, Identifiable {
    case all = "All"
    case image = "Images"

    var id: String { rawValue }
}

struct SearchResult: Identifiable, Decodable {
    let image_id: String?
    let source: String
    let filename: String
    let caption: String
    let tags: [String]
    var score: String
    let obsidian_path: String

    var id: String { "\(source)|\(filename)|\(obsidian_path)|\(score)" }

    var numericScore: Double {
        Double(score) ?? 0.0
    }

    var sourceTitle: String {
        source.capitalized
    }
}

// For CommandPalette bridging
struct SearchResultItem: Identifiable, Decodable {
    let id: String
    let filePath: String
    let caption: String
    let score: Double
}

// New Multimodal Response Structs
struct MultimodalResponse: Decodable {
    let routing_mode: String
    let results: [MultimodalResultItem]
}

struct MultimodalResultItem: Decodable {
    let image_id: String?
    let file_path: String
    let caption: String
    let tags: [String]
    let score: Double
    let source: String?
}

struct ChatResponse: Decodable {
    let answer: String
    let sources: [MultimodalResultItem] // Reusing this as it matches the dict structure
    let confidence: String
}

enum ChatRole: String {
    case user
    case assistant
}

struct ChatTurn: Identifiable {
    let id = UUID()
    let role: ChatRole
    let content: String
    let confidence: String?
}

struct ContextLensResponse: Decodable {
    let target: ContextLensTarget
    let rings: ContextLensRings
    let cluster_info: ContextLensClusterInfo?
    let entity_terms: [String]
    let meta: ContextLensMeta
}

struct ContextLensTarget: Decodable {
    let image_id: String
    let file_path: String
    let caption: String
    let tags: [String]
    let created_at: String
}

struct ContextLensRings: Decodable {
    let similarity: [ContextLensNeighbor]
    let cluster: [ContextLensNeighbor]
    let entity: [ContextLensNeighbor]
    let time: [ContextLensNeighbor]
}

struct ContextLensNeighbor: Decodable, Identifiable {
    let image_id: String
    let file_path: String
    let caption: String
    let tags: [String]
    let score: Double?
    let distance: Double?
    let entity_overlap: Int?
    let matched_terms: [String]?
    let created_at: String?
    let delta_days: Double?
    let delta_hours: Double?
    let relation: String?

    var id: String { image_id }
}

struct ContextLensClusterInfo: Decodable {
    let cluster_id: String
    let cluster_name: String
    let cluster_label: String
}

struct ContextLensMeta: Decodable {
    let top_k: Int
    let counts: [String: Int]
}

struct TimelineResponse: Decodable {
    let granularity: String
    let query: String
    let total_items: Int
    let bucket_count: Int
    let buckets: [TimelineBucket]
    let stats: TimelineStats
}

struct TimelineBucket: Decodable, Identifiable {
    let key: String
    let item_count: Int
    let start_at: String
    let end_at: String
    let sample_path: String

    var id: String { key }
}

struct TimelineStats: Decodable {
    let max_count: Int
    let avg_count: Double
    let peak_key: String
    let peak_count: Int
}

struct PhotosListResponse: Decodable {
    let total_indexed: Int
    let returned: Int
    let limit: Int
    let offset: Int
    let include_missing: Bool
    let path_checks_performed: Bool
    let items: [IndexedPhotoItem]
}

struct IndexedPhotoItem: Decodable {
    let image_id: String
    let file_path: String
    let caption: String
    let summary: String
    let tags: [String]
    let created_at: String
    let updated_at: String
    let is_stale: Bool
    let exists_on_disk: Bool
}

struct ClusterListResponse: Decodable {
    let count: Int
    let clusters: [MemoryCluster]
}

struct ClusterItemsResponse: Decodable {
    let cluster_id: String
    let count: Int
    let items: [ClusterItem]
}

struct ClusterSampleItem: Decodable {
    let image_id: String
    let file_path: String
    let caption: String
    let tags: [String]
    let distance: Double
}

struct MemoryCluster: Decodable, Identifiable {
    let cluster_id: String
    let name: String
    let topic_label: String
    let item_count: Int
    let created_at: String
    let updated_at: String
    let sample_item: ClusterSampleItem?

    var id: String { cluster_id }

    var displayTitle: String {
        let label = topic_label.trimmingCharacters(in: .whitespacesAndNewlines)
        return label.isEmpty ? name : label
    }
}

struct ClusterItem: Decodable, Identifiable {
    let image_id: String
    let file_path: String
    let caption: String
    let tags: [String]
    let created_at: String
    let distance: Double

    var id: String { image_id }
}

@MainActor
final class SmartStackViewModel: ObservableObject {
    static let shared = SmartStackViewModel()
    
    @Published var query: String = ""
    @Published var searchMode: SearchMode = .semantic
    @Published var sourceFilter: SourceFilter = .all
    @Published var topK: Int = 8
    @Published var minScore: Double = 0.0
    @Published var isChatMode: Bool = false // Toggle between Search/Chat
    
    @Published var chatTurns: [ChatTurn] = []
    @Published var chatAnswer: String = ""
    @Published var chatSources: [SearchResult] = []
    @Published var chatConfidence: String = ""
    @Published var attachedChatImage: SearchResult?

    @Published var isBusy: Bool = false
    @Published var results: [SearchResult] = []
    @Published var logs: String = "Ready."
    @Published var showSettings: Bool = false
    @Published var watchedFolders: [WatchedFolder] = []
    @Published var exclusions: [ExclusionPattern] = []
    @Published var showContextLens: Bool = false
    @Published var contextLens: ContextLensResponse?
    @Published var contextLensLoading: Bool = false
    @Published var contextLensError: String = ""
    @Published var showTimeline: Bool = false
    @Published var timelineGranularity: TimelineGranularity = .month
    @Published var timelineData: TimelineResponse?
    @Published var timelineLoading: Bool = false
    @Published var timelineError: String = ""
    @Published var showClusters: Bool = false
    @Published var clusters: [MemoryCluster] = []
    @Published var clusterItems: [ClusterItem] = []
    @Published var selectedClusterID: String = ""
    @Published var clustersLoading: Bool = false
    @Published var clustersError: String = ""
    @Published var visualQueryImagePath: String = ""
    
    // Window References
    var mainWindow: NSWindow?
    
    // Process Management
    private var activeSearchProcess: Process?
    private let commandQueue = DispatchQueue(label: "smartstack.ui.command.queue")
    private var lastSearchQuery: String = ""
    private let allowedImageExtensions: Set<String> = ["png", "jpg", "jpeg", "webp", "heic", "heif", "bmp", "tiff"]

    var hasVisualQueryImage: Bool {
        !visualQueryImagePath.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    var visualQueryFilename: String {
        guard hasVisualQueryImage else { return "" }
        return URL(fileURLWithPath: visualQueryImagePath).lastPathComponent
    }

    var hasAttachedChatImage: Bool {
        attachedChatImage != nil
    }

    var attachedChatImageFilename: String {
        guard let attachedChatImage else { return "" }
        return attachedChatImage.filename
    }

    var filteredResults: [SearchResult] {
        results.filter { row in
            let sourceOK: Bool
            switch sourceFilter {
            case .all:
                sourceOK = true
            case .image:
                sourceOK = row.source == "image"
            }
            return sourceOK && row.numericScore >= minScore
        }
    }
    
    // Command Palette Bridge
    var searchResults: [SearchResultItem] {
        filteredResults.prefix(10).map { res in
            SearchResultItem(
                id: res.id, 
                filePath: res.obsidian_path, 
                caption: res.caption, 
                score: res.numericScore
            )
        }
    }
    
    func performSearch(query: String) async {
        // Wrapper for async usage in new views
        // Since runSearch is sync/callback based, we just call it on main actor
        let normalized = query.trimmingCharacters(in: .whitespacesAndNewlines)
        guard normalized.count >= 3 else { return }
        guard normalized != lastSearchQuery else { return }
        self.query = normalized
        self.visualQueryImagePath = ""
        self.lastSearchQuery = normalized
        runSearch()
    }
    
    func ingestPath(_ path: String) {
        let args = [venvPython, mmCliScript, "ingest-path", path]
        runCommand(args: args, title: "Quick Ingest") { _, _, _ in }
    }

    func runSearch() {
        let q = query.trimmingCharacters(in: .whitespacesAndNewlines)
        let imagePath = visualQueryImagePath.trimmingCharacters(in: .whitespacesAndNewlines)
        let hasImageQuery = !imagePath.isEmpty

        if hasImageQuery {
            runMultimodalImageSearch(imagePath: imagePath)
            return
        }
        if let inlineImagePath = extractImagePathFromQuery(q) {
            visualQueryImagePath = inlineImagePath
            isChatMode = false
            appendLog("Visual query image set from search box: \(URL(fileURLWithPath: inlineImagePath).lastPathComponent)")
            runMultimodalImageSearch(imagePath: inlineImagePath)
            return
        }
        guard !q.isEmpty else {
            appendLog("Search query is empty.")
            return
        }
        switch searchMode {
        case .semantic:
            runMultimodalSearch(query: q)
        case .keyword:
            runKeywordSearch(query: q)
        }
    }

    func setVisualQueryImagePath(_ path: String) {
        let normalized = path.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !normalized.isEmpty else { return }
        let ext = URL(fileURLWithPath: normalized).pathExtension.lowercased()
        guard allowedImageExtensions.contains(ext) else {
            appendLog("Visual query ignored (unsupported file type): \(ext)")
            return
        }
        visualQueryImagePath = normalized
        appendLog("Visual query image set: \(URL(fileURLWithPath: normalized).lastPathComponent)")
    }

    func clearVisualQueryImage() {
        if hasVisualQueryImage {
            appendLog("Cleared visual query image.")
        }
        visualQueryImagePath = ""
    }

    func pickVisualQueryImage() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = true
        panel.canChooseDirectories = false
        panel.allowsMultipleSelection = false
        panel.title = "Select Query Image"
        panel.prompt = "Use Image"
        panel.allowedContentTypes = [.image]
        guard panel.runModal() == .OK, let url = panel.url else { return }
        setVisualQueryImagePath(url.path)
    }

    func pasteClipboardImageForSearch() {
        guard let path = saveClipboardImageToCache() else {
            appendLog("Clipboard has no image to paste.")
            return
        }
        setVisualQueryImagePath(path)
    }

    func pasteClipboardImageAndIngest() {
        guard let path = saveClipboardImageToCache() else {
            appendLog("Clipboard has no image to paste.")
            return
        }
        setVisualQueryImagePath(path)
        ingestPath(path)
        appendLog("Pasted image sent for ingest.")
    }

    private func saveClipboardImageToCache() -> String? {
        let pb = NSPasteboard.general

        if let urls = pb.readObjects(forClasses: [NSURL.self], options: [.urlReadingFileURLsOnly: true]) as? [URL] {
            for url in urls {
                let ext = url.pathExtension.lowercased()
                if allowedImageExtensions.contains(ext) {
                    return url.path
                }
            }
        }

        var image: NSImage? = nil
        if let objects = pb.readObjects(forClasses: [NSImage.self], options: nil) as? [NSImage], let first = objects.first {
            image = first
        } else if let tiffData = pb.data(forType: .tiff), let tiffImage = NSImage(data: tiffData) {
            image = tiffImage
        }

        guard let nsImage = image else { return nil }
        guard let tiff = nsImage.tiffRepresentation,
              let bitmap = NSBitmapImageRep(data: tiff),
              let pngData = bitmap.representation(using: .png, properties: [:]) else {
            return nil
        }

        let cacheDir = URL(fileURLWithPath: stackRoot).appendingPathComponent(".cache/pasted")
        try? FileManager.default.createDirectory(at: cacheDir, withIntermediateDirectories: true)
        let ts = Int(Date().timeIntervalSince1970 * 1000)
        let outURL = cacheDir.appendingPathComponent("pasted_\(ts).png")
        do {
            try pngData.write(to: outURL, options: .atomic)
            return outURL.path
        } catch {
            appendLog("Failed writing pasted image: \(error.localizedDescription)")
            return nil
        }
    }

    private func normalizePotentialImagePath(_ raw: String) -> String? {
        var value = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !value.isEmpty else { return nil }
        if (value.hasPrefix("\"") && value.hasSuffix("\"")) || (value.hasPrefix("'") && value.hasSuffix("'")) {
            value = String(value.dropFirst().dropLast())
        }

        var rawCandidates: [String] = []
        if value.lowercased().hasPrefix("file://"), let url = URL(string: value) {
            rawCandidates.append(url.path)
        } else {
            rawCandidates.append(value)
        }

        for rawPath in rawCandidates {
            let expanded = (rawPath as NSString).expandingTildeInPath
            let candidates: [String]
            if expanded.hasPrefix("/") {
                candidates = [expanded]
            } else {
                candidates = [
                    URL(fileURLWithPath: stackRoot).appendingPathComponent(expanded).path,
                    URL(fileURLWithPath: FileManager.default.currentDirectoryPath).appendingPathComponent(expanded).path,
                ]
            }
            for path in candidates {
                var isDir: ObjCBool = false
                if FileManager.default.fileExists(atPath: path, isDirectory: &isDir), !isDir.boolValue {
                    let ext = URL(fileURLWithPath: path).pathExtension.lowercased()
                    if allowedImageExtensions.contains(ext) {
                        return path
                    }
                }
            }
        }
        return nil
    }

    private func extractImagePathFromQuery(_ text: String) -> String? {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return nil }
        if let direct = normalizePotentialImagePath(trimmed) {
            return direct
        }

        let quotedPattern = "\"([^\"]+)\"|'([^']+)'"
        if let regex = try? NSRegularExpression(pattern: quotedPattern) {
            let ns = trimmed as NSString
            let matches = regex.matches(in: trimmed, range: NSRange(location: 0, length: ns.length))
            for match in matches {
                if match.numberOfRanges > 1, match.range(at: 1).location != NSNotFound {
                    let candidate = ns.substring(with: match.range(at: 1))
                    if let path = normalizePotentialImagePath(candidate) { return path }
                }
                if match.numberOfRanges > 2, match.range(at: 2).location != NSNotFound {
                    let candidate = ns.substring(with: match.range(at: 2))
                    if let path = normalizePotentialImagePath(candidate) { return path }
                }
            }
        }

        for token in trimmed.split(whereSeparator: { $0.isWhitespace }) {
            let candidate = String(token).trimmingCharacters(in: CharacterSet(charactersIn: ",;"))
            if let path = normalizePotentialImagePath(candidate) {
                return path
            }
        }
        return nil
    }

    private func writeChatHistoryToCache(_ json: String) -> String? {
        let cacheDir = URL(fileURLWithPath: stackRoot).appendingPathComponent(".cache/chat")
        try? FileManager.default.createDirectory(at: cacheDir, withIntermediateDirectories: true)
        let ts = Int(Date().timeIntervalSince1970 * 1000)
        let outURL = cacheDir.appendingPathComponent("history_\(ts).json")
        guard let data = json.data(using: .utf8) else { return nil }
        do {
            try data.write(to: outURL, options: .atomic)
            return outURL.path
        } catch {
            appendLog("Failed writing chat history cache: \(error.localizedDescription)")
            return nil
        }
    }

    func attachImageForChat(_ result: SearchResult) {
        guard result.source == "image" else { return }
        attachedChatImage = result
        isChatMode = true
        appendLog("Attached image to chat: \(result.filename)")
    }

    func clearAttachedChatImage() {
        guard attachedChatImage != nil else { return }
        attachedChatImage = nil
        appendLog("Cleared attached chat image.")
    }

    func clearChatConversation() {
        chatTurns = []
        chatAnswer = ""
        chatSources = []
        chatConfidence = ""
        appendLog("Cleared chat conversation.")
    }

    func runChat() {
        let q = query.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !q.isEmpty else { return }
        if let inlineImagePath = extractImagePathFromQuery(q) {
            setVisualQueryImagePath(inlineImagePath)
            isChatMode = false
            query = ""
            appendLog("Detected image path in input. Running visual similarity search.")
            runSearch()
            return
        }

        let historyPayload: [[String: String]] = chatTurns.suffix(6).map { turn in
            [
                "role": turn.role.rawValue,
                "content": String(turn.content.prefix(400)),
            ]
        }
        var historyJSONString = ""
        if let historyData = try? JSONSerialization.data(withJSONObject: historyPayload, options: []),
           let asString = String(data: historyData, encoding: .utf8) {
            historyJSONString = asString
        }

        chatTurns.append(ChatTurn(role: .user, content: q, confidence: nil))
        query = ""

        var args = [
            venvPython,
            mmCliScript,
            "chat",
            q,
            "--json",
            "-n",
            "\(max(1, min(6, topK)))",
        ]
        if let attached = attachedChatImage {
            let imageID = attached.image_id?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
            if !imageID.isEmpty {
                args += ["--image-id", imageID]
            } else {
                args += ["--file-path", attached.obsidian_path]
            }
        }
        if !historyJSONString.isEmpty {
            if let historyPath = writeChatHistoryToCache(historyJSONString) {
                args += ["--history-file", historyPath]
            } else {
                args += ["--history-json", historyJSONString]
            }
        }

        runCommand(args: args, title: "Chat") { output, _, code in
            guard code == 0 else {
                self.appendLog("Chat failed code \(code).")
                self.chatAnswer = "Error: Chat failed. Check logs."
                self.chatTurns.append(ChatTurn(role: .assistant, content: self.chatAnswer, confidence: "Low"))
                return
            }
            
            guard let data = output.data(using: .utf8) else { return }
            do {
                let resp = try JSONDecoder().decode(ChatResponse.self, from: data)
                self.chatAnswer = resp.answer
                self.chatConfidence = resp.confidence
                
                // Map sources to SearchResult for display cards
                self.chatSources = resp.sources.map { item -> SearchResult in
                    let url = URL(fileURLWithPath: item.file_path)
                    return SearchResult(
                        image_id: item.image_id,
                        source: "image", 
                        filename: url.lastPathComponent,
                        caption: item.caption,
                        tags: item.tags,
                        score: String(format: "%.4f", item.score),
                        obsidian_path: item.file_path
                    )
                }
                self.chatTurns.append(
                    ChatTurn(
                        role: .assistant,
                        content: resp.answer,
                        confidence: resp.confidence
                    )
                )
            } catch {
                self.appendLog("Chat Parse Error: \(error)")
                self.chatAnswer = "Error parsing response."
                self.chatTurns.append(ChatTurn(role: .assistant, content: self.chatAnswer, confidence: "Low"))
            }
        }
    }

    func runSafeReprocess() {
        let args = [guardedIngestScript, "--safe-reprocess"]
        runCommand(args: args, title: "Safe Reprocess") { _, _, _ in }
    }

    func runInboxIngest() {
        let args = [guardedIngestScript]
        runCommand(args: args, title: "Inbox Ingest") { _, _, _ in }
    }

    // MARK: - Index-in-Place

    func runIngestPath() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = true
        panel.canChooseDirectories = true
        panel.allowsMultipleSelection = false
        panel.title = "Select File or Folder to Ingest"
        panel.prompt = "Ingest"

        guard panel.runModal() == .OK, let url = panel.url else { return }
        let path = url.path
        let args = [venvPython, mmCliScript, "ingest-path", path]
        runCommand(args: args, title: "Ingest Path") { output, _, code in
            if code == 0 {
                self.appendLog("Ingested: \(path)")
            }
        }
    }

    func runRescan() {
        let args = [venvPython, mmCliScript, "rescan"]
        runCommand(args: args, title: "Rescan Changed") { _, _, _ in }
    }

    func runRescanAll() {
        let args = [venvPython, mmCliScript, "rescan-all"]
        runCommand(args: args, title: "Rescan Watched") { _, _, _ in }
    }

    // MARK: - Watched Folders

    func loadWatchedFolders() {
        let args = [venvPython, mmCliScript, "watch-list"]
        runCommand(args: args, title: "Load Folders") { output, _, code in
            guard code == 0, let data = output.data(using: .utf8) else { return }
            do {
                self.watchedFolders = try JSONDecoder().decode([WatchedFolder].self, from: data)
            } catch {
                self.appendLog("Parse watched folders: \(error)")
            }
        }
    }

    func addWatchedFolder() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.allowsMultipleSelection = false
        panel.title = "Select Folder to Watch"
        panel.prompt = "Watch"

        guard panel.runModal() == .OK, let url = panel.url else { return }
        let args = [venvPython, mmCliScript, "watch-add", url.path]
        runCommand(args: args, title: "Watch Add") { _, _, code in
            if code == 0 { self.loadWatchedFolders() }
        }
    }

    func removeWatchedFolder(_ path: String) {
        let args = [venvPython, mmCliScript, "watch-remove", path]
        runCommand(args: args, title: "Watch Remove") { _, _, code in
            if code == 0 { self.loadWatchedFolders() }
        }
    }

    func toggleWatchedFolder(_ path: String) {
        let args = [venvPython, mmCliScript, "watch-toggle", path]
        runCommand(args: args, title: "Watch Toggle") { _, _, code in
            if code == 0 { self.loadWatchedFolders() }
        }
    }

    func loadExclusions() {
        let args = [venvPython, mmCliScript, "exclude-list"]
        runCommand(args: args, title: "Load Exclusions") { output, _, code in
            guard code == 0, let data = output.data(using: .utf8) else { return }
            do {
                self.exclusions = try JSONDecoder().decode([ExclusionPattern].self, from: data)
            } catch {
                self.appendLog("Parse exclusions: \(error)")
            }
        }
    }

    func addExclusion(_ pattern: String) {
        let args = [venvPython, mmCliScript, "exclude-add", pattern]
        runCommand(args: args, title: "Exclude Add") { _, _, code in
            if code == 0 { self.loadExclusions() }
        }
    }

    func removeExclusion(_ pattern: String) {
        let args = [venvPython, mmCliScript, "exclude-remove", pattern]
        runCommand(args: args, title: "Exclude Remove") { _, _, code in
            if code == 0 { self.loadExclusions() }
        }
    }

    func open(_ result: SearchResult) {
        let path = result.obsidian_path.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !path.isEmpty else {
            appendLog("No file path available for selected row.")
            return
        }
        NSWorkspace.shared.open(URL(fileURLWithPath: path))
    }

    func runContextLens(for result: SearchResult, topK: Int = 8) {
        let imageID = result.image_id?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        let path = result.obsidian_path.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !imageID.isEmpty || !path.isEmpty else {
            appendLog("Context Lens: no image_id/file_path.")
            return
        }
        contextLensLoading = true
        contextLensError = ""
        contextLens = nil
        showContextLens = true
        var args = [venvPython, mmCliScript, "context-lens"]
        if !imageID.isEmpty {
            args += ["--image-id", imageID]
        } else {
            args += ["--file-path", path]
        }
        args += ["-n", "\(max(1, topK))"]

        runCommand(args: args, title: "Context Lens") { output, stderr, code in
            self.contextLensLoading = false
            guard code == 0 else {
                let err = stderr.trimmingCharacters(in: .whitespacesAndNewlines)
                if err.contains("Target image not found in index") {
                    self.contextLensError = "This item is not in multimodal index yet. Run ingest, then retry."
                } else if !err.isEmpty {
                    self.contextLensError = "Context Lens failed: \(err)"
                } else {
                    self.contextLensError = "Context Lens failed with code \(code)."
                }
                return
            }
            guard let data = output.data(using: .utf8) else {
                self.contextLensError = "Context Lens output is empty."
                return
            }
            do {
                self.contextLens = try JSONDecoder().decode(ContextLensResponse.self, from: data)
                self.contextLensError = ""
            } catch {
                self.contextLensError = "Context Lens parse error: \(error.localizedDescription)"
                self.appendLog("Context Lens parse error: \(error)")
            }
        }
    }

    func openTimeline() {
        showTimeline = true
        runTimeline(granularity: timelineGranularity)
    }

    func setTimelineGranularity(_ granularity: TimelineGranularity) {
        guard timelineGranularity != granularity else { return }
        timelineGranularity = granularity
        runTimeline(granularity: granularity)
    }

    func runTimeline(granularity: TimelineGranularity? = nil) {
        let g = granularity ?? timelineGranularity
        timelineLoading = true
        timelineError = ""
        timelineData = nil

        let q = query.trimmingCharacters(in: .whitespacesAndNewlines)
        var args = [
            venvPython,
            mmCliScript,
            "timeline",
            "--granularity",
            g.rawValue,
            "--limit",
            "480",
        ]
        if !q.isEmpty {
            args += ["--query", q]
        }

        runCommand(args: args, title: "Semantic Timeline") { output, stderr, code in
            self.timelineLoading = false
            guard code == 0 else {
                let err = stderr.trimmingCharacters(in: .whitespacesAndNewlines)
                self.timelineError = err.isEmpty ? "Timeline failed with code \(code)." : err
                return
            }
            guard let data = output.data(using: .utf8) else {
                self.timelineError = "Timeline output is empty."
                return
            }
            do {
                self.timelineData = try JSONDecoder().decode(TimelineResponse.self, from: data)
                self.timelineGranularity = g
                self.timelineError = ""
            } catch {
                self.timelineError = "Timeline parse error: \(error.localizedDescription)"
            }
        }
    }

    func runAllPhotos(limit: Int = 180) {
        isChatMode = false
        sourceFilter = .image
        minScore = 0.0
        query = ""
        visualQueryImagePath = ""

        let args = [
            venvPython,
            mmCliScript,
            "photos-list",
            "--limit",
            "\(max(1, limit))",
        ]

        runCommand(args: args, title: "All Photos") { output, stderr, code in
            guard code == 0 else {
                let err = stderr.trimmingCharacters(in: .whitespacesAndNewlines)
                self.appendLog("All Photos failed code \(code). \(err)")
                return
            }
            guard !output.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
                self.appendLog("All Photos output is empty.")
                return
            }
            do {
                let resp = try self.decodePhotosListResponse(from: output)
                self.results = resp.items.map { item in
                    let url = URL(fileURLWithPath: item.file_path)
                    return SearchResult(
                        image_id: item.image_id,
                        source: "image",
                        filename: url.lastPathComponent,
                        caption: item.caption.isEmpty ? item.summary : item.caption,
                        tags: item.tags,
                        score: item.is_stale ? "0.3500" : "1.0000",
                        obsidian_path: item.file_path
                    )
                }
                self.appendLog(
                    "All Photos debug: results=\(self.results.count), filtered=\(self.filteredResults.count), "
                    + "isChatMode=\(self.isChatMode), minScore=\(String(format: "%.2f", self.minScore)), filter=\(self.sourceFilter.rawValue)"
                )
                if resp.path_checks_performed {
                    let missingCount = resp.items.filter { !$0.exists_on_disk }.count
                    self.appendLog(
                        "All Photos: loaded \(resp.returned)/\(resp.total_indexed) indexed photos."
                        + (missingCount > 0 ? " Missing on disk: \(missingCount)." : "")
                    )
                } else {
                    self.appendLog("All Photos: loaded \(resp.returned)/\(resp.total_indexed) indexed photos (fast mode).")
                }
            } catch {
                self.appendLog("All Photos parse error: \(error.localizedDescription)")
                self.appendLog("All Photos output preview: \(self.truncatedLogLine(output, maxChars: 700))")
            }
        }
    }

    func openClusters() {
        showClusters = true
        if clusters.isEmpty && !clustersLoading {
            loadClusters()
        }
    }

    func runAutoCluster(nClusters: Int = 20) {
        clustersLoading = true
        clustersError = ""
        let args = [
            venvPython,
            mmCliScript,
            "cluster",
            "--auto",
            "--n-clusters",
            "\(max(1, nClusters))",
        ]
        runCommand(args: args, title: "Auto Cluster") { _, stderr, code in
            guard code == 0 else {
                self.clustersLoading = false
                let err = stderr.trimmingCharacters(in: .whitespacesAndNewlines)
                self.clustersError = err.isEmpty ? "Auto Cluster failed with code \(code)." : err
                return
            }
            self.loadClusters()
        }
    }

    func loadClusters(limit: Int = 120, minItems: Int = 1) {
        clustersLoading = true
        clustersError = ""
        let args = [
            venvPython,
            mmCliScript,
            "cluster",
            "--list",
            "--limit",
            "\(max(1, limit))",
            "--min-items",
            "\(max(0, minItems))",
        ]
        runCommand(args: args, title: "Load Clusters") { output, stderr, code in
            guard code == 0 else {
                self.clustersLoading = false
                let err = stderr.trimmingCharacters(in: .whitespacesAndNewlines)
                self.clustersError = err.isEmpty ? "Load Clusters failed with code \(code)." : err
                return
            }
            guard let data = output.data(using: .utf8) else {
                self.clustersLoading = false
                self.clustersError = "Load Clusters output is empty."
                return
            }
            do {
                let resp = try JSONDecoder().decode(ClusterListResponse.self, from: data)
                self.clusters = resp.clusters
                if resp.clusters.isEmpty {
                    self.selectedClusterID = ""
                    self.clusterItems = []
                    self.clustersLoading = false
                    return
                }
                let selectedID: String
                if resp.clusters.contains(where: { $0.cluster_id == self.selectedClusterID }) {
                    selectedID = self.selectedClusterID
                } else {
                    selectedID = resp.clusters[0].cluster_id
                }
                self.selectedClusterID = selectedID
                self.loadClusterItems(clusterID: selectedID)
            } catch {
                self.clustersLoading = false
                self.clustersError = "Load Clusters parse error: \(error.localizedDescription)"
            }
        }
    }

    func loadClusterItems(clusterID: String, limit: Int = 240) {
        let normalized = clusterID.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !normalized.isEmpty else {
            clusterItems = []
            clustersLoading = false
            return
        }
        clustersLoading = true
        clustersError = ""
        let args = [
            venvPython,
            mmCliScript,
            "cluster",
            "--items",
            normalized,
            "--limit",
            "\(max(1, limit))",
        ]
        runCommand(args: args, title: "Load Cluster Items") { output, stderr, code in
            self.clustersLoading = false
            guard code == 0 else {
                let err = stderr.trimmingCharacters(in: .whitespacesAndNewlines)
                self.clustersError = err.isEmpty ? "Load Cluster Items failed with code \(code)." : err
                return
            }
            guard let data = output.data(using: .utf8) else {
                self.clustersError = "Load Cluster Items output is empty."
                return
            }
            do {
                let resp = try JSONDecoder().decode(ClusterItemsResponse.self, from: data)
                self.selectedClusterID = resp.cluster_id
                self.clusterItems = resp.items
                self.clustersError = ""
            } catch {
                self.clustersError = "Load Cluster Items parse error: \(error.localizedDescription)"
            }
        }
    }

    private func runMultimodalSearch(query: String) {
        // mm_cli.py search "query" -n topK --mode semantic --json
        let args = [
            venvPython,
            mmCliScript,
            "search",
            query,
            "-n",
            "\(max(1, topK))",
            "--mode",
            "semantic",
            "--json"
        ]

        runMultimodalSearchCommand(args: args, title: "Multimodal Search")
    }

    private func runMultimodalImageSearch(imagePath: String) {
        let args = [
            venvPython,
            mmCliScript,
            "search",
            "--image-path",
            imagePath,
            "-n",
            "\(max(1, topK))",
            "--mode",
            "semantic",
            "--json",
        ]
        runMultimodalSearchCommand(args: args, title: "Visual Search")
    }

    private func runMultimodalSearchCommand(args: [String], title: String) {
        runCommand(args: args, title: title) { output, stderr, code in
            guard code == 0 else {
                let err = stderr.trimmingCharacters(in: .whitespacesAndNewlines)
                self.appendLog("\(title) failed code \(code). \(err)")
                return
            }
            guard let data = output.data(using: .utf8) else { return }
            do {
                let resp = try JSONDecoder().decode(MultimodalResponse.self, from: data)
                let mapped = resp.results.map { item -> SearchResult in
                    let url = URL(fileURLWithPath: item.file_path)
                    return SearchResult(
                        image_id: item.image_id,
                        source: "image",
                        filename: url.lastPathComponent,
                        caption: item.caption,
                        tags: item.tags,
                        score: String(format: "%.4f", item.score),
                        obsidian_path: item.file_path
                    )
                }
                self.results = mapped
                self.appendLog("\(title): \(resp.routing_mode) mode, \(mapped.count) results.")
            } catch {
                self.appendLog("MM Parse Error: \(error)")
            }
        }
    }

    private func runKeywordSearch(query: String) {
        let args = [
            venvPython,
            mmCliScript,
            "search",
            query,
            "-n",
            "\(max(1, topK))",
            "--mode",
            "auto",
            "--semantic-fallback-threshold",
            "0",
            "--json",
        ]
        runMultimodalSearchCommand(args: args, title: "Keyword Search")
    }

    private func decodePhotosListResponse(from raw: String) throws -> PhotosListResponse {
        let decoder = JSONDecoder()
        if let data = raw.data(using: .utf8),
           let parsed = try? decoder.decode(PhotosListResponse.self, from: data) {
            return parsed
        }

        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        if let start = trimmed.firstIndex(of: "{"),
           let end = trimmed.lastIndex(of: "}"),
           start < end {
            let jsonBlob = String(trimmed[start...end])
            if let data = jsonBlob.data(using: .utf8) {
                return try decoder.decode(PhotosListResponse.self, from: data)
            }
        }

        throw NSError(
            domain: "SmartStackUI",
            code: 1,
            userInfo: [NSLocalizedDescriptionKey: "Unable to decode photos-list response"]
        )
    }

    private func truncatedLogLine(_ text: String, maxChars: Int = 4000) -> String {
        guard text.count > maxChars else { return text }
        let head = String(text.prefix(maxChars))
        let omitted = max(0, text.count - maxChars)
        return "\(head)\n...[omitted \(omitted) chars]"
    }

    private func runCommand(args: [String], title: String, completion: @escaping (String, String, Int32) -> Void) {
        guard !args.isEmpty else { return }
        
        // Kill existing search if this is a new search/chat
        if title.contains("Search") || title.contains("Chat") {
            activeSearchProcess?.terminate()
            activeSearchProcess = nil
        }

        isBusy = true
        appendLog("\n[\(title)] $ \(args.joined(separator: " "))")

        commandQueue.async {
            let process = Process()
            process.currentDirectoryURL = URL(fileURLWithPath: stackRoot)
            process.executableURL = URL(fileURLWithPath: args[0])
            process.arguments = Array(args.dropFirst())

            if title.contains("Search") || title.contains("Chat") {
                 DispatchQueue.main.async {
                     self.activeSearchProcess = process
                 }
            }

            let outPipe = Pipe()
            let errPipe = Pipe()
            process.standardOutput = outPipe
            process.standardError = errPipe

            let outLock = NSLock()
            let errLock = NSLock()
            let outBuffer = NSMutableData()
            let errBuffer = NSMutableData()

            outPipe.fileHandleForReading.readabilityHandler = { handle in
                let chunk = handle.availableData
                guard !chunk.isEmpty else { return }
                outLock.lock()
                outBuffer.append(chunk)
                outLock.unlock()
            }
            errPipe.fileHandleForReading.readabilityHandler = { handle in
                let chunk = handle.availableData
                guard !chunk.isEmpty else { return }
                errLock.lock()
                errBuffer.append(chunk)
                errLock.unlock()
            }

            do {
                try process.run()
                process.waitUntilExit()

                outPipe.fileHandleForReading.readabilityHandler = nil
                errPipe.fileHandleForReading.readabilityHandler = nil

                let outTail = outPipe.fileHandleForReading.readDataToEndOfFile()
                let errTail = errPipe.fileHandleForReading.readDataToEndOfFile()
                if !outTail.isEmpty {
                    outLock.lock()
                    outBuffer.append(outTail)
                    outLock.unlock()
                }
                if !errTail.isEmpty {
                    errLock.lock()
                    errBuffer.append(errTail)
                    errLock.unlock()
                }

                outLock.lock()
                let outData = Data(referencing: outBuffer)
                outLock.unlock()
                errLock.lock()
                let errData = Data(referencing: errBuffer)
                errLock.unlock()
                let outText = String(data: outData, encoding: .utf8) ?? ""
                let errText = String(data: errData, encoding: .utf8) ?? ""
                let combined = [outText, errText].filter { !$0.isEmpty }.joined(separator: "\n")

                DispatchQueue.main.async {
                    if self.activeSearchProcess == process {
                        self.activeSearchProcess = nil
                    }
                    self.isBusy = false
                    // Log output (combined) for debugging visibility
                    if !combined.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                        if title == "All Photos" {
                            let trimmedErr = errText.trimmingCharacters(in: .whitespacesAndNewlines)
                            if !trimmedErr.isEmpty {
                                self.appendLog(trimmedErr)
                            }
                            self.appendLog("[All Photos] Raw JSON output suppressed (\(outText.utf8.count) bytes).")
                        } else {
                            self.appendLog(self.truncatedLogLine(combined, maxChars: 10000))
                        }
                    }
                    // Return distinct streams for robust parsing
                    completion(outText, errText, process.terminationStatus)
                }
            } catch {
                DispatchQueue.main.async {
                    if self.activeSearchProcess == process {
                        self.activeSearchProcess = nil
                    }
                    self.isBusy = false
                    self.appendLog("[\(title)] Failed to run command: \(error)")
                    completion("", "", -1)
                }
            }
        }
    }

    private func appendLog(_ line: String) {
        logs += logs.isEmpty ? line : "\n\(line)"
    }

    func clearCommandLog() {
        logs = "Ready."
    }

    func runEmergencyMemoryKillSwitch() {
        appendLog("[Emergency Kill] Triggered. Terminating Smart Stack workers...")
        activeSearchProcess?.terminate()
        activeSearchProcess = nil
        isBusy = true

        let script = #"""
        ROOT="\#(stackRoot)"
        UID="$(id -u)"
        PID_FILE="/tmp/smart_stack_text_embed_${UID}.sock.pid"
        SOCKET_FILE="/tmp/smart_stack_text_embed_${UID}.sock"

        kill_tree() {
          local node="$1"
          [ -z "$node" ] && return
          if ! kill -0 "$node" 2>/dev/null; then
            return
          fi
          local children
          children="$(pgrep -P "$node" 2>/dev/null || true)"
          if [ -n "$children" ]; then
            printf '%s\n' "$children" | while IFS= read -r child; do
              [ -z "$child" ] && continue
              kill_tree "$child"
            done
          fi
          kill -TERM "$node" 2>/dev/null || true
          sleep 0.12
          kill -0 "$node" 2>/dev/null && kill -KILL "$node" 2>/dev/null || true
        }

        collect_pids() {
          {
            if [ -f "$PID_FILE" ]; then
              PID_FROM_FILE="$(tr -dc '0-9' < "$PID_FILE" 2>/dev/null || true)"
              if [ -n "$PID_FROM_FILE" ] && kill -0 "$PID_FROM_FILE" 2>/dev/null; then
                echo "$PID_FROM_FILE"
              fi
            fi
            pgrep -f "$ROOT/\.venv/bin/python" 2>/dev/null || true
            pgrep -f "$ROOT/(mm_cli\.py|search\.py|ingest\.py|openclaw_imgsearch\.py|notes_index\.py|run_guarded_ingest\.sh)" 2>/dev/null || true
            pgrep -f "mm_stack\.text_embed_daemon|smart_stack_text_embed_${UID}\.sock" 2>/dev/null || true
          } | awk -v self="$$" -v parent="$PPID" 'NF>0 && $1 != self && $1 != parent {print $1}' | sort -u
        }

        # Kill daemon tree first from PID file (captures resource_tracker children).
        if [ -f "$PID_FILE" ]; then
          DAEMON_PID="$(tr -dc '0-9' < "$PID_FILE" 2>/dev/null || true)"
          if [ -n "$DAEMON_PID" ]; then
            kill_tree "$DAEMON_PID"
          fi
        fi

        PID_LIST="$(collect_pids)"
        if [ -n "$PID_LIST" ]; then
          echo "$PID_LIST" | xargs kill -TERM 2>/dev/null || true
          sleep 0.45
          printf '%s\n' "$PID_LIST" | while IFS= read -r pid; do
            [ -z "$pid" ] && continue
            kill_tree "$pid"
          done
        fi

        rm -f "$PID_FILE" "$SOCKET_FILE" 2>/dev/null || true
        KILLED="$(printf '%s\n' "$PID_LIST" | paste -sd ' ' -)"
        REMAINING="$(collect_pids | paste -sd ' ' -)"
        echo "killed=$KILLED"
        echo "remaining=$REMAINING"
        """#

        DispatchQueue.global(qos: .userInitiated).async {
            let process = Process()
            process.currentDirectoryURL = URL(fileURLWithPath: stackRoot)
            process.executableURL = URL(fileURLWithPath: "/bin/zsh")
            process.arguments = ["-lc", script]

            let outPipe = Pipe()
            let errPipe = Pipe()
            process.standardOutput = outPipe
            process.standardError = errPipe

            do {
                try process.run()
                process.waitUntilExit()
                let outData = outPipe.fileHandleForReading.readDataToEndOfFile()
                let errData = errPipe.fileHandleForReading.readDataToEndOfFile()
                let outText = String(data: outData, encoding: .utf8)?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
                let errText = String(data: errData, encoding: .utf8)?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""

                DispatchQueue.main.async {
                    self.isBusy = false
                    if !errText.isEmpty {
                        self.appendLog("[Emergency Kill] stderr: \(errText)")
                    }
                    if outText.isEmpty {
                        self.appendLog("[Emergency Kill] No running Smart Stack worker process found.")
                    } else {
                        self.appendLog("[Emergency Kill] Terminated PIDs: \(outText)")
                    }
                }
            } catch {
                DispatchQueue.main.async {
                    self.isBusy = false
                    self.appendLog("[Emergency Kill] Failed: \(error.localizedDescription)")
                }
            }
        }
    }
}

// MARK: - Data Models for Settings

struct WatchedFolder: Identifiable, Decodable {
    let id: Int
    let path: String
    let enabled: Bool
    let added_at: String
}

struct ExclusionPattern: Identifiable, Decodable {
    let id: Int
    let pattern: String
    let added_at: String
}

// MARK: - Settings Sheet

struct SettingsSheet: View {
    @ObservedObject var vm: SmartStackViewModel
    @State private var newExclusion: String = ""
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        VStack(spacing: 0) {
            // Header
            HStack {
                Text("Settings")
                    .font(.system(size: 18, weight: .bold, design: .rounded))
                Spacer()
                Button { dismiss() } label: {
                    Image(systemName: "xmark.circle.fill")
                        .font(.title2)
                        .foregroundStyle(.secondary)
                }
                .buttonStyle(.plain)
            }
            .padding(20)

            Divider()

            ScrollView {
                VStack(alignment: .leading, spacing: 20) {
                    // Watched Folders
                    VStack(alignment: .leading, spacing: 10) {
                        HStack {
                            Label("Watched Folders", systemImage: "folder.badge.gearshape")
                                .font(.system(size: 15, weight: .semibold, design: .rounded))
                            Spacer()
                            Button { vm.addWatchedFolder() } label: {
                                Image(systemName: "plus.circle.fill")
                                    .foregroundStyle(.blue)
                            }
                            .buttonStyle(.plain)
                            .help("Add folder to watch")
                        }

                        if vm.watchedFolders.isEmpty {
                            Text("No watched folders. Add one to start.")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                                .padding(.vertical, 8)
                        } else {
                            ForEach(vm.watchedFolders) { folder in
                                HStack(spacing: 8) {
                                    Button {
                                        vm.toggleWatchedFolder(folder.path)
                                    } label: {
                                        Image(systemName: folder.enabled ? "checkmark.circle.fill" : "circle")
                                            .foregroundStyle(folder.enabled ? .green : .secondary)
                                    }
                                    .buttonStyle(.plain)

                                    VStack(alignment: .leading) {
                                        Text(URL(fileURLWithPath: folder.path).lastPathComponent)
                                            .font(.system(size: 13, weight: .medium))
                                        Text(folder.path)
                                            .font(.system(size: 11))
                                            .foregroundStyle(.secondary)
                                            .lineLimit(1)
                                            .truncationMode(.middle)
                                    }

                                    Spacer()

                                    Button {
                                        vm.removeWatchedFolder(folder.path)
                                    } label: {
                                        Image(systemName: "trash")
                                            .foregroundStyle(.red.opacity(0.7))
                                    }
                                    .buttonStyle(.plain)
                                }
                                .padding(8)
                                .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 8))
                            }
                        }
                    }

                    Divider()

                    // Exclusions
                    VStack(alignment: .leading, spacing: 10) {
                        Label("Excluded Patterns", systemImage: "eye.slash")
                            .font(.system(size: 15, weight: .semibold, design: .rounded))

                        HStack {
                            TextField("e.g. *.tmp or /path/to/skip", text: $newExclusion)
                                .textFieldStyle(.plain)
                                .font(.system(size: 13))
                                .padding(8)
                                .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 8))

                            Button {
                                let pat = newExclusion.trimmingCharacters(in: .whitespacesAndNewlines)
                                guard !pat.isEmpty else { return }
                                vm.addExclusion(pat)
                                newExclusion = ""
                            } label: {
                                Image(systemName: "plus.circle.fill")
                                    .foregroundStyle(.blue)
                            }
                            .buttonStyle(.plain)
                        }

                        if vm.exclusions.isEmpty {
                            Text("No exclusions.")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        } else {
                            ForEach(vm.exclusions) { excl in
                                HStack {
                                    Text(excl.pattern)
                                        .font(.system(size: 13, design: .monospaced))
                                    Spacer()
                                    Button {
                                        vm.removeExclusion(excl.pattern)
                                    } label: {
                                        Image(systemName: "trash")
                                            .foregroundStyle(.red.opacity(0.7))
                                    }
                                    .buttonStyle(.plain)
                                }
                                .padding(6)
                                .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 6))
                            }
                        }
                    }

                    Divider()

                    // Actions
                    VStack(spacing: 10) {
                        Button {
                            vm.runRescanAll()
                        } label: {
                            HStack {
                                Image(systemName: "arrow.clockwise")
                                Text("Rescan Now")
                            }
                            .frame(maxWidth: .infinity)
                            .padding(10)
                            .background(.blue.opacity(0.2), in: RoundedRectangle(cornerRadius: 10))
                        }
                        .buttonStyle(.plain)
                        .disabled(vm.isBusy)
                    }
                }
                .padding(20)
            }
        }
        .frame(width: 450, height: 550)
        .background(.regularMaterial)
        .onAppear {
            vm.loadWatchedFolders()
            vm.loadExclusions()
        }
    }
}

struct SemanticTimelineSheet: View {
    @ObservedObject var vm: SmartStackViewModel
    @Environment(\.dismiss) private var dismiss
    @State private var selectedBucket: TimelineBucket?
    @State private var statusHint: String = "Pinch in/out to switch Year ↔ Month ↔ Day."

    private var granularityBinding: Binding<TimelineGranularity> {
        Binding(
            get: { vm.timelineGranularity },
            set: { vm.setTimelineGranularity($0) }
        )
    }

    private func zoomIn() {
        if let next = vm.timelineGranularity.zoomIn {
            vm.setTimelineGranularity(next)
        }
    }

    private func zoomOut() {
        if let prev = vm.timelineGranularity.zoomOut {
            vm.setTimelineGranularity(prev)
        }
    }

    var body: some View {
        VStack(spacing: 0) {
            HStack {
                VStack(alignment: .leading, spacing: 3) {
                    Text("Semantic Timeline")
                        .font(.system(size: 18, weight: .bold, design: .rounded))
                    Text(vm.query.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? "All indexed items" : "Filtered by current query")
                        .font(.system(size: 12, weight: .medium, design: .rounded))
                        .foregroundStyle(.secondary)
                }
                Spacer()
                Button {
                    dismiss()
                } label: {
                    Image(systemName: "xmark.circle.fill")
                        .font(.title2)
                        .foregroundStyle(.secondary)
                }
                .buttonStyle(.plain)
            }
            .padding(18)

            Divider()

            if vm.timelineLoading {
                VStack(spacing: 12) {
                    ProgressView()
                    Text("Building timeline buckets...")
                        .font(.system(size: 12, weight: .medium))
                        .foregroundStyle(.secondary)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else if !vm.timelineError.isEmpty {
                VStack(spacing: 12) {
                    Image(systemName: "exclamationmark.triangle.fill")
                        .font(.title)
                        .foregroundStyle(.orange)
                    Text(vm.timelineError)
                        .font(.system(size: 13, weight: .medium))
                        .multilineTextAlignment(.center)
                }
                .padding(20)
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else if let data = vm.timelineData {
                VStack(alignment: .leading, spacing: 14) {
                    HStack(spacing: 12) {
                        Picker("Granularity", selection: granularityBinding) {
                            ForEach(TimelineGranularity.allCases) { g in
                                Text(g.title).tag(g)
                            }
                        }
                        .pickerStyle(.segmented)
                        .frame(width: 230)

                        Button {
                            zoomOut()
                        } label: {
                            Image(systemName: "minus.magnifyingglass")
                        }
                        .buttonStyle(.bordered)
                        .disabled(vm.timelineGranularity.zoomOut == nil)

                        Button {
                            zoomIn()
                        } label: {
                            Image(systemName: "plus.magnifyingglass")
                        }
                        .buttonStyle(.bordered)
                        .disabled(vm.timelineGranularity.zoomIn == nil)

                        Spacer()

                        Text("Items: \(data.total_items)  Buckets: \(data.bucket_count)")
                            .font(.system(size: 12, weight: .semibold, design: .monospaced))
                            .foregroundStyle(.secondary)
                    }

                    TimelineBarPlot(
                        buckets: data.buckets,
                        maxCount: max(1, data.stats.max_count),
                        selectedBucket: $selectedBucket,
                        statusHint: $statusHint
                    )
                    .frame(height: 340)
                    .gesture(
                        MagnificationGesture()
                            .onEnded { value in
                                if value > 1.08 {
                                    zoomIn()
                                } else if value < 0.92 {
                                    zoomOut()
                                }
                            }
                    )

                    Text(statusHint)
                        .font(.system(size: 12, weight: .medium))
                        .foregroundStyle(.secondary)

                    if let selectedBucket {
                        HStack(spacing: 10) {
                            VStack(alignment: .leading, spacing: 3) {
                                Text(selectedBucket.key)
                                    .font(.system(size: 14, weight: .bold, design: .rounded))
                                Text("Count: \(selectedBucket.item_count)")
                                    .font(.system(size: 12, weight: .semibold, design: .monospaced))
                                    .foregroundStyle(.secondary)
                            }
                            Spacer()
                            if !selectedBucket.sample_path.isEmpty {
                                Button("Open Sample") {
                                    NSWorkspace.shared.open(URL(fileURLWithPath: selectedBucket.sample_path))
                                }
                                .buttonStyle(.borderedProminent)
                            }
                        }
                        .padding(10)
                        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 10))
                    } else {
                        Text("Select a bucket for details.")
                            .font(.system(size: 12, weight: .medium))
                            .foregroundStyle(.secondary)
                    }
                }
                .padding(18)
            } else {
                Text("No timeline data.")
                    .font(.system(size: 13, weight: .medium))
                    .foregroundStyle(.secondary)
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            }
        }
        .frame(width: 980, height: 680)
        .background(.regularMaterial)
        .onAppear {
            if vm.timelineData == nil && !vm.timelineLoading {
                vm.runTimeline(granularity: vm.timelineGranularity)
            }
        }
    }
}

struct MemoryClustersSheet: View {
    @ObservedObject var vm: SmartStackViewModel
    @Environment(\.dismiss) private var dismiss

    private var selectedCluster: MemoryCluster? {
        vm.clusters.first(where: { $0.cluster_id == vm.selectedClusterID }) ?? vm.clusters.first
    }

    private var selectedClusterResults: [SearchResult] {
        vm.clusterItems.map { item in
            let url = URL(fileURLWithPath: item.file_path)
            return SearchResult(
                image_id: item.image_id,
                source: "image",
                filename: url.lastPathComponent,
                caption: item.caption,
                tags: item.tags,
                score: String(format: "%.4f", max(0.0, 1.0 - item.distance)),
                obsidian_path: item.file_path
            )
        }
    }

    private func select(_ cluster: MemoryCluster) {
        vm.selectedClusterID = cluster.cluster_id
        vm.loadClusterItems(clusterID: cluster.cluster_id)
    }

    var body: some View {
        VStack(spacing: 0) {
            HStack {
                VStack(alignment: .leading, spacing: 3) {
                    Text("Photo Clusters")
                        .font(.system(size: 18, weight: .bold, design: .rounded))
                    Text("Auto-grouped by visual similarity")
                        .font(.system(size: 12, weight: .medium, design: .rounded))
                        .foregroundStyle(.secondary)
                }
                Spacer()
                Button("Auto Cluster") {
                    vm.runAutoCluster()
                }
                .buttonStyle(.borderedProminent)
                .disabled(vm.clustersLoading)
                Button("Refresh") {
                    vm.loadClusters()
                }
                .buttonStyle(.bordered)
                .disabled(vm.clustersLoading)
                Button {
                    dismiss()
                } label: {
                    Image(systemName: "xmark.circle.fill")
                        .font(.title2)
                        .foregroundStyle(.secondary)
                }
                .buttonStyle(.plain)
            }
            .padding(18)

            Divider()

            if vm.clustersLoading && vm.clusters.isEmpty {
                VStack(spacing: 12) {
                    ProgressView()
                    Text("Building memory clusters...")
                        .font(.system(size: 12, weight: .medium))
                        .foregroundStyle(.secondary)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else if !vm.clustersError.isEmpty && vm.clusters.isEmpty {
                VStack(spacing: 12) {
                    Image(systemName: "exclamationmark.triangle.fill")
                        .font(.title)
                        .foregroundStyle(.orange)
                    Text(vm.clustersError)
                        .font(.system(size: 13, weight: .medium))
                        .multilineTextAlignment(.center)
                }
                .padding(20)
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else if vm.clusters.isEmpty {
                VStack(spacing: 12) {
                    Image(systemName: "square.grid.3x3")
                        .font(.system(size: 34, weight: .regular))
                        .foregroundStyle(.secondary.opacity(0.7))
                    Text("No clusters yet.")
                        .font(.system(size: 13, weight: .semibold, design: .rounded))
                    Button("Run Auto Cluster") {
                        vm.runAutoCluster()
                    }
                    .buttonStyle(.borderedProminent)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
                HStack(spacing: 0) {
                    ScrollView {
                        VStack(spacing: 8) {
                            ForEach(vm.clusters) { cluster in
                                let isSelected = cluster.cluster_id == (selectedCluster?.cluster_id ?? "")
                                Button {
                                    select(cluster)
                                } label: {
                                    HStack(spacing: 10) {
                                        if let sample = cluster.sample_item {
                                            AsyncImage(url: URL(fileURLWithPath: sample.file_path)) { image in
                                                image.resizable().aspectRatio(contentMode: .fill)
                                            } placeholder: {
                                                Color.white.opacity(0.08)
                                            }
                                            .frame(width: 42, height: 42)
                                            .clipShape(RoundedRectangle(cornerRadius: 7))
                                        } else {
                                            RoundedRectangle(cornerRadius: 7)
                                                .fill(Color.white.opacity(0.08))
                                                .frame(width: 42, height: 42)
                                                .overlay(
                                                    Image(systemName: "photo")
                                                        .foregroundStyle(.secondary)
                                                )
                                        }
                                        VStack(alignment: .leading, spacing: 2) {
                                            Text(cluster.displayTitle)
                                                .font(.system(size: 13, weight: .semibold, design: .rounded))
                                                .lineLimit(1)
                                            Text("\(cluster.item_count) photo(s)")
                                                .font(.system(size: 11, weight: .medium, design: .monospaced))
                                                .foregroundStyle(.secondary)
                                        }
                                        Spacer()
                                    }
                                    .padding(10)
                                    .background(
                                        RoundedRectangle(cornerRadius: 10)
                                            .fill(isSelected ? Color.blue.opacity(0.22) : Color.white.opacity(0.06))
                                    )
                                }
                                .buttonStyle(.plain)
                            }
                        }
                        .padding(12)
                    }
                    .frame(width: 280)

                    Divider()

                    VStack(alignment: .leading, spacing: 10) {
                        if let selected = selectedCluster {
                            HStack {
                                VStack(alignment: .leading, spacing: 3) {
                                    Text(selected.displayTitle)
                                        .font(.system(size: 16, weight: .bold, design: .rounded))
                                    Text("\(selected.item_count) photo(s)")
                                        .font(.system(size: 12, weight: .semibold, design: .monospaced))
                                        .foregroundStyle(.secondary)
                                }
                                Spacer()
                                if vm.clustersLoading {
                                    ProgressView()
                                        .controlSize(.small)
                                }
                            }

                            if !vm.clustersError.isEmpty {
                                Text(vm.clustersError)
                                    .font(.system(size: 12, weight: .medium))
                                    .foregroundStyle(.orange)
                            }

                            if selectedClusterResults.isEmpty && !vm.clustersLoading {
                                Text("No photos in this cluster.")
                                    .font(.system(size: 12, weight: .medium))
                                    .foregroundStyle(.secondary)
                                    .padding(.top, 18)
                            } else {
                                ScrollView {
                                    ContentView.MasonryGrid(items: selectedClusterResults, columns: 3) { item in
                                        ResultCard(
                                            result: item,
                                            openAction: { vm.open(item) },
                                            contextAction: item.image_id == nil ? nil : { vm.runContextLens(for: item) },
                                            attachChatAction: { vm.attachImageForChat(item) }
                                        )
                                        .padding(.bottom, 12)
                                    }
                                    .padding(.top, 6)
                                }
                            }
                        } else {
                            Text("Select a cluster.")
                                .font(.system(size: 12, weight: .medium))
                                .foregroundStyle(.secondary)
                        }
                    }
                    .padding(14)
                }
            }
        }
        .frame(width: 1140, height: 760)
        .background(.regularMaterial)
        .onAppear {
            if vm.clusters.isEmpty && !vm.clustersLoading {
                vm.loadClusters()
            } else if vm.clusterItems.isEmpty, let selected = selectedCluster {
                vm.loadClusterItems(clusterID: selected.cluster_id)
            }
        }
    }
}

struct TimelineBarPlot: View {
    let buckets: [TimelineBucket]
    let maxCount: Int
    @Binding var selectedBucket: TimelineBucket?
    @Binding var statusHint: String

    private func shortLabel(_ key: String) -> String {
        if key.count <= 7 { return key }
        return String(key.suffix(5))
    }

    var body: some View {
        GeometryReader { geo in
            let count = max(1, buckets.count)
            let labelStride = max(1, count / 14)
            let plotHeight = max(80.0, geo.size.height - 30.0)

            ScrollView(.horizontal) {
                HStack(alignment: .bottom, spacing: 6) {
                    ForEach(Array(buckets.enumerated()), id: \.element.id) { idx, bucket in
                        let ratio = CGFloat(bucket.item_count) / CGFloat(maxCount)
                        let h = max(8, ratio * plotHeight)
                        let isSelected = selectedBucket?.id == bucket.id

                        VStack(spacing: 4) {
                            RoundedRectangle(cornerRadius: 3)
                                .fill(isSelected ? Color.blue : Color.blue.opacity(0.6))
                                .frame(width: 16, height: h)
                                .overlay(
                                    RoundedRectangle(cornerRadius: 3)
                                        .stroke(Color.white.opacity(isSelected ? 0.35 : 0.0), lineWidth: 0.8)
                                )
                            Text(idx % labelStride == 0 ? shortLabel(bucket.key) : "")
                                .font(.system(size: 9, weight: .medium, design: .monospaced))
                                .foregroundStyle(.secondary)
                                .frame(height: 11)
                        }
                        .contentShape(Rectangle())
                        .onTapGesture {
                            selectedBucket = bucket
                            statusHint = "\(bucket.key): \(bucket.item_count) item(s)"
                        }
                    }
                }
                .frame(height: geo.size.height, alignment: .bottom)
                .padding(.horizontal, 8)
            }
        }
        .background(.thinMaterial, in: RoundedRectangle(cornerRadius: 12))
    }
}

struct ContextRing: Identifiable {
    let id: String
    let title: String
    let color: Color
    let hint: String
    let neighbors: [ContextLensNeighbor]
}

struct ContextLensSheet: View {
    @ObservedObject var vm: SmartStackViewModel
    @Environment(\.dismiss) private var dismiss
    @State private var hoveredHint: String = "Hover a node to see why it appears."
    @State private var selectedNeighbor: ContextLensNeighbor?

    private var rings: [ContextRing] {
        guard let payload = vm.contextLens else { return [] }
        return [
            ContextRing(
                id: "similarity",
                title: "Similarity",
                color: .mint,
                hint: "CLIP nearest neighbors by visual similarity.",
                neighbors: payload.rings.similarity
            ),
            ContextRing(
                id: "cluster",
                title: "Cluster",
                color: .indigo,
                hint: "Items from the same memory cluster.",
                neighbors: payload.rings.cluster
            ),
            ContextRing(
                id: "entity",
                title: "Entity",
                color: .orange,
                hint: "Shared tag/caption terms.",
                neighbors: payload.rings.entity
            ),
            ContextRing(
                id: "time",
                title: "Time",
                color: .pink,
                hint: "Nearest neighbors by ingestion timestamp.",
                neighbors: payload.rings.time
            ),
        ]
    }

    var body: some View {
        VStack(spacing: 0) {
            HStack {
                VStack(alignment: .leading, spacing: 3) {
                    Text("Context Lens")
                        .font(.system(size: 18, weight: .bold, design: .rounded))
                    if let payload = vm.contextLens {
                        Text(URL(fileURLWithPath: payload.target.file_path).lastPathComponent)
                            .font(.system(size: 12, weight: .medium, design: .monospaced))
                            .foregroundStyle(.secondary)
                            .lineLimit(1)
                    }
                }
                Spacer()
                Button {
                    dismiss()
                } label: {
                    Image(systemName: "xmark.circle.fill")
                        .font(.title2)
                        .foregroundStyle(.secondary)
                }
                .buttonStyle(.plain)
            }
            .padding(18)

            Divider()

            if vm.contextLensLoading {
                VStack(spacing: 12) {
                    ProgressView()
                    Text("Building relation graph...")
                        .font(.system(size: 12, weight: .medium))
                        .foregroundStyle(.secondary)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else if !vm.contextLensError.isEmpty {
                VStack(spacing: 12) {
                    Image(systemName: "exclamationmark.triangle.fill")
                        .font(.title)
                        .foregroundStyle(.orange)
                    Text(vm.contextLensError)
                        .font(.system(size: 13, weight: .medium))
                        .multilineTextAlignment(.center)
                    Button("Close") { dismiss() }
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .padding(20)
            } else if let payload = vm.contextLens {
                VStack(spacing: 10) {
                    ContextLensRingPlot(
                        payload: payload,
                        rings: rings,
                        hoveredHint: $hoveredHint,
                        selectedNeighbor: $selectedNeighbor
                    )
                    .frame(height: 430)

                    Text(hoveredHint)
                        .font(.system(size: 12, weight: .medium, design: .rounded))
                        .foregroundStyle(.secondary)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding(.horizontal, 14)

                    if let selectedNeighbor {
                        HStack(spacing: 10) {
                            AsyncImage(url: URL(fileURLWithPath: selectedNeighbor.file_path)) { image in
                                image.resizable().aspectRatio(contentMode: .fill)
                            } placeholder: {
                                Color.white.opacity(0.08)
                            }
                            .frame(width: 58, height: 58)
                            .clipShape(RoundedRectangle(cornerRadius: 8))

                            VStack(alignment: .leading, spacing: 3) {
                                Text(URL(fileURLWithPath: selectedNeighbor.file_path).lastPathComponent)
                                    .font(.system(size: 12, weight: .semibold, design: .rounded))
                                    .lineLimit(1)
                                if !selectedNeighbor.caption.isEmpty {
                                    Text(selectedNeighbor.caption)
                                        .font(.system(size: 11))
                                        .foregroundStyle(.secondary)
                                        .lineLimit(2)
                                }
                            }
                            Spacer()
                            Button("Open") {
                                NSWorkspace.shared.open(URL(fileURLWithPath: selectedNeighbor.file_path))
                            }
                            .buttonStyle(.bordered)
                        }
                        .padding(.horizontal, 14)
                        .padding(.bottom, 14)
                    } else {
                        Spacer(minLength: 10)
                    }
                }
                .padding(.top, 10)
            } else {
                Text("No context data available.")
                    .font(.system(size: 13, weight: .medium))
                    .foregroundStyle(.secondary)
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            }
        }
        .frame(width: 860, height: 700)
        .background(.regularMaterial)
    }
}

struct ContextLensRingPlot: View {
    let payload: ContextLensResponse
    let rings: [ContextRing]
    @Binding var hoveredHint: String
    @Binding var selectedNeighbor: ContextLensNeighbor?

    private func point(
        center: CGPoint,
        radius: CGFloat,
        index: Int,
        count: Int
    ) -> CGPoint {
        let total = max(1, count)
        let angle = (Double(index) / Double(total)) * (2.0 * Double.pi) - (Double.pi / 2.0)
        return CGPoint(
            x: center.x + CGFloat(cos(angle)) * radius,
            y: center.y + CGFloat(sin(angle)) * radius
        )
    }

    private func ringNodeLabel(for neighbor: ContextLensNeighbor) -> String {
        let name = URL(fileURLWithPath: neighbor.file_path).lastPathComponent
        let first = name.first.map(String.init) ?? "•"
        return first.uppercased()
    }

    var body: some View {
        GeometryReader { geo in
            let size = geo.size
            let center = CGPoint(x: size.width / 2.0, y: size.height / 2.0)
            let coreRadius = min(size.width, size.height) * 0.12
            let ringGap = min(size.width, size.height) * 0.09

            ZStack {
                ForEach(Array(rings.enumerated()), id: \.element.id) { idx, ring in
                    let radius = coreRadius + CGFloat(idx + 1) * ringGap
                    Circle()
                        .stroke(ring.color.opacity(0.35), lineWidth: 1.4)
                        .frame(width: radius * 2, height: radius * 2)

                    Text(ring.title)
                        .font(.system(size: 11, weight: .bold, design: .rounded))
                        .foregroundStyle(ring.color.opacity(0.95))
                        .position(x: center.x, y: center.y - radius - 10)

                    ForEach(Array(ring.neighbors.enumerated()), id: \.element.id) { nIdx, neighbor in
                        let pos = point(center: center, radius: radius, index: nIdx, count: ring.neighbors.count)
                        Button {
                            selectedNeighbor = neighbor
                            hoveredHint = ring.hint
                        } label: {
                            Text(ringNodeLabel(for: neighbor))
                                .font(.system(size: 10, weight: .bold, design: .rounded))
                                .foregroundStyle(.white)
                                .frame(width: 24, height: 24)
                                .background(ring.color.opacity(0.9), in: Circle())
                                .overlay(Circle().stroke(.white.opacity(0.25), lineWidth: 0.8))
                                .shadow(color: .black.opacity(0.25), radius: 2, x: 0, y: 1)
                        }
                        .buttonStyle(.plain)
                        .position(x: pos.x, y: pos.y)
                        .help("\(ring.title): \(URL(fileURLWithPath: neighbor.file_path).lastPathComponent)")
                        .onHover { inside in
                            if inside {
                                hoveredHint = ring.hint
                                selectedNeighbor = neighbor
                            }
                        }
                    }
                }

                VStack(spacing: 8) {
                    AsyncImage(url: URL(fileURLWithPath: payload.target.file_path)) { image in
                        image.resizable().aspectRatio(contentMode: .fill)
                    } placeholder: {
                        Color.white.opacity(0.08)
                    }
                    .frame(width: 92, height: 92)
                    .clipShape(RoundedRectangle(cornerRadius: 14))

                    Text("Target")
                        .font(.system(size: 11, weight: .bold, design: .rounded))
                        .foregroundStyle(.secondary)
                }
                .padding(8)
                .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 16))
            }
            .frame(width: size.width, height: size.height)
        }
    }
}

struct ResultCard: View {
    let result: SearchResult
    let openAction: () -> Void
    let contextAction: (() -> Void)?
    let attachChatAction: (() -> Void)?
    @State private var isHovering = false

    init(
        result: SearchResult,
        openAction: @escaping () -> Void,
        contextAction: (() -> Void)? = nil,
        attachChatAction: (() -> Void)? = nil
    ) {
        self.result = result
        self.openAction = openAction
        self.contextAction = contextAction
        self.attachChatAction = attachChatAction
    }

    private var badgeColor: Color {
        result.source == "note" ? .orange : .mint
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
             // Image / Icon Placeholder (Visual First)
            if result.source == "image" {
                // Async load from local file URL
                AsyncImage(url: URL(fileURLWithPath: result.obsidian_path)) { phase in
                    if let image = phase.image {
                        image
                            .resizable()
                            .aspectRatio(contentMode: .fill)
                            .frame(minHeight: 120, maxHeight: 300)
                            .clipped()
                    } else if phase.error != nil {
                         ZStack {
                             Rectangle().fill(Color.gray.opacity(0.2))
                             Image(systemName: "photo.badge.exclamationmark")
                                .font(.title)
                                .foregroundStyle(.secondary)
                         }
                         .aspectRatio(1.5, contentMode: .fit)
                    } else {
                        // Placeholder / Loading
                         ZStack {
                             Rectangle().fill(Color.black.opacity(0.1))
                             ProgressView()
                         }
                         .aspectRatio(1.5, contentMode: .fit)
                    }
                }
                .cornerRadius(12)
            } else {
                 ZStack(alignment: .topLeading) {
                     Rectangle()
                        .fill(Color.yellow.opacity(0.1))
                        .aspectRatio(1.2, contentMode: .fit)
                     
                     Image(systemName: "note.text")
                        .font(.title2)
                        .foregroundStyle(.orange.opacity(0.8))
                        .padding(12)
                        
                    Text(result.filename)
                        .font(.system(size: 14, weight: .bold, design: .serif))
                        .foregroundStyle(.primary)
                        .padding(.top, 44)
                        .padding(.horizontal, 12)
                        .lineLimit(4)
                }
                .cornerRadius(12)
            }

            VStack(alignment: .leading, spacing: 4) {
                 if result.source == "image" {
                     Text(result.filename)
                        .font(.system(size: 14, weight: .bold, design: .rounded))
                        .foregroundStyle(.primary)
                        .lineLimit(2)
                 }
                
                if !result.caption.isEmpty {
                    Text(result.caption)
                        .font(.system(size: 12, weight: .regular, design: .rounded))
                        .foregroundStyle(.secondary)
                        .lineLimit(3)
                }
                
                if !result.tags.isEmpty {
                     Text(result.tags.joined(separator: ", "))
                        .font(.system(size: 11, weight: .medium, design: .rounded))
                        .foregroundStyle(.tertiary)
                        .lineLimit(1)
                }
            }
            .padding(.horizontal, 8)
            .padding(.bottom, 12)

            if (contextAction != nil || attachChatAction != nil), result.source == "image" {
                HStack {
                    Spacer()
                    if let attachChatAction {
                        Button {
                            attachChatAction()
                        } label: {
                            Label("Chat", systemImage: "paperclip")
                                .font(.system(size: 11, weight: .semibold, design: .rounded))
                                .padding(.horizontal, 8)
                                .padding(.vertical, 5)
                                .background(.ultraThinMaterial, in: Capsule())
                        }
                        .buttonStyle(.plain)
                    }
                    if let contextAction {
                        Button {
                            contextAction()
                        } label: {
                            Label("Lens", systemImage: "scope")
                                .font(.system(size: 11, weight: .semibold, design: .rounded))
                                .padding(.horizontal, 8)
                                .padding(.vertical, 5)
                                .background(.ultraThinMaterial, in: Capsule())
                        }
                        .buttonStyle(.plain)
                    }
                }
                .padding(.horizontal, 8)
                .padding(.bottom, 10)
            }
            
            // Hover Overlay
             if isHovering {
                HStack {
                    Spacer()
                    Image(systemName: "arrow.up.right.square.fill")
                        .font(.system(size: 24))
                        .foregroundStyle(.white)
                        .shadow(radius: 4)
                        .padding(12)
                }
                .background(
                    LinearGradient(colors: [.black.opacity(0.6), .clear], startPoint: .bottom, endPoint: .center)
                )
                .cornerRadius(16)
            }
        }
        .background(.thinMaterial)
        .cornerRadius(16)
        .shadow(color: .black.opacity(0.1), radius: 6, x: 0, y: 3)
        .contentShape(Rectangle()) // Make entire area clickable
        .onTapGesture {
            openAction()
        }
        .contextMenu {
            if let attachChatAction, result.source == "image" {
                Button("Attach to Chat") {
                    attachChatAction()
                }
            }
            if let contextAction, result.source == "image" {
                Button("Open Context Lens") {
                    contextAction()
                }
            }
        }
        .onHover { hover in
            withAnimation(.easeInOut(duration: 0.2)) {
                isHovering = hover
                if hover {
                    NSCursor.pointingHand.push()
                } else {
                    NSCursor.pop()
                }
            }
        }
    }
}

struct ContentView: View {
    @ObservedObject var vm: SmartStackViewModel
    @FocusState private var queryFocused: Bool
    @State private var isExpandedMode: Bool = false
    @State private var isVisualDropTarget: Bool = false

    var body: some View {
        ZStack {
            VisualEffect(material: .sidebar, blendingMode: .behindWindow)
                .ignoresSafeArea()

            WindowAccessor { window in
                window.isOpaque = false
                window.backgroundColor = .clear
                window.titlebarAppearsTransparent = true
                window.styleMask.insert(.fullSizeContentView)
            }

            // Dark tint overlay for readability & mood
            Color.black.opacity(0.40)
                .ignoresSafeArea()



            VStack(spacing: 0) {
                // Minimal Header & Search
                header
                    .padding(.horizontal, 20)
                    .padding(.top, 20)
                    .padding(.bottom, 10)
                
                // Expanded Controls
                if isExpandedMode {
                    controls
                        .padding(.horizontal, 20)
                        .padding(.bottom, 10)
                        .transition(.move(edge: .top).combined(with: .opacity))
                }
                
                // Main Content
                if vm.isChatMode {
                    chatSection
                        .padding(.horizontal, 20)
                } else {
                    resultsSection
                        .padding(.horizontal, 20)
                }
                
                // Logs (Collapsible or bottom)
                logsSection
                    .padding(20)
            }
        }
        .sheet(isPresented: $vm.showSettings) {
            SettingsSheet(vm: vm)
        }
        .sheet(isPresented: $vm.showContextLens) {
            ContextLensSheet(vm: vm)
        }
        .sheet(isPresented: $vm.showTimeline) {
            SemanticTimelineSheet(vm: vm)
        }
        .sheet(isPresented: $vm.showClusters) {
            MemoryClustersSheet(vm: vm)
        }
    }

    private func topButtonLabel(_ text: String) -> some View {
        Text(text)
            .font(.system(size: 9, weight: .semibold, design: .rounded))
            .foregroundStyle(.secondary)
            .lineLimit(1)
    }

    private func labeledIconButton<LabelContent: View>(
        title: String,
        help: String,
        action: @escaping () -> Void,
        @ViewBuilder label: @escaping () -> LabelContent
    ) -> some View {
        VStack(spacing: 2) {
            topButtonLabel(title)
            Button(action: action, label: label)
                .buttonStyle(.plain)
                .help(help)
        }
    }

    private func labeledIconMenu<MenuContent: View, LabelContent: View>(
        title: String,
        help: String,
        @ViewBuilder content: @escaping () -> MenuContent,
        @ViewBuilder label: @escaping () -> LabelContent
    ) -> some View {
        VStack(spacing: 2) {
            topButtonLabel(title)
            Menu(content: content, label: label)
                .menuStyle(.button)
                .buttonStyle(.plain)
                .help(help)
        }
    }

    private var header: some View {
        HStack(spacing: 16) {
            // Search Bar + Visual Query Chip
            VStack(alignment: .leading, spacing: 6) {
                HStack {
                    Image(systemName: "magnifyingglass")
                        .foregroundStyle(.secondary)
                    TextField("Search...", text: $vm.query)
                        .textFieldStyle(.plain)
                        .font(.system(size: 16, weight: .medium, design: .rounded))
                        .foregroundStyle(.primary)
                        .focused($queryFocused)
                        .onSubmit {
                            if vm.isChatMode {
                                vm.runChat()
                            } else {
                                vm.runSearch()
                            }
                        }
                    if !vm.query.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                        Button {
                            vm.query = ""
                        } label: {
                            Image(systemName: "xmark.circle.fill")
                                .foregroundStyle(.secondary)
                        }
                        .buttonStyle(.plain)
                    }
                }

                if vm.hasVisualQueryImage {
                    HStack(spacing: 8) {
                        Image(systemName: "photo.fill.on.rectangle.fill")
                            .foregroundStyle(.blue.opacity(0.9))
                        Text(vm.visualQueryFilename)
                            .font(.system(size: 11, weight: .semibold, design: .monospaced))
                            .lineLimit(1)
                            .truncationMode(.middle)
                        Spacer()
                        Button {
                            vm.clearVisualQueryImage()
                        } label: {
                            Image(systemName: "xmark.circle.fill")
                                .foregroundStyle(.secondary)
                        }
                        .buttonStyle(.plain)
                    }
                }

                if vm.isChatMode, vm.hasAttachedChatImage {
                    HStack(spacing: 8) {
                        Image(systemName: "paperclip.circle.fill")
                            .foregroundStyle(.mint.opacity(0.95))
                        Text("Attached: \(vm.attachedChatImageFilename)")
                            .font(.system(size: 11, weight: .semibold, design: .monospaced))
                            .lineLimit(1)
                            .truncationMode(.middle)
                        Spacer()
                        Button {
                            vm.clearAttachedChatImage()
                        } label: {
                            Image(systemName: "xmark.circle.fill")
                                .foregroundStyle(.secondary)
                        }
                        .buttonStyle(.plain)
                    }
                }
            }
            .padding(12)
            .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 18))
            .overlay(
                RoundedRectangle(cornerRadius: 18)
                    .stroke(isVisualDropTarget ? Color.blue.opacity(0.65) : Color.white.opacity(0.08), lineWidth: 1.0)
            )
            .shadow(color: .black.opacity(0.1), radius: 4, x: 0, y: 2)
            .onPasteCommand(of: [.image, .fileURL]) { _ in
                vm.pasteClipboardImageForSearch()
            }
            .onDrop(of: ["public.file-url"], isTargeted: $isVisualDropTarget) { providers in
                for provider in providers {
                    provider.loadItem(forTypeIdentifier: "public.file-url", options: nil) { item, _ in
                        guard let data = item as? Data,
                              let url = URL(dataRepresentation: data, relativeTo: nil) else { return }
                        DispatchQueue.main.async {
                            vm.setVisualQueryImagePath(url.path)
                        }
                    }
                }
                return true
            }

            VStack(spacing: 2) {
                topButtonLabel("Mode")
                HStack(spacing: 0) {
                    Button { withAnimation(.snappy) { vm.isChatMode = false } } label: {
                        Image(systemName: "magnifyingglass")
                            .font(.system(size: 14, weight: .bold))
                            .foregroundStyle(vm.isChatMode ? Color.secondary : Color.white)
                            .frame(width: 40, height: 32)
                            .background(vm.isChatMode ? Color.clear : Color.white.opacity(0.25))
                            .clipShape(Capsule())
                    }
                    .buttonStyle(.plain)

                    Button { withAnimation(.snappy) { vm.isChatMode = true } } label: {
                        Image(systemName: "message.fill")
                            .font(.system(size: 13, weight: .bold))
                            .foregroundStyle(!vm.isChatMode ? Color.secondary : Color.white)
                            .frame(width: 40, height: 32)
                            .background(!vm.isChatMode ? Color.clear : Color.blue.opacity(0.6))
                            .clipShape(Capsule())
                    }
                    .buttonStyle(.plain)
                }
                .padding(2)
                .background(.ultraThinMaterial, in: Capsule())
                .overlay(
                    Capsule()
                        .stroke(Color.white.opacity(0.1), lineWidth: 0.5)
                )
                .frame(width: 80)
            }

            labeledIconButton(
                title: "Kill",
                help: "Emergency RAM Kill Switch",
                action: { vm.runEmergencyMemoryKillSwitch() }
            ) {
                Image(systemName: "power.circle.fill")
                    .font(.system(size: 16, weight: .bold))
                    .foregroundStyle(.white)
                    .padding(10)
                    .background(Color.red.opacity(0.88), in: Circle())
            }

            labeledIconButton(
                title: "Timeline",
                help: "Semantic Timeline",
                action: { vm.openTimeline() }
            ) {
                Image(systemName: "calendar.badge.clock")
                    .font(.system(size: 16, weight: .semibold))
                    .foregroundStyle(.secondary)
                    .padding(10)
                    .background(.ultraThinMaterial, in: Circle())
            }

            labeledIconButton(
                title: "Clusters",
                help: "Photo Clusters",
                action: { vm.openClusters() }
            ) {
                Image(systemName: "square.grid.3x3.fill")
                    .font(.system(size: 16, weight: .semibold))
                    .foregroundStyle(.secondary)
                    .padding(10)
                    .background(.ultraThinMaterial, in: Circle())
            }

            labeledIconButton(
                title: "All Photos",
                help: "All Indexed Photos",
                action: { vm.runAllPhotos() }
            ) {
                Image(systemName: "photo.stack")
                    .font(.system(size: 16, weight: .semibold))
                    .foregroundStyle(.secondary)
                    .padding(10)
                    .background(.ultraThinMaterial, in: Circle())
            }

            labeledIconButton(
                title: "Visual",
                help: "Pick visual query image",
                action: { vm.pickVisualQueryImage() }
            ) {
                Image(systemName: vm.hasVisualQueryImage ? "photo.badge.checkmark" : "photo.on.rectangle")
                    .font(.system(size: 16, weight: .semibold))
                    .foregroundStyle(vm.hasVisualQueryImage ? .blue : .secondary)
                    .padding(10)
                    .background(.ultraThinMaterial, in: Circle())
            }

            labeledIconMenu(
                title: "Paste",
                help: "Paste copied image",
                content: {
                    Button("Paste Image as Visual Query") { vm.pasteClipboardImageForSearch() }
                    Button("Paste Image and Ingest") { vm.pasteClipboardImageAndIngest() }
                },
                label: {
                    Image(systemName: "doc.on.clipboard")
                        .font(.system(size: 16, weight: .semibold))
                        .foregroundStyle(.secondary)
                        .padding(10)
                        .background(.ultraThinMaterial, in: Circle())
                }
            )

            labeledIconMenu(
                title: "Settings",
                help: "Search and ingest settings",
                content: {
                // Mode
                Picker("Mode", selection: $vm.searchMode) {
                    ForEach(SearchMode.allCases) { mode in
                        Text(mode.rawValue).tag(mode)
                    }
                }

                Divider()

                // Filters
                Picker("Filter", selection: $vm.sourceFilter) {
                    ForEach(SourceFilter.allCases) { f in
                        Text(f.rawValue).tag(f)
                    }
                }
                
                Divider()
                
                // Sliders
                Text("Top K: \(vm.topK)")
                Stepper("Top K", value: $vm.topK, in: 1...50)
                
                Text("Min Score: \(String(format: "%.2f", vm.minScore))")
                Slider(value: $vm.minScore, in: 0...1)

                Divider()

                // Actions
                Button("Pick Visual Query Image") { vm.pickVisualQueryImage() }
                Button("Paste Image as Visual Query") { vm.pasteClipboardImageForSearch() }
                Button("Paste Image and Ingest") { vm.pasteClipboardImageAndIngest() }
                Button("All Indexed Photos") { vm.runAllPhotos() }
                Button("Open Photo Clusters") { vm.openClusters() }
                Button("Auto Cluster Photos") { vm.runAutoCluster() }
                if vm.hasVisualQueryImage {
                    Button("Clear Visual Query Image") { vm.clearVisualQueryImage() }
                }
                if vm.hasAttachedChatImage {
                    Button("Clear Attached Chat Image") { vm.clearAttachedChatImage() }
                }
                Button("Clear Chat Conversation") { vm.clearChatConversation() }

                Divider()

                Button("Ingest File/Folder") { vm.runIngestPath() }
                Button("Ingest Inbox") { vm.runInboxIngest() }
                Button("Rescan Changed") { vm.runRescan() }
                Button("Safe Reprocess") { vm.runSafeReprocess() }
                Button(role: .destructive) { vm.runEmergencyMemoryKillSwitch() } label: { Text("Emergency Kill Switch") }

                Divider()

                Button("Settings...") { vm.showSettings = true }
                
                },
                label: {
                    Image(systemName: "slider.horizontal.3")
                        .font(.system(size: 18, weight: .semibold))
                        .foregroundStyle(.secondary)
                        .padding(10)
                        .background(.ultraThinMaterial, in: Circle())
                }
            )
            
            labeledIconButton(
                title: "Expand",
                help: "Toggle Expanded Controls",
                action: {
                    withAnimation(.spring(response: 0.4, dampingFraction: 0.7)) {
                        isExpandedMode.toggle()
                    }
                }
            ) {
                Image(systemName: isExpandedMode ? "chevron.up" : "chevron.down")
                    .font(.system(size: 14, weight: .bold))
                    .foregroundStyle(.secondary)
                    .padding(10)
                    .background(.ultraThinMaterial, in: Circle())
            }

            if vm.isBusy {
                ProgressView()
                    .controlSize(.small)
                    .padding(.leading, 8)
            }
        }
    }
    
    private var controls: some View {
        VStack(spacing: 12) {
            HStack(spacing: 12) {
                Text("MM-Only Search")
                    .font(.system(size: 12, weight: .semibold, design: .monospaced))
                    .padding(.horizontal, 10)
                    .padding(.vertical, 8)
                    .background(.thinMaterial, in: RoundedRectangle(cornerRadius: 8))

                Stepper("Top K: \(vm.topK)", value: $vm.topK, in: 1...50)
                    .frame(width: 130)
            }

            HStack(spacing: 14) {
                Picker("Filter", selection: $vm.sourceFilter) {
                    ForEach(SourceFilter.allCases) { f in
                        Text(f.rawValue).tag(f)
                    }
                }
                .pickerStyle(.segmented)
                .frame(width: 250)
                .labelsHidden()

                HStack {
                    Text("Min Score")
                    Slider(value: $vm.minScore, in: 0...1)
                    Text(String(format: "%.2f", vm.minScore))
                        .font(.system(size: 12, weight: .semibold, design: .monospaced))
                        .frame(width: 40)
                }

                Spacer()

                Button("Ingest File/Folder") { vm.runIngestPath() }
                    .buttonStyle(.bordered)
                Button("Ingest Inbox") { vm.runInboxIngest() }
                    .buttonStyle(.bordered)
                Button("Rescan Changed") { vm.runRescan() }
                    .buttonStyle(.bordered)
                Button("Safe Reprocess") { vm.runSafeReprocess() }
                    .buttonStyle(.bordered)
                Button("Emergency Kill") { vm.runEmergencyMemoryKillSwitch() }
                    .buttonStyle(.borderedProminent)
                    .tint(.red)
            }
        }
        .padding(14)
        .background(
            RoundedRectangle(cornerRadius: 18)
                .fill(.ultraThinMaterial)
                .shadow(color: .black.opacity(0.2), radius: 5, x: 0, y: 4)
        )
        .foregroundStyle(.white)
    }

    private var resultsSection: some View {
        ScrollView {
            MasonryGrid(items: vm.filteredResults, columns: 3) { item in
                ResultCard(
                    result: item,
                    openAction: { vm.open(item) },
                    contextAction: item.image_id == nil ? nil : { vm.runContextLens(for: item) },
                    attachChatAction: { vm.attachImageForChat(item) }
                )
                .padding(.bottom, 12)
            }
            .padding(.top, 10)
            .animation(.spring(response: 0.4, dampingFraction: 0.8), value: vm.filteredResults.count)
        }
    }

    private var chatSection: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                if !vm.chatTurns.isEmpty {
                    ForEach(vm.chatTurns) { turn in
                        VStack(alignment: .leading, spacing: 8) {
                            HStack {
                                Image(systemName: turn.role == .assistant ? "sparkles" : "person.fill")
                                    .foregroundStyle(turn.role == .assistant ? .yellow : .blue)
                                Text(turn.role == .assistant ? "Assistant" : "You")
                                    .font(.headline)
                                    .foregroundStyle(.secondary)
                                Spacer()
                                if turn.role == .assistant, let confidence = turn.confidence, !confidence.isEmpty {
                                    Text(confidence)
                                        .font(.caption)
                                        .padding(4)
                                        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 4))
                                }
                            }

                            Text(turn.content)
                                .font(.system(size: 16, weight: .regular, design: .rounded))
                                .lineSpacing(4)
                                .foregroundStyle(.primary)
                                .textSelection(.enabled)
                        }
                        .padding(20)
                        .background(.thinMaterial)
                        .cornerRadius(16)
                    }

                    if !vm.chatSources.isEmpty {
                        Text("Latest Sources")
                            .font(.headline)
                            .foregroundStyle(.secondary)
                            .padding(.leading, 4)

                        MasonryGrid(items: vm.chatSources, columns: 3) { item in
                            ResultCard(
                                result: item,
                                openAction: { vm.open(item) },
                                contextAction: item.image_id == nil ? nil : { vm.runContextLens(for: item) },
                                attachChatAction: { vm.attachImageForChat(item) }
                            )
                            .padding(.bottom, 12)
                        }
                    }
                } else if vm.isBusy {
                    VStack(alignment: .leading, spacing: 12) {
                        Rectangle().fill(.white.opacity(0.1)).frame(height: 20).cornerRadius(4)
                        Rectangle().fill(.white.opacity(0.1)).frame(height: 20).cornerRadius(4)
                        Rectangle().fill(.white.opacity(0.1)).frame(width: 200, height: 20).cornerRadius(4)
                    }
                    .padding(20)
                } else {
                    VStack(spacing: 20) {
                        Image(systemName: "message.badge.waveform")
                            .font(.system(size: 48))
                            .foregroundStyle(.secondary.opacity(0.5))
                        Text("Start a continuous chat over your retrieved images.")
                            .font(.title3)
                            .foregroundStyle(.secondary)
                    }
                    .frame(maxWidth: .infinity, minHeight: 300)
                }
            }
            .padding(.top, 10)
            .animation(.spring(response: 0.4), value: vm.chatTurns.count)
        }
    }
    
    // Quick helper for Masonry layout
    struct MasonryGrid<Data: RandomAccessCollection, Content: View>: View where Data.Element: Identifiable {
        let items: Data
        let columns: Int
        let content: (Data.Element) -> Content

        init(items: Data, columns: Int, @ViewBuilder content: @escaping (Data.Element) -> Content) {
            self.items = items
            self.columns = columns
            self.content = content
        }

        var body: some View {
            HStack(alignment: .top, spacing: 16) {
                ForEach(0..<columns, id: \.self) { columnIndex in
                    LazyVStack(spacing: 0) {
                        ForEach(items.filter { index(of: $0) % columns == columnIndex }) { item in
                            content(item)
                        }
                    }
                }
            }
        }
        
        private func index(of item: Data.Element) -> Int {
             // O(N) lookup but fine for small N (topK <= 50)
            items.firstIndex(where: { $0.id == item.id }) as? Int ?? 0
        }
    }

    private var logsSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 10) {
                Text("Command Log")
                    .font(.system(size: 15, weight: .bold, design: .rounded))
                    .foregroundStyle(.white)
                Spacer()
                Button {
                    vm.clearCommandLog()
                } label: {
                    Label("Clear", systemImage: "trash")
                        .font(.system(size: 11, weight: .semibold))
                }
                .buttonStyle(.plain)
                .padding(.horizontal, 10)
                .padding(.vertical, 5)
                .background(Color.white.opacity(0.12), in: Capsule())
                .foregroundStyle(.white.opacity(0.95))
                .help("Clear command log")
            }

            TextEditor(text: $vm.logs)
                .font(.system(size: 12, weight: .regular, design: .monospaced))
                .frame(minHeight: 140, maxHeight: 190)
                .scrollContentBackground(.hidden)
                .background(Color.black.opacity(0.2))
                .overlay(
                    RoundedRectangle(cornerRadius: 12)
                        .stroke(Color.white.opacity(0.1), lineWidth: 1)
                )
                .clipShape(RoundedRectangle(cornerRadius: 12))
                .foregroundStyle(.white)
        }
        .padding(14)
        .background(
            RoundedRectangle(cornerRadius: 18)
                .fill(.ultraThinMaterial)
                .shadow(color: .black.opacity(0.2), radius: 5, x: 0, y: 4)
        )
    }
    }


class AppDelegate: NSObject, NSApplicationDelegate {
    var commandPalette: CommandPaletteWindow?
    var statusItem: NSStatusItem?
    private var hotKeyRef: EventHotKeyRef?
    private var hotKeyHandlerRef: EventHandlerRef?

    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.setActivationPolicy(.accessory)

        // Create Command Palette
        let palette = CommandPaletteWindow()
        let hostingController = NSHostingController(rootView: CommandPaletteView())
        palette.contentViewController = hostingController
        self.commandPalette = palette
        registerGlobalHotkey()
        
        NSEvent.addLocalMonitorForEvents(matching: .keyDown) { event in
            if event.keyCode == 53, let palette = self.commandPalette, palette.isVisible {
                palette.dismissOverlay()
                return nil
            }
            return event
        }
    }

    func applicationWillTerminate(_ notification: Notification) {
        unregisterGlobalHotkey()
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        false
    }

    private func registerGlobalHotkey() {
        var eventType = EventTypeSpec(eventClass: OSType(kEventClassKeyboard), eventKind: UInt32(kEventHotKeyPressed))
        let userData = UnsafeMutableRawPointer(Unmanaged.passUnretained(self).toOpaque())

        let handler: EventHandlerUPP = { _, eventRef, userData in
            guard
                let eventRef,
                let userData
            else { return noErr }

            var hotKeyID = EventHotKeyID()
            let status = GetEventParameter(
                eventRef,
                EventParamName(kEventParamDirectObject),
                EventParamType(typeEventHotKeyID),
                nil,
                MemoryLayout<EventHotKeyID>.size,
                nil,
                &hotKeyID
            )
            guard status == noErr else { return noErr }

            let delegate = Unmanaged<AppDelegate>.fromOpaque(userData).takeUnretainedValue()
            if hotKeyID.id == 1 {
                delegate.togglePalette()
            }
            return noErr
        }

        InstallEventHandler(
            GetApplicationEventTarget(),
            handler,
            1,
            &eventType,
            userData,
            &hotKeyHandlerRef
        )

        let hotKeyID = EventHotKeyID(signature: OSType(0x5353544B), id: 1) // "SSTK"
        RegisterEventHotKey(
            UInt32(kVK_Space),
            UInt32(cmdKey | shiftKey),
            hotKeyID,
            GetApplicationEventTarget(),
            0,
            &hotKeyRef
        )
    }

    private func unregisterGlobalHotkey() {
        if let ref = hotKeyRef {
            UnregisterEventHotKey(ref)
            hotKeyRef = nil
        }
        if let ref = hotKeyHandlerRef {
            RemoveEventHandler(ref)
            hotKeyHandlerRef = nil
        }
    }
    
    @objc func togglePalette() {
        guard let palette = commandPalette else { return }
        if palette.isVisible {
            palette.dismissOverlay()
        } else {
            palette.presentOverlay()
            NSApp.activate(ignoringOtherApps: true)
        }
    }
}

@main
struct SmartStackUIApp: App {
    @StateObject private var vm = SmartStackViewModel.shared
    @Environment(\.openWindow) private var openWindow
    @NSApplicationDelegateAdaptor(AppDelegate.self) var appDelegate

    private func runQuick(_ args: [String]) {
        guard !args.isEmpty else { return }
        DispatchQueue.global(qos: .utility).async {
            let process = Process()
            process.currentDirectoryURL = URL(fileURLWithPath: stackRoot)
            process.executableURL = URL(fileURLWithPath: args[0])
            process.arguments = Array(args.dropFirst())
            process.standardOutput = Pipe()
            process.standardError = Pipe()
            do {
                try process.run()
                process.waitUntilExit()
            } catch {
                NSLog("Quick action failed: \(error)")
            }
        }
    }

    var body: some Scene {
        WindowGroup("Smart Stack", id: "main") {
            ContentView(vm: vm)
                .frame(minWidth: 1120, minHeight: 760)
                .background(WindowAccessor { window in
                    self.vm.mainWindow = window
                })
        }
        .windowStyle(.hiddenTitleBar)
        .windowResizability(.contentSize)

        MenuBarExtra("Smart Stack", systemImage: "sparkles.rectangle.stack") {
            Button("Open Console") {
                openWindow(id: "main")
                NSApp.activate(ignoringOtherApps: true)
            }
            Divider()
            Button("Ingest File/Folder") {
                // We'll use vm methods for consistency if possible, but MenuBarExtra might need a workaround if VM isn't accessible global-style.
                // However, SmartStackApp usually has the VM.
                vm.runIngestPath()
            }
            Button("Ingest Inbox") {
                vm.runInboxIngest()
            }
            Button("Pick Visual Query Image") {
                vm.pickVisualQueryImage()
            }
            Button("Paste Image as Visual Query") {
                vm.pasteClipboardImageForSearch()
            }
            Button("Paste Image and Ingest") {
                vm.pasteClipboardImageAndIngest()
            }
            Button("Semantic Timeline") {
                vm.openTimeline()
            }
            Button("Photo Clusters") {
                vm.openClusters()
            }
            Button("All Indexed Photos") {
                vm.runAllPhotos()
            }
            Button("Auto Cluster Photos") {
                vm.runAutoCluster()
            }
            Button("Rescan Changed") {
                vm.runRescan()
            }
            Button("Safe Reprocess") {
                vm.runSafeReprocess()
            }
            Button("Emergency Kill Switch", role: .destructive) {
                vm.runEmergencyMemoryKillSwitch()
            }
            Divider()
            Button("Quit") {
                NSApp.terminate(nil)
            }
        }
        .menuBarExtraStyle(.menu)
    }
}

struct WindowAccessor: NSViewRepresentable {
    var callback: (NSWindow) -> Void

    func makeNSView(context: Context) -> NSView {
        let view = NSView()
        DispatchQueue.main.async {
            if let window = view.window {
                self.callback(window)
            }
        }
        return view
    }

    func updateNSView(_ nsView: NSView, context: Context) {}
}

struct VisualEffect: NSViewRepresentable {
    var material: NSVisualEffectView.Material
    var blendingMode: NSVisualEffectView.BlendingMode

    func makeNSView(context: Context) -> NSVisualEffectView {
        let view = NSVisualEffectView()
        view.material = material
        view.blendingMode = blendingMode
        view.state = .active
        return view
    }

    func updateNSView(_ nsView: NSVisualEffectView, context: Context) {
        nsView.material = material
        nsView.blendingMode = blendingMode
    }
}
