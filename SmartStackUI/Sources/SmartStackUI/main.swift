import AppKit
import Foundation
import SwiftUI

private let stackRoot = "/Users/pranjal/garage/smart_stack"
private let searchScript = "\(stackRoot)/search.py"
private let guardedIngestScript = "\(stackRoot)/run_guarded_ingest.sh"
private let notesIndexScript = "\(stackRoot)/notes_index.py"
private let sqlitePath = "/Users/pranjal/Pranjal-Obs/clawd/smart_stack.db"
private let venvPython = "\(stackRoot)/.venv/bin/python"
private let mmCliScript = "\(stackRoot)/mm_cli.py"

enum SearchMode: String, CaseIterable, Identifiable {
    case semantic = "Semantic"
    case keyword = "Keyword"

    var id: String { rawValue }
}

enum SourceFilter: String, CaseIterable, Identifiable {
    case all = "All"
    case image = "Images"
    case note = "Notes"

    var id: String { rawValue }
}

struct SearchResult: Identifiable, Decodable {
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

struct SearchPayload: Decodable {
    let query: String
    let embed_model: String
    let top_k: Int
    let results: [SearchResult]
}

struct KeywordRow: Decodable {
    let filename: String?
    let caption: String?
    let tags: String?
    let obsidian_path: String?
}

// New Multimodal Response Structs
struct MultimodalResponse: Decodable {
    let routing_mode: String
    let results: [MultimodalResultItem]
}

struct MultimodalResultItem: Decodable {
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

@MainActor
final class SmartStackViewModel: ObservableObject {
    @Published var query: String = ""
    @Published var embedModel: String = "nomic-ai/nomic-embed-text-v1.5"
    @Published var searchMode: SearchMode = .semantic
    @Published var sourceFilter: SourceFilter = .all
    @Published var includeNotes: Bool = false
    @Published var topK: Int = 8
    @Published var minScore: Double = 0.0
    @Published var useMultimodal: Bool = false
    @Published var isChatMode: Bool = false // Toggle between Search/Chat
    
    @Published var chatAnswer: String = ""
    @Published var chatSources: [SearchResult] = []
    @Published var chatConfidence: String = ""

    @Published var isBusy: Bool = false
    @Published var results: [SearchResult] = []
    @Published var logs: String = "Ready."
    @Published var showSettings: Bool = false
    @Published var watchedFolders: [WatchedFolder] = []
    @Published var exclusions: [ExclusionPattern] = []

    var filteredResults: [SearchResult] {
        results.filter { row in
            let sourceOK: Bool
            switch sourceFilter {
            case .all:
                sourceOK = true
            case .image:
                sourceOK = row.source == "image"
            case .note:
                sourceOK = row.source == "note"
            }
            return sourceOK && row.numericScore >= minScore
        }
    }

    func runSearch() {
        let q = query.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !q.isEmpty else {
            appendLog("Search query is empty.")
            return
        }

        if useMultimodal {
            runMultimodalSearch(query: q)
        } else {
            switch searchMode {
            case .semantic:
                runSemanticSearch(query: q)
            case .keyword:
                runKeywordSearch(query: q)
            }
        }
    }

    func runChat() {
        let q = query.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !q.isEmpty else { return }
        
        // mm_cli.py chat "query" --json
        let args = [venvPython, mmCliScript, "chat", q, "--json"]
        
        // Clear previous
        self.chatAnswer = ""
        self.chatSources = []
        
        runCommand(args: args, title: "Chat") { output, stderr, code in
            guard code == 0 else {
                self.appendLog("Chat failed code \(code).")
                self.chatAnswer = "Error: Chat failed. Check logs."
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
                        source: "image", 
                        filename: url.lastPathComponent,
                        caption: item.caption,
                        tags: item.tags,
                        score: String(format: "%.4f", item.score),
                        obsidian_path: item.file_path
                    )
                }
            } catch {
                self.appendLog("Chat Parse Error: \(error)")
                self.chatAnswer = "Error parsing response."
            }
        }
    }

    func runSafeReprocess() {
        if useMultimodal {
            // New Stack Safe Reprocess
            // mm_cli.py ingest-inbox --safe-reprocess
            let args = [venvPython, mmCliScript, "ingest-inbox", "--safe-reprocess"]
            runCommand(args: args, title: "MM Safe Reprocess") { _, _, _ in }
        } else {
            // Legacy Safe Reprocess
            var args = [guardedIngestScript, "--safe-reprocess", "--embed-model", embedModel]
            args.append("--no-print-fields")
            runCommand(args: args, title: "Safe Reprocess") { _, _, _ in }
        }
    }

    func runInboxIngest() {
        if useMultimodal {
            // New Stack Ingest
            // mm_cli.py ingest-inbox --limit 0
            let args = [venvPython, mmCliScript, "ingest-inbox"]
            runCommand(args: args, title: "MM Inbox Ingest") { _, _, _ in }
        } else {
            // Legacy Ingest
            var args = [guardedIngestScript, "--embed-model", embedModel]
            args.append("--no-print-fields")
            runCommand(args: args, title: "Inbox Ingest") { _, _, _ in }
        }
    }

    func runNotesIndex() {
        let args = [venvPython, notesIndexScript, "--embed-model", embedModel]
        runCommand(args: args, title: "Notes Index") { _, _, _ in }
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

    private func runSemanticSearch(query: String) {
        var args = [
            venvPython,
            searchScript,
            query,
            "--embed-model",
            embedModel,
            "-n",
            "\(max(1, topK))",
            "--min-score",
            String(format: "%.4f", minScore),
            "--json",
        ]
        if !includeNotes {
            args.append("--no-notes")
        }

        runCommand(args: args, title: "Semantic Search") { output, stderr, code in
            guard code == 0 else {
                self.appendLog("Semantic search failed with code \(code).")
                return
            }

            let marker = "@@SMARTSTACK_JSON@@"
            let jsonLine = output.split(separator: "\n").map(String.init).last { $0.contains(marker) }
            guard let line = jsonLine, let payload = line.components(separatedBy: marker).last else {
                self.appendLog("Could not parse JSON output from search.py.")
                return
            }

            guard let data = payload.data(using: .utf8) else {
                self.appendLog("Search output is not valid UTF-8 JSON.")
                return
            }

            do {
                let parsed = try JSONDecoder().decode(SearchPayload.self, from: data)
                let vectorResults = parsed.results
                
                // Chain keyword search for hybrid results
                self.executeKeywordSearch(query: query) { keywordResults in
                    var mergedMap: [String: SearchResult] = [:]
                    
                    // 1. Add vector results first
                    for res in vectorResults {
                        mergedMap[res.obsidian_path] = res
                    }
                    
                    // 2. Overlay keyword results (boost score to 1.0+)
                    for res in keywordResults {
                        // If it existed, we overwrite with the boosted/exact match version (or just update score)
                        // A keyword match is definitely a 1.0 relevance or higher.
                        var boosted = res
                        boosted.score = "1.0000" // Force high score for keyword match
                        mergedMap[res.obsidian_path] = boosted
                    }
                    
                    let finalResults = mergedMap.values.sorted { $0.numericScore > $1.numericScore }
                    self.results = Array(finalResults.prefix(self.topK))
                    self.appendLog("Hybrid search partials: \(vectorResults.count) vector, \(keywordResults.count) keyword. Merged: \(finalResults.count).")
                }
            } catch {
                self.appendLog("JSON parse error: \(error)")
            }
        }
    }

    private func runMultimodalSearch(query: String) {
        // mm_cli.py search "query" -n topK --json
        let args = [
            venvPython,
            mmCliScript,
            "search",
            query,
            "-n",
            "\(max(1, topK))",
            "--json"
        ]

        runCommand(args: args, title: "Multimodal Search") { output, stderr, code in
            guard code == 0 else {
                self.appendLog("MM Search failed code \(code).")
                return
            }
            
            // Parse MM JSON
            guard let data = output.data(using: .utf8) else { return }
            do {
                let resp = try JSONDecoder().decode(MultimodalResponse.self, from: data)
                
                // Map to existing SearchResult
                let mapped = resp.results.map { item -> SearchResult in
                    let url = URL(fileURLWithPath: item.file_path)
                    return SearchResult(
                        source: "image", // MM stack is primarily image/hybrid for now
                        filename: url.lastPathComponent,
                        caption: item.caption,
                        tags: item.tags,
                        score: String(format: "%.4f", item.score),
                        obsidian_path: item.file_path
                    )
                }
                
                self.results = mapped
                self.appendLog("MM Search: \(resp.routing_mode) mode, \(mapped.count) results.")
            } catch {
                // If it's just raw JSON not matching our struct
                self.appendLog("MM Parse Error: \(error)")
            }
        }
    }

    private func executeKeywordSearch(query: String, completion: @escaping ([SearchResult]) -> Void) {
        let escaped = query.lowercased().replacingOccurrences(of: "'", with: "''")
        let sql = """
        select filename, caption, tags, obsidian_path
        from processed_images
        where lower(filename) like '%\(escaped)%'
           or lower(caption) like '%\(escaped)%'
           or lower(tags) like '%\(escaped)%'
           or lower(ocr_text) like '%\(escaped)%'
        order by processed_at desc
        limit \(max(1, topK));
        """

        let args = ["/usr/bin/sqlite3", "-json", sqlitePath, sql]
        runCommand(args: args, title: "Hybrid-Key Search") { output, stderr, code in
            guard code == 0 else {
                completion([])
                return
            }

            let text = output.trimmingCharacters(in: .whitespacesAndNewlines)
            guard let data = text.data(using: .utf8), !text.isEmpty else {
                completion([])
                return
            }

            do {
                let rows = try JSONDecoder().decode([KeywordRow].self, from: data)
                let results = rows.map { row in
                    SearchResult(
                        source: "image",
                        filename: row.filename ?? "unknown",
                        caption: row.caption ?? "",
                        tags: self.decodeTags(row.tags),
                        score: "1.0000",
                        obsidian_path: row.obsidian_path ?? ""
                    )
                }
                completion(results)
            } catch {
                self.appendLog("Keyword JSON parse error: \(error)")
                completion([])
            }
        }
    }

    private func runKeywordSearch(query: String) {
        executeKeywordSearch(query: query) { results in
            self.results = results
            self.appendLog("Keyword search complete: \(results.count) results.")
        }
    }

    private func decodeTags(_ raw: String?) -> [String] {
        guard let raw, !raw.isEmpty, let data = raw.data(using: .utf8) else {
            return []
        }
        if let parsed = try? JSONDecoder().decode([String].self, from: data) {
            return parsed
        }
        return raw
            .split(separator: ",")
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
    }

    private func runCommand(args: [String], title: String, completion: @escaping (String, String, Int32) -> Void) {
        guard !args.isEmpty else { return }
        isBusy = true
        appendLog("\n[\(title)] $ \(args.joined(separator: " "))")

        DispatchQueue.global(qos: .userInitiated).async {
            let process = Process()
            process.currentDirectoryURL = URL(fileURLWithPath: stackRoot)
            process.executableURL = URL(fileURLWithPath: args[0])
            process.arguments = Array(args.dropFirst())

            let outPipe = Pipe()
            let errPipe = Pipe()
            process.standardOutput = outPipe
            process.standardError = errPipe

            do {
                try process.run()
                process.waitUntilExit()

                let outData = outPipe.fileHandleForReading.readDataToEndOfFile()
                let errData = errPipe.fileHandleForReading.readDataToEndOfFile()
                let outText = String(data: outData, encoding: .utf8) ?? ""
                let errText = String(data: errData, encoding: .utf8) ?? ""
                let combined = [outText, errText].filter { !$0.isEmpty }.joined(separator: "\n")

                DispatchQueue.main.async {
                    self.isBusy = false
                    // Log output (combined) for debugging visibility
                    if !combined.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                        self.appendLog(combined)
                    }
                    // Return distinct streams for robust parsing
                    completion(outText, errText, process.terminationStatus)
                }
            } catch {
                DispatchQueue.main.async {
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

struct ResultCard: View {
    let result: SearchResult
    let openAction: () -> Void
    @State private var isHovering = false

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
    }

    private var header: some View {
        HStack(spacing: 16) {
             // Search Bar
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
            }
            .padding(12)
            .background(.ultraThinMaterial, in: Capsule())
            .background(.ultraThinMaterial, in: Capsule())
            .shadow(color: .black.opacity(0.1), radius: 4, x: 0, y: 2)

            // Chat Mode Toggle
            // Custom Mode Toggle
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

            // Settings Toggle
            Menu {
                // Mode
                Picker("Mode", selection: $vm.searchMode) {
                    ForEach(SearchMode.allCases) { mode in
                        Text(mode.rawValue).tag(mode)
                    }
                }
                
                Divider()
                
                // Embed Model
                TextField("Embedding Model", text: $vm.embedModel)
                    .disabled(vm.searchMode == .keyword)
                
                Toggle("Include Notes", isOn: $vm.includeNotes)
                    .disabled(vm.searchMode == .keyword)

                Divider()

                // Filters
                Picker("Filter", selection: $vm.sourceFilter) {
                    ForEach(SourceFilter.allCases) { f in
                        Text(f.rawValue).tag(f)
                    }
                }
                
                Divider()
                
                // Multimodal Toggle
                Button {
                    vm.useMultimodal.toggle()
                } label: {
                    HStack {
                        Text("Pro Stack (Multimodal)")
                        if vm.useMultimodal {
                            Image(systemName: "checkmark")
                        }
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
                Button("Ingest File/Folder") { vm.runIngestPath() }
                Button("Ingest Inbox") { vm.runInboxIngest() }
                Button("Rescan Changed") { vm.runRescan() }
                Button("Safe Reprocess") { vm.runSafeReprocess() }
                Button("Index Notes") { vm.runNotesIndex() }

                Divider()

                Button("Settings...") { vm.showSettings = true }
                
            } label: {
                Image(systemName: "slider.horizontal.3")
                    .font(.system(size: 18, weight: .semibold))
                    .foregroundStyle(.secondary)
                    .padding(10)
                    .background(.ultraThinMaterial, in: Circle())
            }
            .menuStyle(.button)
            .buttonStyle(.plain)
            
            // Toggle for Expanded/Focus Mode
            Button {
                withAnimation(.spring(response: 0.4, dampingFraction: 0.7)) {
                    isExpandedMode.toggle()
                }
            } label: {
                Image(systemName: isExpandedMode ? "chevron.up" : "chevron.down")
                    .font(.system(size: 14, weight: .bold))
                    .foregroundStyle(.secondary)
                    .padding(10)
                    .background(.ultraThinMaterial, in: Circle())
            }
            .buttonStyle(.plain)
            .help("Toggle Expanded Controls")

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
                TextField("Embedding model", text: $vm.embedModel)
                    .textFieldStyle(.plain)
                    .padding(6)
                    .background(.thinMaterial, in: RoundedRectangle(cornerRadius: 6))
                    .font(.system(size: 12, weight: .medium, design: .monospaced))
                    .foregroundStyle(.secondary)
                    .disabled(vm.searchMode == .keyword)

                Toggle("Include Notes", isOn: $vm.includeNotes)
                    .disabled(vm.searchMode == .keyword)

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
                Button("Index Notes") { vm.runNotesIndex() }
                    .buttonStyle(.bordered)
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
                ResultCard(result: item) {
                    vm.open(item)
                }
                .padding(.bottom, 12)
            }
            .padding(.top, 10)
            .animation(.spring(response: 0.4, dampingFraction: 0.8), value: vm.filteredResults.count)
        }
    }

    private var chatSection: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                if !vm.chatAnswer.isEmpty {
                    VStack(alignment: .leading, spacing: 12) {
                        HStack {
                            Image(systemName: "sparkles")
                                .foregroundStyle(.yellow)
                            Text("Answer")
                                .font(.headline)
                                .foregroundStyle(.secondary)
                            Spacer()
                            if !vm.chatConfidence.isEmpty {
                                Text(vm.chatConfidence)
                                    .font(.caption)
                                    .padding(4)
                                    .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 4))
                            }
                        }
                        
                        Text(vm.chatAnswer)
                            .font(.system(size: 16, weight: .regular, design: .rounded))
                            .lineSpacing(4)
                            .foregroundStyle(.primary)
                            .textSelection(.enabled)
                    }
                    .padding(20)
                    .background(.thinMaterial)
                    .cornerRadius(16)
                    
                    if !vm.chatSources.isEmpty {
                        Text("Sources")
                            .font(.headline)
                            .foregroundStyle(.secondary)
                            .padding(.leading, 4)
                        
                        MasonryGrid(items: vm.chatSources, columns: 3) { item in
                            ResultCard(result: item) {
                                vm.open(item)
                            }
                            .padding(.bottom, 12)
                        }
                    }
                } else if vm.isBusy { 
                     // Skeleton / Loading state
                     VStack(alignment: .leading, spacing: 12) {
                        Rectangle().fill(.white.opacity(0.1)).frame(height: 20).cornerRadius(4)
                        Rectangle().fill(.white.opacity(0.1)).frame(height: 20).cornerRadius(4)
                        Rectangle().fill(.white.opacity(0.1)).frame(width: 200, height: 20).cornerRadius(4)
                     }
                     .padding(20)
                } else {
                    // Empty state
                    VStack(spacing: 20) {
                        Image(systemName: "message.badge.waveform")
                            .font(.system(size: 48))
                            .foregroundStyle(.secondary.opacity(0.5))
                        Text("Ask questions about your images.")
                            .font(.title3)
                            .foregroundStyle(.secondary)
                    }
                    .frame(maxWidth: .infinity, minHeight: 300)
                }
            }
            .padding(.top, 10)
            .animation(.spring(response: 0.4), value: vm.chatAnswer)
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
            Text("Command Log")
                .font(.system(size: 15, weight: .bold, design: .rounded))
                .foregroundStyle(.white)

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


@main
struct SmartStackUIApp: App {
    @StateObject private var vm = SmartStackViewModel()
    @Environment(\.openWindow) private var openWindow

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
            Button("Rescan Changed") {
                vm.runRescan()
            }
            Button("Safe Reprocess") {
                vm.runSafeReprocess()
            }
            Button("Index Notes") {
                vm.runNotesIndex()
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
