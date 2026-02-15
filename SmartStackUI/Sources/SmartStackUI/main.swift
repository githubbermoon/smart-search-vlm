import AppKit
import Foundation
import SwiftUI

private let stackRoot = "/Users/pranjal/garage/smart_stack"
private let searchScript = "\(stackRoot)/search.py"
private let guardedIngestScript = "\(stackRoot)/run_guarded_ingest.sh"
private let notesIndexScript = "\(stackRoot)/notes_index.py"
private let sqlitePath = "/Users/pranjal/Pranjal-Obs/clawd/smart_stack.db"
private let venvPython = "\(stackRoot)/.venv/bin/python"

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

@MainActor
final class SmartStackViewModel: ObservableObject {
    @Published var query: String = ""
    @Published var embedModel: String = "nomic-ai/nomic-embed-text-v1.5"
    @Published var searchMode: SearchMode = .semantic
    @Published var sourceFilter: SourceFilter = .all
    @Published var includeNotes: Bool = false
    @Published var topK: Int = 8
    @Published var minScore: Double = 0.0

    @Published var isBusy: Bool = false
    @Published var results: [SearchResult] = []
    @Published var logs: String = "Ready."

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

        switch searchMode {
        case .semantic:
            runSemanticSearch(query: q)
        case .keyword:
            runKeywordSearch(query: q)
        }
    }

    func runSafeReprocess() {
        var args = [guardedIngestScript, "--safe-reprocess", "--embed-model", embedModel]
        args.append("--no-print-fields")
        runCommand(args: args, title: "Safe Reprocess") { _, _ in }
    }

    func runInboxIngest() {
        var args = [guardedIngestScript, "--embed-model", embedModel]
        args.append("--no-print-fields")
        runCommand(args: args, title: "Inbox Ingest") { _, _ in }
    }

    func runNotesIndex() {
        let args = [venvPython, notesIndexScript, "--embed-model", embedModel]
        runCommand(args: args, title: "Notes Index") { _, _ in }
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

        runCommand(args: args, title: "Semantic Search") { output, code in
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

        let args = ["sqlite3", "-json", sqlitePath, sql]
        runCommand(args: args, title: "Hybrid-Key Search") { output, code in
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

    private func runCommand(args: [String], title: String, completion: @escaping (String, Int32) -> Void) {
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
                    if !combined.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                        self.appendLog(combined)
                    }
                    completion(combined, process.terminationStatus)
                }
            } catch {
                DispatchQueue.main.async {
                    self.isBusy = false
                    self.appendLog("[\(title)] Failed to run command: \(error)")
                    completion("", -1)
                }
            }
        }
    }

    private func appendLog(_ line: String) {
        logs += logs.isEmpty ? line : "\n\(line)"
    }
}

struct ResultCard: View {
    let result: SearchResult
    let openAction: () -> Void

    private var badgeColor: Color {
        result.source == "note" ? .orange : .mint
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Text(result.sourceTitle)
                    .font(.system(size: 11, weight: .semibold, design: .rounded))
                    .padding(.horizontal, 8)
                    .padding(.vertical, 4)
                    .background(badgeColor.opacity(0.25))
                    .clipShape(Capsule())

                Spacer()

                Text("score \(result.score)")
                    .font(.system(size: 12, weight: .medium, design: .monospaced))
                    .foregroundStyle(.secondary)
            }

            Text(result.filename)
                .font(.system(size: 18, weight: .bold, design: .serif))

            Text(result.caption.isEmpty ? "(no caption/snippet)" : result.caption)
                .font(.system(size: 13, weight: .regular, design: .rounded))
                .foregroundStyle(.secondary)
                .lineLimit(4)

            if !result.tags.isEmpty {
                Text(result.tags.joined(separator: " • "))
                    .font(.system(size: 12, weight: .semibold, design: .rounded))
                    .foregroundStyle(.primary.opacity(0.8))
            }

            HStack {
                Text(result.obsidian_path)
                    .font(.system(size: 11, weight: .regular, design: .monospaced))
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
                    .truncationMode(.middle)

                Spacer()

                Button("Open") {
                    openAction()
                }
                .buttonStyle(.borderedProminent)
                .tint(.teal)
            }
        }
        .padding(14)
        .background(
            RoundedRectangle(cornerRadius: 16)
                .fill(.thinMaterial)
                .shadow(color: .black.opacity(0.15), radius: 4, x: 0, y: 2)
        )
        .overlay(
            RoundedRectangle(cornerRadius: 16)
                .stroke(Color.white.opacity(0.14), lineWidth: 1)
        )
    }
}

struct ContentView: View {
    @StateObject private var vm = SmartStackViewModel()
    @FocusState private var queryFocused: Bool

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



            VStack(spacing: 16) {
                header
                controls
                resultsSection
                logsSection
            }
            .padding(20)
        }
    }

    private var header: some View {
        HStack(alignment: .lastTextBaseline) {
            Text("Smart Stack Console")
                .font(.system(size: 38, weight: .bold, design: .serif))
                .tracking(-0.5)
                .shadow(color: .black.opacity(0.5), radius: 4, x: 0, y: 2)
            Spacer()
            if vm.isBusy {
                ProgressView()
                    .controlSize(.small)
            }
        }
        .foregroundStyle(.white)
    }

    private var controls: some View {
        VStack(spacing: 12) {
            HStack(spacing: 10) {

                TextField("Search query", text: $vm.query)
                    .textFieldStyle(.plain)
                    .padding(8)
                    .background(.thinMaterial, in: RoundedRectangle(cornerRadius: 8))
                    .font(.system(size: 14, weight: .medium, design: .rounded))
                    .foregroundStyle(.white)
                    .focused($queryFocused)

                Picker("Mode", selection: $vm.searchMode) {
                    ForEach(SearchMode.allCases) { mode in
                        Text(mode.rawValue).tag(mode)
                    }
                }
                .pickerStyle(.segmented)
                .frame(width: 220)

                Button("Search") { vm.runSearch() }
                    .buttonStyle(.borderedProminent)
                    .tint(.orange)
            }

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

                Button("Ingest Inbox") { vm.runInboxIngest() }
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
        .onAppear {
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) {
                queryFocused = true
            }
        }
    }

    private var resultsSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Text("Results")
                    .font(.system(size: 22, weight: .bold, design: .serif))
                Text("\(vm.filteredResults.count)")
                    .font(.system(size: 14, weight: .bold, design: .rounded))
                    .padding(.horizontal, 9)
                    .padding(.vertical, 3)
                    .background(Color.white.opacity(0.15))
                    .clipShape(Capsule())
                Spacer()
            }
            .foregroundStyle(.white)

            ScrollView {
                LazyVStack(spacing: 10) {
                    ForEach(vm.filteredResults) { row in
                        ResultCard(result: row) {
                            vm.open(row)
                        }
                    }
                }
                .padding(.bottom, 8)
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
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
            ContentView()
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
            Button("Ingest Inbox") {
                runQuick([guardedIngestScript, "--no-print-fields"])
            }
            Button("Safe Reprocess") {
                runQuick([guardedIngestScript, "--safe-reprocess", "--no-print-fields"])
            }
            Button("Index Notes") {
                runQuick([venvPython, notesIndexScript])
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
