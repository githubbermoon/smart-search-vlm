import SwiftUI
import AppKit

class CommandPaletteWindow: NSPanel {
    private static let minOverlayWidth: CGFloat = 760
    private static let maxOverlayWidth: CGFloat = 1120
    private static let minOverlayHeight: CGFloat = 280
    private static let maxOverlayHeight: CGFloat = 620

    init() {
        super.init(
            contentRect: NSRect(x: 0, y: 0, width: 900, height: 440),
            styleMask: [.nonactivatingPanel, .borderless, .fullSizeContentView],
            backing: .buffered,
            defer: false
        )
        self.isReleasedWhenClosed = false
        self.isFloatingPanel = true
        self.level = .statusBar
        self.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary, .transient, .ignoresCycle]
        self.hidesOnDeactivate = true
        self.isMovableByWindowBackground = false
        self.isOpaque = false
        self.backgroundColor = .clear
        self.hasShadow = true
        self.minSize = NSSize(width: Self.minOverlayWidth, height: Self.minOverlayHeight)
        self.maxSize = NSSize(width: Self.maxOverlayWidth, height: Self.maxOverlayHeight)
    }
    
    override var canBecomeKey: Bool {
        return true
    }

    override var canBecomeMain: Bool {
        return true
    }

    func presentOverlay() {
        let mouse = NSEvent.mouseLocation
        let activeScreen = NSScreen.screens.first(where: { $0.frame.contains(mouse) })
        guard let screen = activeScreen ?? NSScreen.main ?? NSScreen.screens.first else {
            makeKeyAndOrderFront(nil)
            return
        }
        let visible = screen.visibleFrame
        let width = min(Self.maxOverlayWidth, max(Self.minOverlayWidth, visible.width * 0.72))
        let height = min(Self.maxOverlayHeight, max(Self.minOverlayHeight, visible.height * 0.46))
        let x = visible.midX - (width / 2.0)
        let y = visible.maxY - height - 72.0
        setFrame(NSRect(x: x, y: y, width: width, height: height), display: true, animate: true)
        orderFrontRegardless()
        makeKey()
    }

    func dismissOverlay() {
        orderOut(nil)
    }
}

struct CommandPaletteView: View {
    @State private var query: String = ""
    @State private var isHovering: Bool = false
    @StateObject var viewModel = SmartStackViewModel.shared
    @FocusState private var isFocused: Bool
    @State private var submittedQuery: String = ""

    private var canSearch: Bool {
        query.trimmingCharacters(in: .whitespacesAndNewlines).count >= 3
    }

    private var showResults: Bool {
        !submittedQuery.isEmpty && submittedQuery == query.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private func submitSearch() {
        let trimmed = query.trimmingCharacters(in: .whitespacesAndNewlines)
        guard trimmed.count >= 3 else { return }
        submittedQuery = trimmed
        Task {
            await viewModel.performSearch(query: trimmed)
        }
    }
    
    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack(alignment: .center, spacing: 10) {
                Image(systemName: "sparkle.magnifyingglass")
                    .font(.system(size: 15, weight: .bold))
                    .foregroundStyle(.secondary)
                Text("Smart Stack Search")
                    .font(.system(size: 15, weight: .semibold, design: .rounded))
                    .lineLimit(1)
                Spacer()
                Text("Enter to run")
                    .font(.system(size: 11, weight: .medium, design: .rounded))
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
            }

            HStack(spacing: 10) {
                Image(systemName: "magnifyingglass")
                    .font(.system(size: 17, weight: .semibold))
                    .foregroundStyle(.secondary)

                TextField("Type query (min 3 chars)", text: $query)
                    .font(.system(size: 21, weight: .medium, design: .rounded))
                    .textFieldStyle(.plain)
                    .focused($isFocused)
                    .onSubmit {
                        submitSearch()
                    }

                if !query.isEmpty {
                    Button {
                        query = ""
                        submittedQuery = ""
                    } label: {
                        Image(systemName: "xmark.circle.fill")
                            .foregroundStyle(.secondary)
                    }
                    .buttonStyle(.plain)
                }

                Button {
                    submitSearch()
                } label: {
                    Text("Search")
                        .font(.system(size: 13, weight: .semibold))
                        .padding(.horizontal, 12)
                        .padding(.vertical, 8)
                }
                .buttonStyle(.plain)
                .background(canSearch ? Color.blue.opacity(0.2) : Color.gray.opacity(0.15), in: Capsule())
                .foregroundStyle(canSearch ? Color.blue : Color.secondary)
                .disabled(!canSearch || viewModel.isBusy)
            }
            .padding(.horizontal, 14)
            .padding(.vertical, 12)
            .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 14))
            .overlay(
                RoundedRectangle(cornerRadius: 14)
                    .stroke(isHovering ? Color.blue.opacity(0.45) : Color.white.opacity(0.12), lineWidth: 1)
            )

            if viewModel.isBusy {
                HStack(spacing: 8) {
                    ProgressView()
                        .controlSize(.small)
                    Text("Searching locally...")
                        .font(.system(size: 12, weight: .medium))
                        .foregroundStyle(.secondary)
                }
            } else if !showResults {
                Text("Drag file here to ingest. Search runs only after pressing Enter or Search.")
                    .font(.system(size: 12, weight: .regular))
                    .foregroundStyle(.secondary)
            }

            if showResults && !viewModel.searchResults.isEmpty {
                ScrollView {
                    VStack(alignment: .leading, spacing: 8) {
                        ForEach(viewModel.searchResults) { result in
                            ResultRow(result: result)
                                .onTapGesture {
                                    NSWorkspace.shared.open(URL(fileURLWithPath: result.filePath))
                                }
                        }
                    }
                    .padding(.top, 10)
                }
                .frame(maxHeight: 500)
                .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 14))
                .cornerRadius(16)
            } else if showResults && !viewModel.isBusy {
                Text("No results.")
                    .font(.system(size: 12, weight: .medium))
                    .foregroundStyle(.secondary)
                    .padding(.vertical, 8)
            }
        }
        .padding(20)
        .frame(minWidth: 760, maxWidth: 1120, minHeight: 250, maxHeight: .infinity, alignment: .topLeading)
        .background(
            LinearGradient(
                colors: [Color.black.opacity(0.22), Color.black.opacity(0.10)],
                startPoint: .topLeading,
                endPoint: .bottomTrailing
            )
        )
        .onAppear {
            isFocused = true
        }
        .onDrop(of: ["public.file-url"], isTargeted: $isHovering) { providers in
            for provider in providers {
                provider.loadItem(forTypeIdentifier: "public.file-url", options: nil) { (urlData, error) in
                    if let urlData = urlData as? Data,
                       let url = URL(dataRepresentation: urlData, relativeTo: nil) {
                        DispatchQueue.main.async {
                            viewModel.ingestPath(url.path)
                        }
                    }
                }
            }
            return true
        }
    }
}

struct ResultRow: View {
    let result: SearchResultItem
    
    var body: some View {
        HStack(alignment: .top, spacing: 12) {
            AsyncImage(url: URL(fileURLWithPath: result.filePath)) { image in
                image.resizable().aspectRatio(contentMode: .fill)
            } placeholder: {
                Color.gray.opacity(0.2)
            }
            .frame(width: 40, height: 40)
            .cornerRadius(6)
            
            VStack(alignment: .leading, spacing: 2) {
                Text(URL(fileURLWithPath: result.filePath).lastPathComponent)
                    .font(.headline)
                    .foregroundColor(.primary)
                
                if !result.caption.isEmpty {
                    Text(result.caption)
                        .font(.caption)
                        .foregroundColor(.secondary)
                        .lineLimit(2)
                }
            }
            Spacer()
            
            Text(String(format: "%.2f", result.score))
                .font(.caption2)
                .padding(4)
                .background(Color.blue.opacity(0.1))
                .foregroundColor(.blue)
                .cornerRadius(4)
        }
        .padding(8)
        .background(Color.primary.opacity(0.05))
        .cornerRadius(8)
    }
}
