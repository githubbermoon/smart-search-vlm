import AppKit
import Carbon.HIToolbox
import SwiftUI

extension Notification.Name {
    static let smartStackGlobalShortcutChanged = Notification.Name("SmartStackGlobalShortcutChanged")
}

struct GlobalShortcutPreference {
    static let keyCodeKey = "smartStack.globalShortcut.keyCode"
    static let modifierFlagsKey = "smartStack.globalShortcut.modifierFlags"
    static let displayKey = "smartStack.globalShortcut.display"

    static let defaultKeyCode = UInt32(kVK_Space)
    static let defaultModifierFlags = NSEvent.ModifierFlags.option.rawValue
    static let defaultDisplay = "⌥Space"

    let keyCode: UInt32
    let modifierFlags: NSEvent.ModifierFlags
    let display: String

    static var current: GlobalShortcutPreference {
        let defaults = UserDefaults.standard
        guard defaults.object(forKey: keyCodeKey) != nil else {
            return GlobalShortcutPreference(
                keyCode: defaultKeyCode,
                modifierFlags: NSEvent.ModifierFlags(rawValue: defaultModifierFlags),
                display: defaultDisplay
            )
        }
        return GlobalShortcutPreference(
            keyCode: UInt32(defaults.integer(forKey: keyCodeKey)),
            modifierFlags: NSEvent.ModifierFlags(
                rawValue: UInt(defaults.integer(forKey: modifierFlagsKey))
            ),
            display: defaults.string(forKey: displayKey) ?? defaultDisplay
        )
    }

    static func save(keyCode: UInt32, modifierFlags: NSEvent.ModifierFlags, display: String) {
        let defaults = UserDefaults.standard
        defaults.set(Int(keyCode), forKey: keyCodeKey)
        defaults.set(UInt64(modifierFlags.rawValue), forKey: modifierFlagsKey)
        defaults.set(display, forKey: displayKey)
        NotificationCenter.default.post(name: .smartStackGlobalShortcutChanged, object: nil)
    }

    static func reset() {
        save(
            keyCode: defaultKeyCode,
            modifierFlags: NSEvent.ModifierFlags(rawValue: defaultModifierFlags),
            display: defaultDisplay
        )
    }

    var carbonModifiers: UInt32 {
        var value: UInt32 = 0
        if modifierFlags.contains(.command) { value |= UInt32(cmdKey) }
        if modifierFlags.contains(.option) { value |= UInt32(optionKey) }
        if modifierFlags.contains(.control) { value |= UInt32(controlKey) }
        if modifierFlags.contains(.shift) { value |= UInt32(shiftKey) }
        return value
    }
}

@MainActor
final class GlobalShortcutRegistrationState: ObservableObject {
    static let shared = GlobalShortcutRegistrationState()

    @Published var message = "The shortcut will be available while SmartStack is running."
    @Published var isRegistered = true

    private init() {}

    func update(registered: Bool, message: String) {
        isRegistered = registered
        self.message = message
    }
}

private func shortcutDisplay(for event: NSEvent, modifiers: NSEvent.ModifierFlags) -> String {
    var output = ""
    if modifiers.contains(.control) { output += "⌃" }
    if modifiers.contains(.option) { output += "⌥" }
    if modifiers.contains(.shift) { output += "⇧" }
    if modifiers.contains(.command) { output += "⌘" }

    let keyName: String
    switch Int(event.keyCode) {
    case kVK_Space: keyName = "Space"
    case kVK_Return: keyName = "Return"
    case kVK_Tab: keyName = "Tab"
    case kVK_Delete: keyName = "Delete"
    case kVK_ForwardDelete: keyName = "Forward Delete"
    case kVK_LeftArrow: keyName = "←"
    case kVK_RightArrow: keyName = "→"
    case kVK_UpArrow: keyName = "↑"
    case kVK_DownArrow: keyName = "↓"
    case kVK_Home: keyName = "Home"
    case kVK_End: keyName = "End"
    case kVK_PageUp: keyName = "Page Up"
    case kVK_PageDown: keyName = "Page Down"
    case kVK_F1: keyName = "F1"
    case kVK_F2: keyName = "F2"
    case kVK_F3: keyName = "F3"
    case kVK_F4: keyName = "F4"
    case kVK_F5: keyName = "F5"
    case kVK_F6: keyName = "F6"
    case kVK_F7: keyName = "F7"
    case kVK_F8: keyName = "F8"
    case kVK_F9: keyName = "F9"
    case kVK_F10: keyName = "F10"
    case kVK_F11: keyName = "F11"
    case kVK_F12: keyName = "F12"
    default:
        let raw = event.charactersIgnoringModifiers?.uppercased() ?? ""
        keyName = raw.isEmpty ? "Key \(event.keyCode)" : raw
    }
    return output + keyName
}

private final class ShortcutRecorderNSView: NSView {
    var display = GlobalShortcutPreference.current.display {
        didSet { needsDisplay = true }
    }
    var onCapture: ((UInt32, NSEvent.ModifierFlags, String) -> Void)?
    private var isRecording = false

    override var acceptsFirstResponder: Bool { true }
    override var intrinsicContentSize: NSSize { NSSize(width: 190, height: 34) }

    override func mouseDown(with event: NSEvent) {
        isRecording = true
        window?.makeFirstResponder(self)
        needsDisplay = true
    }

    override func resignFirstResponder() -> Bool {
        isRecording = false
        needsDisplay = true
        return super.resignFirstResponder()
    }

    override func keyDown(with event: NSEvent) {
        if event.keyCode == UInt16(kVK_Escape) {
            isRecording = false
            window?.makeFirstResponder(nil)
            needsDisplay = true
            return
        }

        let modifiers = event.modifierFlags.intersection([.command, .option, .control, .shift])
        guard !modifiers.isEmpty else {
            NSSound.beep()
            return
        }

        let rendered = shortcutDisplay(for: event, modifiers: modifiers)
        display = rendered
        isRecording = false
        window?.makeFirstResponder(nil)
        onCapture?(UInt32(event.keyCode), modifiers, rendered)
    }

    override func draw(_ dirtyRect: NSRect) {
        super.draw(dirtyRect)
        let background = isRecording ? NSColor.controlAccentColor.withAlphaComponent(0.16) : NSColor.controlBackgroundColor
        background.setFill()
        NSBezierPath(roundedRect: bounds, xRadius: 7, yRadius: 7).fill()

        (isRecording ? NSColor.controlAccentColor : NSColor.separatorColor).setStroke()
        let border = NSBezierPath(roundedRect: bounds.insetBy(dx: 0.5, dy: 0.5), xRadius: 7, yRadius: 7)
        border.lineWidth = isRecording ? 1.5 : 1
        border.stroke()

        let text = isRecording ? "Press shortcut…" : display
        let attributes: [NSAttributedString.Key: Any] = [
            .font: NSFont.monospacedSystemFont(ofSize: 13, weight: .semibold),
            .foregroundColor: isRecording ? NSColor.controlAccentColor : NSColor.labelColor,
        ]
        let size = text.size(withAttributes: attributes)
        text.draw(
            at: NSPoint(x: (bounds.width - size.width) / 2, y: (bounds.height - size.height) / 2),
            withAttributes: attributes
        )
    }
}

private struct ShortcutRecorder: NSViewRepresentable {
    @Binding var display: String
    let onCapture: (UInt32, NSEvent.ModifierFlags, String) -> Void

    func makeNSView(context: Context) -> ShortcutRecorderNSView {
        let view = ShortcutRecorderNSView()
        view.display = display
        view.onCapture = onCapture
        view.setAccessibilityRole(.button)
        view.setAccessibilityLabel("Global shortcut recorder")
        return view
    }

    func updateNSView(_ nsView: ShortcutRecorderNSView, context: Context) {
        nsView.display = display
        nsView.onCapture = onCapture
    }
}

struct GlobalShortcutSettingsSection: View {
    @AppStorage(GlobalShortcutPreference.displayKey) private var shortcutDisplay = GlobalShortcutPreference.defaultDisplay
    @ObservedObject private var registrationState = GlobalShortcutRegistrationState.shared

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Label("Global Shortcut", systemImage: "keyboard")
                .font(.system(size: 15, weight: .semibold, design: .rounded))

            Text("Click the shortcut field, then press a key together with Command, Option, Control, or Shift.")
                .font(.caption)
                .foregroundStyle(.secondary)

            HStack(spacing: 10) {
                ShortcutRecorder(display: $shortcutDisplay) { keyCode, modifiers, display in
                    shortcutDisplay = display
                    GlobalShortcutPreference.save(
                        keyCode: keyCode,
                        modifierFlags: modifiers,
                        display: display
                    )
                }

                Button("Reset") {
                    shortcutDisplay = GlobalShortcutPreference.defaultDisplay
                    GlobalShortcutPreference.reset()
                }
                .buttonStyle(.bordered)
            }

            Label(
                registrationState.message,
                systemImage: registrationState.isRegistered ? "checkmark.circle.fill" : "exclamationmark.triangle.fill"
            )
            .font(.caption)
            .foregroundStyle(registrationState.isRegistered ? .green : .orange)
        }
    }
}
