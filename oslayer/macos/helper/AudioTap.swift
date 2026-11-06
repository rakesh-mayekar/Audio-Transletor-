// olt-audio-tap: streams system playback audio to stdout as interleaved float32 LE,
// 48 kHz stereo, using ScreenCaptureKit (macOS 13+). Diagnostics go to stderr.
// Exit codes: 0 = clean stop, 77 = Screen Recording permission denied, 69 = no display/stream lost.
import AVFoundation
import CoreMedia
import Foundation
import ScreenCaptureKit

let exitPermission: Int32 = 77
let exitUnavailable: Int32 = 69

func log(_ s: String) { FileHandle.standardError.write((s + "\n").data(using: .utf8)!) }

final class Tap: NSObject, SCStreamOutput, SCStreamDelegate {
    private let out = FileHandle.standardOutput
    private var stream: SCStream?
    private let queue = DispatchQueue(label: "olt.audio", qos: .userInteractive)

    func start() async {
        do {
            let content = try await SCShareableContent.excludingDesktopWindows(false, onScreenWindowsOnly: false)
            guard let display = content.displays.first else {
                log("no display available"); exit(exitUnavailable)
            }
            let filter = SCContentFilter(display: display, excludingWindows: [])
            let cfg = SCStreamConfiguration()
            cfg.capturesAudio = true
            cfg.excludesCurrentProcessAudio = true
            cfg.sampleRate = 48000
            cfg.channelCount = 2
            cfg.width = 2; cfg.height = 2                    // video is unused; keep it minimal
            cfg.minimumFrameInterval = CMTime(value: 1, timescale: 1)
            let s = SCStream(filter: filter, configuration: cfg, delegate: self)
            try s.addStreamOutput(self, type: .audio, sampleHandlerQueue: queue)
            try await s.startCapture()
            stream = s
        } catch {
            let ns = error as NSError
            log("start failed: \(ns.domain) \(ns.code) \(ns.localizedDescription)")
            // -3801 = userDeclined (SCStreamError); permission problems surface as this code
            exit(ns.code == -3801 ? exitPermission : exitUnavailable)
        }
    }

    func stop() async {
        try? await stream?.stopCapture()
    }

    func stream(_ stream: SCStream, didStopWithError error: Error) {
        log("stream stopped: \(error.localizedDescription)"); exit(exitUnavailable)
    }

    func stream(_ stream: SCStream, didOutputSampleBuffer sb: CMSampleBuffer, of type: SCStreamOutputType) {
        guard type == .audio, sb.isValid, CMSampleBufferGetNumSamples(sb) > 0 else { return }
        var block: CMBlockBuffer?
        let size = MemoryLayout<AudioBufferList>.size + 7 * MemoryLayout<AudioBuffer>.size
        let raw = UnsafeMutableRawPointer.allocate(byteCount: size, alignment: 16)
        defer { raw.deallocate() }
        let list = raw.bindMemory(to: AudioBufferList.self, capacity: 1)
        let st = CMSampleBufferGetAudioBufferListWithRetainedBlockBuffer(
            sb, bufferListSizeNeededOut: nil, bufferListOut: list, bufferListSize: size,
            blockBufferAllocator: nil, blockBufferMemoryAllocator: nil, flags: 0, blockBufferOut: &block)
        guard st == noErr else { return }
        let buffers = UnsafeMutableAudioBufferListPointer(list)
        let frames = Int(CMSampleBufferGetNumSamples(sb))
        var data = Data()
        if buffers.count == 1, buffers[0].mNumberChannels == 2, let p = buffers[0].mData {
            data = Data(bytes: p, count: Int(buffers[0].mDataByteSize))        // already interleaved
        } else if buffers.count >= 2 {                                          // planar L/R
            guard let l = buffers[0].mData?.assumingMemoryBound(to: Float.self),
                  let r = buffers[1].mData?.assumingMemoryBound(to: Float.self) else { return }
            var inter = [Float](repeating: 0, count: frames * 2)
            for i in 0..<frames { inter[2 * i] = l[i]; inter[2 * i + 1] = r[i] }
            data = inter.withUnsafeBufferPointer { Data(buffer: $0) }
        } else { return }
        do { try out.write(contentsOf: data) } catch { exit(0) }               // reader went away
    }
}

setvbuf(stdout, nil, _IONBF, 0)
var signalSources: [DispatchSourceSignal] = []
let tap = Tap()
signal(SIGPIPE, SIG_IGN)
for sig in [SIGTERM, SIGINT] {
    signal(sig, SIG_IGN)
    let src = DispatchSource.makeSignalSource(signal: sig, queue: .main)
    src.setEventHandler { Task { await tap.stop(); exit(0) } }
    src.resume()
    signalSources.append(src)
}
Task { await tap.start() }
RunLoop.main.run()
