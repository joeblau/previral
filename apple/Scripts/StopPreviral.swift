import AppKit
import Foundation

// Match the app's bundle ID so older copies launched from other build folders
// are stopped too, without matching unrelated processes by name or command line.
let bundleID = "com.joeblau.previral"
let instances = NSRunningApplication.runningApplications(withBundleIdentifier: bundleID)

func waitForExit(seconds: TimeInterval) -> [NSRunningApplication] {
    let deadline = Date().addingTimeInterval(seconds)
    while Date() < deadline {
        let remaining = instances.filter { !$0.isTerminated }
        if remaining.isEmpty { return [] }
        RunLoop.current.run(until: Date().addingTimeInterval(0.1))
    }
    return instances.filter { !$0.isTerminated }
}

if !instances.isEmpty {
    print("Stopping \(instances.count) running previral instance(s)…")
    for app in instances { _ = app.terminate() }
    let remaining = waitForExit(seconds: 5)
    if !remaining.isEmpty {
        print("Force-stopping \(remaining.count) unresponsive instance(s)…")
        for app in remaining { _ = app.forceTerminate() }
    }
    let survivors = waitForExit(seconds: 3)
    if !survivors.isEmpty {
        let pids = survivors.map { String($0.processIdentifier) }.joined(separator: ", ")
        FileHandle.standardError.write(Data("Cannot stop previral process(es) \(pids); build cancelled.\n".utf8))
        exit(1)
    }
}
