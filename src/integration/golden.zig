//! The golden-file check of one apache/arrow-testing integration case
//! (`data/arrow-ipc-stream/integration/<version>/<case>.{json.gz,arrow_file,stream}`), shared by
//! tools/integration_runner.zig and the in-tree test over src/ipc/testdata. See
//! docs/integration-suite.md.
const std = @import("std");
const json = @import("json.zig");
const json_writer = @import("json_writer.zig");
const compare = @import("compare.zig");
const ipc_stream_reader = @import("../ipc/stream_reader.zig");
const ipc_stream_writer = @import("../ipc/stream_writer.zig");
const ipc_file_reader = @import("../ipc/file_reader.zig");
const ipc_file_writer = @import("../ipc/file_writer.zig");

const Allocator = std.mem.Allocator;
const Document = json.Document;
const RecordBatch = json.RecordBatch;

pub const Expect = enum {
    /// Every step must pass.
    accept,
    /// A case zarrow's IPC validation rejects on purpose (docs/integration-suite.md lists them):
    /// the JSON must read and both IPC files must be rejected.
    reject_known,
};

pub const Outcome = enum { pass, rejected_known, fail };

/// What a check found: the outcome and, for a rejection or a failure, where and why.
pub const Result = struct {
    outcome: Outcome = .pass,
    buf: [1200]u8 = undefined,
    len: usize = 0,

    pub fn message(self: *const Result) []const u8 {
        return self.buf[0..self.len];
    }

    fn set(self: *Result, outcome: Outcome, comptime fmt: []const u8, args: anytype) void {
        self.outcome = outcome;
        var w: std.Io.Writer = .fixed(&self.buf);
        w.print(fmt, args) catch {};
        self.len = w.end;
    }
};

/// Decompresses a gzip member (the `.json.gz` files) with std.compress.flate.
pub fn gunzip(allocator: Allocator, gz: []const u8) ![]u8 {
    var in: std.Io.Reader = .fixed(gz);
    var window: [std.compress.flate.max_window_len]u8 = undefined;
    var d: std.compress.flate.Decompress = .init(&in, .gzip, &window);
    return d.reader.allocRemaining(allocator, .unlimited);
}

/// Runs the golden steps on one case:
///   (a) the JSON, the IPC file and the IPC stream give equal schemas and batches (every column
///       passes validateFull),
///   (b) JSON -> zarrow stream writer -> stream reader equals the JSON,
///   (c) JSON -> zarrow file writer -> file reader equals the JSON,
///   (d) IPC file -> stream -> file (zarrow writers and readers) equals the JSON,
///   (e) the JSON writer's output of the JSON batches reads back equal.
pub fn check(allocator: Allocator, json_text: []const u8, arrow_file: []const u8, stream: []const u8, expect: Expect) Result {
    var r: Result = .{};
    var doc = json.read(allocator, json_text) catch |e| {
        r.set(.fail, "JSON read: {s}", .{@errorName(e)});
        return r;
    };
    defer doc.deinit();

    if (expect != .accept) {
        var file_doc = readFile(allocator, arrow_file) catch |fe| {
            var stream_doc = readStream(allocator, stream) catch |se| {
                r.set(.rejected_known, "file: {s}, stream: {s}", .{ @errorName(fe), @errorName(se) });
                return r;
            };
            stream_doc.deinit();
            r.set(.fail, "the stream was accepted, expected a rejection", .{});
            return r;
        };
        file_doc.deinit();
        r.set(.fail, "the file was accepted, expected a rejection", .{});
        return r;
    }

    // (a)
    var file_doc = readFile(allocator, arrow_file) catch |e| {
        r.set(.fail, "(a) IPC file read: {s}", .{@errorName(e)});
        return r;
    };
    defer file_doc.deinit();
    if (!same(&r, "(a) JSON vs IPC file", &doc, &file_doc)) return r;
    var stream_doc = readStream(allocator, stream) catch |e| {
        r.set(.fail, "(a) IPC stream read: {s}", .{@errorName(e)});
        return r;
    };
    defer stream_doc.deinit();
    if (!same(&r, "(a) JSON vs IPC stream", &doc, &stream_doc)) return r;

    // (b)
    if (!roundTrip(allocator, &r, "(b) JSON -> stream", &doc, &doc, .stream)) return r;
    // (c)
    if (!roundTrip(allocator, &r, "(c) JSON -> file", &doc, &doc, .file)) return r;
    // (d)
    {
        const bytes = writeIpc(allocator, &file_doc, .stream) catch |e| {
            r.set(.fail, "(d) IPC file -> stream write: {s}", .{@errorName(e)});
            return r;
        };
        defer allocator.free(bytes);
        var mid = readStream(allocator, bytes) catch |e| {
            r.set(.fail, "(d) IPC file -> stream read: {s}", .{@errorName(e)});
            return r;
        };
        defer mid.deinit();
        if (!roundTrip(allocator, &r, "(d) IPC file -> stream -> file", &mid, &doc, .file)) return r;
    }
    // (e)
    {
        var out: std.Io.Writer.Allocating = .init(allocator);
        defer out.deinit();
        json_writer.write(allocator, &out.writer, doc.schema().*, doc.batches) catch |e| {
            r.set(.fail, "(e) JSON write: {s}", .{@errorName(e)});
            return r;
        };
        var again = json.read(allocator, out.written()) catch |e| {
            r.set(.fail, "(e) JSON re-read: {s}", .{@errorName(e)});
            return r;
        };
        defer again.deinit();
        if (!same(&r, "(e) JSON vs JSON written by zarrow", &doc, &again)) return r;
    }
    return r;
}

const Container = enum { stream, file };

/// Writes `src` with a zarrow writer, reads it back and compares it with `expected`.
fn roundTrip(allocator: Allocator, r: *Result, step: []const u8, src: *const Document, expected: *const Document, container: Container) bool {
    const bytes = writeIpc(allocator, src, container) catch |e| {
        r.set(.fail, "{s} write: {s}", .{ step, @errorName(e) });
        return false;
    };
    defer allocator.free(bytes);
    var back = (switch (container) {
        .stream => readStream(allocator, bytes),
        .file => readFile(allocator, bytes),
    }) catch |e| {
        r.set(.fail, "{s} read: {s}", .{ step, @errorName(e) });
        return false;
    };
    defer back.deinit();
    return same(r, step, expected, &back);
}

/// Equal schemas, batch counts and batches.
pub fn same(r: *Result, step: []const u8, a: *const Document, b: *const Document) bool {
    var why: compare.Mismatch = .{};
    if (!compare.schemasEqual(a.schema().*, b.schema().*, &why)) {
        r.set(.fail, "{s}: {s}", .{ step, why.message() });
        return false;
    }
    if (a.batches.len != b.batches.len) {
        r.set(.fail, "{s}: {d} vs {d} batches", .{ step, a.batches.len, b.batches.len });
        return false;
    }
    for (a.batches, b.batches, 0..) |*x, *y, i| {
        if (!compare.batchesEqual(x, y, &why)) {
            r.set(.fail, "{s}: batch {d}: {s}", .{ step, i, why.message() });
            return false;
        }
    }
    return true;
}

fn validate(doc: *const Document) !void {
    for (doc.batches) |*b| for (b.columns) |c| try c.data().validateFull();
}

/// Reads an IPC file into a document; every column passes validateFull.
pub fn readFile(allocator: Allocator, bytes: []const u8) !Document {
    var reader = ipc_file_reader.FileReader.init(allocator, bytes);
    defer reader.deinit();
    _ = try reader.readSchema();
    const n = try reader.recordBatchCount();
    const batches = try allocator.alloc(RecordBatch, n);
    var filled: usize = 0;
    errdefer {
        for (batches[0..filled]) |*b| b.deinit();
        allocator.free(batches);
    }
    while (filled < n) : (filled += 1) batches[filled] = try reader.readRecordBatchAt(filled);
    const doc: Document = .{ .allocator = allocator, .schema_ref = reader.schema_ref.?.retain(), .batches = batches };
    errdefer {
        var d = doc;
        d.schema_ref.release();
    }
    try validate(&doc);
    return doc;
}

/// Reads an IPC stream into a document; every column passes validateFull.
pub fn readStream(allocator: Allocator, bytes: []const u8) !Document {
    var in: std.Io.Reader = .fixed(bytes);
    var reader = ipc_stream_reader.StreamReader.init(allocator, &in);
    defer reader.deinit();
    _ = try reader.readSchema();
    var batches: std.ArrayList(RecordBatch) = .empty;
    errdefer {
        for (batches.items) |*b| b.deinit();
        batches.deinit(allocator);
    }
    while (true) {
        try batches.ensureUnusedCapacity(allocator, 1);
        const batch = try reader.nextRecordBatch() orelse break;
        batches.appendAssumeCapacity(batch);
    }
    const owned = try batches.toOwnedSlice(allocator);
    var doc: Document = .{ .allocator = allocator, .schema_ref = reader.schema_ref.?.retain(), .batches = owned };
    errdefer doc.deinit();
    try validate(&doc);
    return doc;
}

/// The document written as an IPC stream or file by zarrow's writers.
pub fn writeIpc(allocator: Allocator, doc: *const Document, container: Container) ![]u8 {
    var out: std.Io.Writer.Allocating = .init(allocator);
    defer out.deinit();
    switch (container) {
        .stream => {
            var w = ipc_stream_writer.StreamWriter.init(allocator, &out.writer);
            defer w.deinit();
            try w.writeSchema(doc.schema().*);
            for (doc.batches) |b| try w.writeRecordBatch(b);
            try w.writeEnd();
        },
        .file => {
            var w = try ipc_file_writer.FileWriter.init(allocator, &out.writer);
            defer w.deinit();
            try w.writeSchema(doc.schema().*);
            for (doc.batches) |b| try w.writeRecordBatch(b);
            try w.writeEnd();
        },
    }
    return out.toOwnedSlice();
}
