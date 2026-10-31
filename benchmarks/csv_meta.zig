const std = @import("std");

pub const CsvMeta = struct {
    git_sha: []const u8,
    timestamp: i64,
    owns_git_sha: bool,

    pub fn deinit(self: CsvMeta, allocator: std.mem.Allocator) void {
        if (self.owns_git_sha) allocator.free(self.git_sha);
    }
};

pub fn resolve(init: std.process.Init, allocator: std.mem.Allocator) !CsvMeta {
    const timestamp = std.Io.Timestamp.now(init.io, .real).toSeconds();
    const git_sha = init.minimal.environ.getAlloc(allocator, "ZARROW_GIT_SHA") catch |err| switch (err) {
        error.EnvironmentVariableMissing => null,
        else => return err,
    };

    if (git_sha) |sha| {
        return .{ .git_sha = sha, .timestamp = timestamp, .owns_git_sha = true };
    }

    return .{ .git_sha = "unknown", .timestamp = timestamp, .owns_git_sha = false };
}
