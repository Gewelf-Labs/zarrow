//! Benchmark allocator: `std.heap.smp_allocator`, except that freed blocks larger than 32 KiB
//! (the size above which smp_allocator maps fresh pages from the OS on every call) are kept and
//! handed back to the next allocation with the same length and alignment. A benchmark iteration
//! allocates the same output buffers as the one before, so the timed iterations reuse pages that
//! are already mapped and touched and measure the kernel, not first-touch page faults (a retaining
//! allocator such as jemalloc or mimalloc behaves the same way in a real application).
//!
//! Not thread-safe: the benchmarks are single-threaded.
const std = @import("std");
const Allocator = std.mem.Allocator;
const Alignment = std.mem.Alignment;

const backing = std.heap.smp_allocator;
const min_len = 32 * 1024 + 1;

const Block = struct {
    ptr: [*]u8 = undefined,
    len: usize = 0,
    alignment: Alignment = .@"1",
};

// ponytail: fixed ring, the oldest kept block is evicted; raise slot_count if one iteration frees
// more large blocks than this.
const slot_count = 64;
var slots: [slot_count]Block = @splat(.{});
var next_slot: usize = 0;

pub const allocator: Allocator = .{
    .ptr = undefined,
    .vtable = &.{ .alloc = alloc, .resize = resize, .remap = remap, .free = free },
};

fn alloc(_: *anyopaque, len: usize, alignment: Alignment, ra: usize) ?[*]u8 {
    if (len >= min_len) {
        for (&slots) |*slot| {
            if (slot.len == len and slot.alignment == alignment) {
                slot.len = 0;
                return slot.ptr;
            }
        }
    }
    return backing.rawAlloc(len, alignment, ra);
}

/// Large blocks never grow in place: growth goes through `alloc`, which can reuse a kept block.
fn resize(_: *anyopaque, memory: []u8, alignment: Alignment, new_len: usize, ra: usize) bool {
    if (new_len > memory.len and new_len >= min_len) return false;
    return backing.rawResize(memory, alignment, new_len, ra);
}

fn remap(_: *anyopaque, memory: []u8, alignment: Alignment, new_len: usize, ra: usize) ?[*]u8 {
    if (new_len > memory.len and new_len >= min_len) return null;
    return backing.rawRemap(memory, alignment, new_len, ra);
}

fn free(_: *anyopaque, memory: []u8, alignment: Alignment, ra: usize) void {
    if (memory.len < min_len) return backing.rawFree(memory, alignment, ra);
    const slot = &slots[next_slot];
    if (slot.len != 0) backing.rawFree(slot.ptr[0..slot.len], slot.alignment, ra);
    slot.* = .{ .ptr = memory.ptr, .len = memory.len, .alignment = alignment };
    next_slot = (next_slot + 1) % slot_count;
}

test "a freed large block is reused by the next allocation of the same size" {
    const a = try allocator.alloc(u8, min_len);
    const ptr = a.ptr;
    allocator.free(a);
    const b = try allocator.alloc(u8, min_len);
    defer allocator.free(b);
    try std.testing.expectEqual(ptr, b.ptr);
    const small = try allocator.alloc(u8, 16);
    defer allocator.free(small);
    var list: std.ArrayList(u8) = .empty;
    defer list.deinit(allocator);
    try list.appendNTimes(allocator, 7, 3 * min_len);
    try std.testing.expectEqual(@as(u8, 7), list.items[list.items.len - 1]);
}
