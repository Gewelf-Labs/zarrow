//! Lazy DFA over a byte program: states are ordered sets of program instructions (byte
//! consumers, matches and assertions that wait for the next byte), built on demand and cached
//! per (state, byte class). Leftmost-first mode cuts the threads behind a match (the forward
//! search, which finds where the leftmost-first match ends); longest mode keeps them (the reverse
//! search, which finds where it starts). The cache is bounded: when it fills, it is cleared and
//! the search reports `GiveUp` so the caller can run the Pike VM instead.
const std = @import("std");
const parse = @import("parse.zig");
const prog_mod = @import("prog.zig");

const Prog = prog_mod.Prog;

pub const Error = error{ OutOfMemory, GiveUp };

/// Most cached states and bytes of state sets before the cache is cleared.
pub const max_states = 4096;
pub const max_state_bytes = 4 * 1024 * 1024;

/// What precedes a position: the text start, a newline, a word byte or another byte.
const Prev = enum(u8) { begin, newline, word, other };

fn prevOf(b: u8) Prev {
    return if (b == '\n') .newline else if (prog_mod.isWordByte(b)) .word else .other;
}

const Next = union(enum) { unknown, end, byte: u8 };

const unknown_trans = std.math.maxInt(u32);
const dead: u32 = 0;

pub const Dfa = struct {
    allocator: std.mem.Allocator,
    prog: *const Prog,
    longest: bool,
    classes: [256]u8 = undefined,
    reps: [256]u8 = undefined,
    nclasses: u32 = 0,
    /// State `i`: its key (one flag byte, then instruction indices as little-endian u32).
    states: std.ArrayList([]const u8) = .empty,
    map: std.StringHashMapUnmanaged(u32) = .empty,
    trans: std.ArrayList(u32) = .empty,
    /// 0 unknown, 1 no match at the end of the text, 2 a match.
    end_match: std.ArrayList(u8) = .empty,
    /// Start states by [anchored][prev].
    starts: [2][4]u32 = @splat(@splat(unknown_trans)),
    arena: std.heap.ArenaAllocator,
    state_bytes: usize = 0,
    // Scratch of the closure.
    sparse: []u32 = &.{},
    dense: []u32 = &.{},
    set_len: u32 = 0,
    stack: std.ArrayList(u32) = .empty,
    list: std.ArrayList(u32) = .empty,
    seeds: std.ArrayList(u32) = .empty,
    key: std.ArrayList(u8) = .empty,

    pub fn init(allocator: std.mem.Allocator, prog: *const Prog, longest: bool) error{OutOfMemory}!Dfa {
        var self: Dfa = .{ .allocator = allocator, .prog = prog, .longest = longest, .arena = .init(allocator) };
        errdefer self.deinit();
        self.computeClasses();
        self.sparse = try allocator.alloc(u32, prog.insts.len);
        @memset(self.sparse, 0);
        self.dense = try allocator.alloc(u32, prog.insts.len);
        try self.reset();
        return self;
    }

    pub fn deinit(self: *Dfa) void {
        const a = self.allocator;
        self.states.deinit(a);
        self.map.deinit(a);
        self.trans.deinit(a);
        self.end_match.deinit(a);
        self.arena.deinit();
        a.free(self.sparse);
        a.free(self.dense);
        self.stack.deinit(a);
        self.list.deinit(a);
        self.seeds.deinit(a);
        self.key.deinit(a);
    }

    /// Byte classes: bytes that every instruction and assertion treats alike.
    fn computeClasses(self: *Dfa) void {
        var boundary: [257]bool = @splat(false);
        boundary[0] = true;
        boundary['\n'] = true;
        boundary['\n' + 1] = true;
        for (1..256) |b| {
            if (prog_mod.isWordByte(@intCast(b)) != prog_mod.isWordByte(@intCast(b - 1))) boundary[b] = true;
        }
        for (self.prog.insts) |inst| switch (inst.op) {
            .byte => {
                boundary[inst.lo] = true;
                boundary[@as(usize, inst.hi) + 1] = true;
            },
            .set => {
                const s = self.prog.sets[inst.arg];
                for (1..256) |b| {
                    if (s[b] != s[b - 1]) boundary[b] = true;
                }
            },
            else => {},
        };
        var id: u32 = 0;
        for (0..256) |b| {
            if (b > 0 and boundary[b]) id += 1;
            if (b == 0 or boundary[b]) self.reps[id] = @intCast(b);
            self.classes[b] = @intCast(id);
        }
        self.nclasses = id + 1;
    }

    /// Clear the cache; state 0 is the dead state.
    fn reset(self: *Dfa) error{OutOfMemory}!void {
        self.states.clearRetainingCapacity();
        self.map.clearRetainingCapacity();
        self.trans.clearRetainingCapacity();
        self.end_match.clearRetainingCapacity();
        _ = self.arena.reset(.retain_capacity);
        self.state_bytes = 0;
        self.starts = @splat(@splat(unknown_trans));
        self.key.clearRetainingCapacity();
        try self.key.append(self.allocator, 0);
        _ = self.intern() catch |err| return switch (err) {
            error.OutOfMemory => error.OutOfMemory,
            error.GiveUp => unreachable,
        };
    }

    fn setClear(self: *Dfa) void {
        self.set_len = 0;
    }

    fn setInsert(self: *Dfa, pc: u32) bool {
        const k = self.sparse[pc];
        if (k < self.set_len and self.dense[k] == pc) return false;
        self.sparse[pc] = self.set_len;
        self.dense[self.set_len] = pc;
        self.set_len += 1;
        return true;
    }

    fn eval(kind: parse.Assert, prev: Prev, next: Next) ?bool {
        return switch (kind) {
            .begin_text => prev == .begin,
            .begin_line => prev == .begin or prev == .newline,
            .end_text => switch (next) {
                .unknown => null,
                .end => true,
                .byte => false,
            },
            .end_line => switch (next) {
                .unknown => null,
                .end => true,
                .byte => |b| b == '\n',
            },
            .word_boundary, .not_word_boundary => switch (next) {
                .unknown => null,
                else => {
                    const after = next == .byte and prog_mod.isWordByte(next.byte);
                    return ((prev == .word) != after) == (kind == .word_boundary);
                },
            },
        };
    }

    /// Append to `self.list`, in priority order, the instructions reached from `seeds` through
    /// empty transitions: byte consumers, matches and assertions that need an unknown next byte.
    fn closure(self: *Dfa, seeds: []const u32, prev: Prev, next: Next) error{OutOfMemory}!void {
        self.list.clearRetainingCapacity();
        self.setClear();
        const insts = self.prog.insts;
        for (seeds) |seed| {
            try self.stack.append(self.allocator, seed);
            while (self.stack.pop()) |pc| {
                if (!self.setInsert(pc)) continue;
                const inst = insts[pc];
                switch (inst.op) {
                    .byte, .set, .match => try self.list.append(self.allocator, pc),
                    .split => {
                        try self.stack.append(self.allocator, inst.arg);
                        try self.stack.append(self.allocator, inst.out);
                    },
                    .save, .nop => try self.stack.append(self.allocator, inst.out),
                    .fail => {},
                    .assert => if (eval(@enumFromInt(inst.arg), prev, next)) |ok| {
                        if (ok) try self.stack.append(self.allocator, inst.out);
                    } else try self.list.append(self.allocator, pc),
                }
            }
        }
    }

    /// The state of `self.list` after `prev` (kept only when an assertion waits).
    fn internList(self: *Dfa, prev: Prev) Error!u32 {
        var waits = false;
        for (self.list.items) |pc| {
            if (self.prog.insts[pc].op == .assert) waits = true;
        }
        self.key.clearRetainingCapacity();
        try self.key.append(self.allocator, if (waits) @intFromEnum(prev) else 0);
        for (self.list.items) |pc| {
            var b: [4]u8 = undefined;
            std.mem.writeInt(u32, &b, pc, .little);
            try self.key.appendSlice(self.allocator, &b);
        }
        return self.intern();
    }

    fn intern(self: *Dfa) Error!u32 {
        if (self.map.get(self.key.items)) |id| return id;
        if (self.states.items.len >= max_states or self.state_bytes + self.key.items.len > max_state_bytes) return error.GiveUp;
        const key = try self.arena.allocator().dupe(u8, self.key.items);
        const id: u32 = @intCast(self.states.items.len);
        try self.states.append(self.allocator, key);
        try self.map.put(self.allocator, key, id);
        try self.trans.appendNTimes(self.allocator, unknown_trans, self.nclasses);
        try self.end_match.append(self.allocator, 0);
        self.state_bytes += key.len;
        return id;
    }

    fn stateInsts(self: *Dfa, id: u32) struct { prev: Prev, pcs: []const u8 } {
        const key = self.states.items[id];
        return .{ .prev = @enumFromInt(key[0]), .pcs = key[1..] };
    }

    /// Copy the instructions of state `id` to `self.seeds`.
    fn loadSeeds(self: *Dfa, id: u32) error{OutOfMemory}!Prev {
        const s = self.stateInsts(id);
        self.seeds.clearRetainingCapacity();
        var i: usize = 0;
        while (i < s.pcs.len) : (i += 4) try self.seeds.append(self.allocator, std.mem.readInt(u32, s.pcs[i..][0..4], .little));
        return s.prev;
    }

    fn startState(self: *Dfa, anchored: bool, prev: Prev) Error!u32 {
        const slot = &self.starts[@intFromBool(anchored)][@intFromEnum(prev)];
        if (slot.* != unknown_trans) return slot.*;
        const pc = if (anchored) self.prog.start else self.prog.start_unanchored;
        try self.closure(&.{pc}, prev, .unknown);
        slot.* = try self.internList(prev);
        return slot.*;
    }

    /// Transition of `id` on byte class `cls`: the next state shifted left once, the low bit set
    /// when a match ends before the byte.
    fn computeTrans(self: *Dfa, id: u32, cls: u32) Error!u32 {
        const b = self.reps[cls];
        const prev = try self.loadSeeds(id);
        try self.closure(self.seeds.items, prev, .{ .byte = b });
        var matched = false;
        self.seeds.clearRetainingCapacity();
        for (self.list.items) |pc| {
            const inst = self.prog.insts[pc];
            switch (inst.op) {
                .match => {
                    matched = true;
                    if (!self.longest) break;
                },
                .byte => if (b >= inst.lo and b <= inst.hi) try self.seeds.append(self.allocator, inst.out),
                .set => if (self.prog.sets[inst.arg][b]) try self.seeds.append(self.allocator, inst.out),
                else => {},
            }
        }
        const p = prevOf(b);
        try self.closure(self.seeds.items, p, .unknown);
        const next = try self.internList(p);
        // The next state as its row offset in `trans` (no multiply in the search loops).
        const t = (next * self.nclasses) << 1 | @intFromBool(matched);
        self.trans.items[id * self.nclasses + cls] = t;
        return t;
    }

    fn endMatches(self: *Dfa, id: u32) error{OutOfMemory}!bool {
        if (self.end_match.items[id] != 0) return self.end_match.items[id] == 2;
        const prev = try self.loadSeeds(id);
        try self.closure(self.seeds.items, prev, .end);
        var m = false;
        for (self.list.items) |pc| {
            if (self.prog.insts[pc].op == .match) m = true;
        }
        self.end_match.items[id] = if (m) 2 else 1;
        return m;
    }

    inline fn step(self: *Dfa, s: u32, b: u8) Error!u32 {
        const cls = self.classes[b];
        const t = self.trans.items[s * self.nclasses + cls];
        return if (t != unknown_trans) t else self.computeTrans(s, cls);
    }

    /// Clear the cache when a search filled it, so the next search starts empty.
    fn giveUp(self: *Dfa, err: Error) Error {
        if (err == error.GiveUp) try self.reset();
        return err;
    }

    /// End of the leftmost-first match at or after `start` (at `start` when `anchored`; the
    /// first match end seen when `earliest`), null when none.
    pub fn searchForward(self: *Dfa, text: []const u8, start: usize, anchored: bool, earliest: bool) Error!?usize {
        return self.forward(text, start, anchored, earliest) catch |err| self.giveUp(err);
    }

    fn forward(self: *Dfa, text: []const u8, start: usize, anchored: bool, earliest: bool) Error!?usize {
        const nc = self.nclasses;
        var row = nc * try self.startState(anchored, if (start == 0) .begin else prevOf(text[start - 1]));
        var last: ?usize = null;
        const classes = &self.classes;
        var trans = self.trans.items;
        var i = start;
        while (i < text.len) : (i += 1) {
            const cls = classes[text[i]];
            var t = trans[row + cls];
            if (t == unknown_trans) {
                t = try self.computeTrans(row / nc, cls);
                trans = self.trans.items;
            }
            if (t & 1 != 0) {
                last = i;
                if (earliest) return i;
            }
            row = t >> 1;
            if (row == dead) return last;
        }
        if (try self.endMatches(row / nc)) last = text.len;
        return last;
    }

    /// Start of the longest match of the reverse program ending at `end`, scanning back no
    /// further than `min_start`; null when none.
    pub fn searchReverse(self: *Dfa, text: []const u8, end: usize, min_start: usize) Error!?usize {
        return self.reverse(text, end, min_start) catch |err| self.giveUp(err);
    }

    fn reverse(self: *Dfa, text: []const u8, end: usize, min_start: usize) Error!?usize {
        const nc = self.nclasses;
        var row = nc * try self.startState(true, if (end == text.len) .begin else prevOf(text[end]));
        var last: ?usize = null;
        const classes = &self.classes;
        var trans = self.trans.items;
        var i = end;
        while (i > min_start) : (i -= 1) {
            const cls = classes[text[i - 1]];
            var t = trans[row + cls];
            if (t == unknown_trans) {
                t = try self.computeTrans(row / nc, cls);
                trans = self.trans.items;
            }
            if (t & 1 != 0) last = i;
            row = t >> 1;
            if (row == dead) return last;
        }
        if (min_start == 0) {
            if (try self.endMatches(row / nc)) last = 0;
        } else if ((try self.step(row / nc, text[min_start - 1])) & 1 != 0) last = min_start;
        return last;
    }
};
