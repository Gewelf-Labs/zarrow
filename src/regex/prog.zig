//! Byte program of a parsed regular expression (Thompson NFA) and the Pike VM that runs it with
//! capture groups in O(len(text) * len(program)). Classes compile to UTF-8 byte sequences (or to
//! bytes in Latin-1 mode) with shared suffixes. A reverse program (concatenations reversed,
//! begin / end assertions swapped, no captures) lets the DFA find where a match starts.
const std = @import("std");
const parse = @import("parse.zig");

const Error = parse.Error;
const Node = parse.Node;

/// Most instructions of one program.
pub const max_insts = 100_000;
/// Most capture slots times instructions (the Pike VM keeps one slot vector per thread).
pub const max_slot_cells = 2_000_000;

pub const Op = enum(u8) { byte, set, split, save, assert, match, fail, nop };

pub const Inst = struct {
    op: Op,
    lo: u8 = 0,
    hi: u8 = 0,
    /// split: the lower-priority branch; save: the slot; assert: the kind; set: the set index.
    arg: u32 = 0,
    /// The next instruction (split: the preferred branch).
    out: u32 = 0,
};

/// Bytes accepted by a `set` instruction.
pub const ByteSet = [256]bool;

pub const Prog = struct {
    insts: []const Inst,
    sets: []const ByteSet,
    /// Entry of the anchored program.
    start: u32,
    /// Entry behind a lowest-priority loop over any byte (unanchored search; forward only).
    start_unanchored: u32,
    /// Capture slots: two per group, group 0 included (0 in a reverse program).
    nslots: u32,
};

pub fn isWordByte(b: u8) bool {
    return std.ascii.isAlphanumeric(b) or b == '_';
}

/// Whether the assertion holds at `pos` of `text` (the whole text is the context).
pub fn assertHolds(kind: parse.Assert, text: []const u8, pos: usize) bool {
    return switch (kind) {
        .begin_text => pos == 0,
        .end_text => pos == text.len,
        .begin_line => pos == 0 or text[pos - 1] == '\n',
        .end_line => pos == text.len or text[pos] == '\n',
        .word_boundary, .not_word_boundary => {
            const before = pos > 0 and isWordByte(text[pos - 1]);
            const after = pos < text.len and isWordByte(text[pos]);
            return (before != after) == (kind == .word_boundary);
        },
    };
}

/// Compile `ast` forward (with capture slots and the unanchored entry) or reversed.
pub fn compile(allocator: std.mem.Allocator, ast: parse.Ast, reverse: bool, diag: ?*parse.Diagnostic) Error!Prog {
    var c: Compiler = .{ .allocator = allocator, .ast = ast, .reverse = reverse, .diag = diag };
    defer c.suffix.deinit(allocator);
    errdefer {
        c.insts.deinit(allocator);
        c.sets.deinit(allocator);
    }
    const match_pc = try c.emit(.{ .op = .match });
    var start: u32 = undefined;
    var nslots: u32 = 0;
    if (reverse) {
        start = try c.compileNode(ast.root, match_pc, 0);
    } else {
        nslots = @intCast(2 * (ast.names.len + 1));
        const end = try c.emit(.{ .op = .save, .arg = 1, .out = match_pc });
        const body = try c.compileNode(ast.root, end, 0);
        start = try c.emit(.{ .op = .save, .arg = 0, .out = body });
    }
    const loop = try c.emit(.{ .op = .split, .out = start });
    const any = try c.emit(.{ .op = .byte, .lo = 0, .hi = 0xFF, .out = loop });
    c.insts.items[loop].arg = any;
    if (c.insts.items.len * @max(nslots, 2) > max_slot_cells) return c.fail("pattern too large: too many capture groups");
    return .{
        .insts = try c.insts.toOwnedSlice(allocator),
        .sets = try c.sets.toOwnedSlice(allocator),
        .start = start,
        .start_unanchored = loop,
        .nslots = nslots,
    };
}

pub fn deinit(prog: *Prog, allocator: std.mem.Allocator) void {
    allocator.free(prog.insts);
    allocator.free(prog.sets);
}

const Compiler = struct {
    allocator: std.mem.Allocator,
    ast: parse.Ast,
    reverse: bool,
    diag: ?*parse.Diagnostic,
    insts: std.ArrayList(Inst) = .empty,
    sets: std.ArrayList(ByteSet) = .empty,
    /// Byte instructions by (lo, hi, out): shared suffixes of UTF-8 sequences.
    suffix: std.AutoHashMapUnmanaged(u64, u32) = .empty,

    fn fail(c: *Compiler, message: []const u8) Error {
        if (c.diag) |d| d.* = .{ .message = message, .offset = 0 };
        return error.RegexTooLarge;
    }

    fn emit(c: *Compiler, inst: Inst) Error!u32 {
        if (c.insts.items.len >= max_insts) return c.fail("pattern too large: program exceeds the instruction limit");
        try c.insts.append(c.allocator, inst);
        return @intCast(c.insts.items.len - 1);
    }

    fn split(c: *Compiler, preferred: u32, other: u32) Error!u32 {
        return c.emit(.{ .op = .split, .out = preferred, .arg = other });
    }

    fn byteInst(c: *Compiler, lo: u8, hi: u8, out: u32) Error!u32 {
        const key = @as(u64, lo) | @as(u64, hi) << 8 | @as(u64, out) << 16;
        if (c.suffix.get(key)) |pc| return pc;
        const pc = try c.emit(.{ .op = .byte, .lo = lo, .hi = hi, .out = out });
        try c.suffix.put(c.allocator, key, pc);
        return pc;
    }

    fn compileNode(c: *Compiler, id: u32, next: u32, depth: u32) Error!u32 {
        std.debug.assert(depth <= 4 * parse.max_depth + 8);
        switch (c.ast.nodes[id]) {
            .empty => return next,
            .class => |s| return c.compileClass(c.ast.classRanges(s), next),
            .any_byte => return c.emit(.{ .op = .byte, .lo = 0, .hi = 0xFF, .out = next }),
            .assert => |a| {
                const kind: parse.Assert = if (!c.reverse) a else switch (a) {
                    .begin_text => .end_text,
                    .end_text => .begin_text,
                    .begin_line => .end_line,
                    .end_line => .begin_line,
                    else => a,
                };
                return c.emit(.{ .op = .assert, .arg = @intFromEnum(kind), .out = next });
            },
            .capture => |cap| {
                if (c.reverse) return c.compileNode(cap.child, next, depth + 1);
                const close = try c.emit(.{ .op = .save, .arg = 2 * cap.index + 1, .out = next });
                const body = try c.compileNode(cap.child, close, depth + 1);
                return c.emit(.{ .op = .save, .arg = 2 * cap.index, .out = body });
            },
            .concat => |s| {
                const kids = c.ast.children(s);
                var cur = next;
                if (c.reverse) {
                    for (kids) |k| cur = try c.compileNode(k, cur, depth + 1);
                } else {
                    var i = kids.len;
                    while (i > 0) {
                        i -= 1;
                        cur = try c.compileNode(kids[i], cur, depth + 1);
                    }
                }
                return cur;
            },
            .alt => |s| {
                const kids = c.ast.children(s);
                var cur = try c.compileNode(kids[kids.len - 1], next, depth + 1);
                var i = kids.len - 1;
                while (i > 0) {
                    i -= 1;
                    cur = try c.split(try c.compileNode(kids[i], next, depth + 1), cur);
                }
                return cur;
            },
            .repeat => |r| {
                var cur = next;
                if (r.max == Node.inf) {
                    // The loop: split(body, next) with the body jumping back to the split.
                    const loop = try c.emit(.{ .op = .split });
                    const body = try c.compileNode(r.child, loop, depth + 1);
                    c.insts.items[loop].out = if (r.greedy) body else next;
                    c.insts.items[loop].arg = if (r.greedy) next else body;
                    // e{n,}: n - 1 copies before e+, or e* when n is 0.
                    cur = if (r.min == 0) loop else body;
                    for (1..@max(r.min, 1)) |_| cur = try c.compileNode(r.child, cur, depth + 1);
                    return cur;
                }
                // (e(e(e)?)?)? for the optional copies, then the mandatory ones.
                for (0..r.max - r.min) |_| {
                    const body = try c.compileNode(r.child, cur, depth + 1);
                    cur = if (r.greedy) try c.split(body, next) else try c.split(next, body);
                }
                for (0..r.min) |_| cur = try c.compileNode(r.child, cur, depth + 1);
                return cur;
            },
        }
    }

    fn compileClass(c: *Compiler, ranges: []const parse.Range, next: u32) Error!u32 {
        if (ranges.len == 0) return c.emit(.{ .op = .fail });
        var single: ByteSet = @splat(false);
        var any_single = false;
        var alts: std.ArrayList(u32) = .empty;
        defer alts.deinit(c.allocator);
        if (c.ast.latin1) {
            if (ranges.len == 1) return c.byteInst(@intCast(ranges[0].lo), @intCast(ranges[0].hi), next);
            for (ranges) |r| @memset(single[r.lo .. @as(usize, r.hi) + 1], true);
            try c.sets.append(c.allocator, single);
            return c.emit(.{ .op = .set, .arg = @intCast(c.sets.items.len - 1), .out = next });
        }
        var seqs: std.ArrayList(Seq) = .empty;
        defer seqs.deinit(c.allocator);
        for (ranges) |r| try utf8Sequences(c.allocator, r.lo, r.hi, &seqs);
        var one_byte_ranges: usize = 0;
        for (seqs.items) |s| {
            if (s.len == 1) {
                @memset(single[s.lo[0] .. @as(usize, s.hi[0]) + 1], true);
                any_single = true;
                one_byte_ranges += 1;
                continue;
            }
            // Forward: first byte first; reverse: last byte first.
            var cur = next;
            var k: usize = s.len;
            while (k > 0) {
                k -= 1;
                const j = if (c.reverse) s.len - 1 - k else k;
                cur = try c.byteInst(s.lo[j], s.hi[j], cur);
            }
            try alts.append(c.allocator, cur);
        }
        if (any_single) {
            const pc = if (one_byte_ranges == 1) blk: {
                for (seqs.items) |s| if (s.len == 1) break :blk try c.byteInst(s.lo[0], s.hi[0], next);
                unreachable;
            } else blk: {
                try c.sets.append(c.allocator, single);
                break :blk try c.emit(.{ .op = .set, .arg = @intCast(c.sets.items.len - 1), .out = next });
            };
            try alts.insert(c.allocator, 0, pc);
        }
        var cur = alts.items[alts.items.len - 1];
        var i = alts.items.len - 1;
        while (i > 0) {
            i -= 1;
            cur = try c.split(alts.items[i], cur);
        }
        return cur;
    }
};

/// A UTF-8 byte sequence of ranges: byte `k` in `[lo[k], hi[k]]`.
const Seq = struct { len: u8, lo: [4]u8, hi: [4]u8 };

/// Split `[lo, hi]` (surrogates excluded) into UTF-8 byte range sequences.
fn utf8Sequences(allocator: std.mem.Allocator, lo: u21, hi: u21, out: *std.ArrayList(Seq)) Error!void {
    var stack: std.ArrayList([2]u21) = .empty;
    defer stack.deinit(allocator);
    try stack.append(allocator, .{ lo, hi });
    outer: while (stack.pop()) |r| {
        const s = r[0];
        const e = r[1];
        if (s > e) continue;
        if (s <= 0xDFFF and e >= 0xD800) {
            if (s < 0xD800) try stack.append(allocator, .{ s, 0xD7FF });
            if (e > 0xDFFF) try stack.append(allocator, .{ 0xE000, e });
            continue;
        }
        for ([_]u21{ 0x7F, 0x7FF, 0xFFFF }) |max| {
            if (s <= max and e > max) {
                try stack.append(allocator, .{ s, max });
                try stack.append(allocator, .{ max + 1, e });
                continue :outer;
            }
        }
        const n = std.unicode.utf8CodepointSequenceLength(s) catch unreachable;
        var i: u5 = 1;
        while (i < n) : (i += 1) {
            const m: u21 = (@as(u21, 1) << (6 * i)) - 1;
            if (s & ~m != e & ~m) {
                if (s & m != 0) {
                    try stack.append(allocator, .{ s, s | m });
                    try stack.append(allocator, .{ (s | m) + 1, e });
                    continue :outer;
                }
                if (e & m != m) {
                    try stack.append(allocator, .{ s, (e & ~m) - 1 });
                    try stack.append(allocator, .{ e & ~m, e });
                    continue :outer;
                }
            }
        }
        var seq: Seq = .{ .len = @intCast(n), .lo = undefined, .hi = undefined };
        _ = std.unicode.utf8Encode(s, &seq.lo) catch unreachable;
        _ = std.unicode.utf8Encode(e, &seq.hi) catch unreachable;
        try out.append(allocator, seq);
    }
}

// ------------------------------------------------------------------------------ Pike VM

/// Thread lists of the Pike VM, reused across searches.
pub const Pike = struct {
    allocator: std.mem.Allocator,
    queues: [2]Queue = .{ .{}, .{} },
    stack: std.ArrayList(Entry) = .empty,
    scratch: std.ArrayList(usize) = .empty,

    pub const none = std.math.maxInt(usize);

    const Entry = union(enum) { pc: u32, restore: struct { slot: u32, value: usize } };

    const Queue = struct {
        sparse: []u32 = &.{},
        dense: []u32 = &.{},
        caps: []usize = &.{},
        len: u32 = 0,

        fn contains(q: *const Queue, pc: u32) bool {
            const k = q.sparse[pc];
            return k < q.len and q.dense[k] == pc;
        }

        fn insert(q: *Queue, pc: u32) u32 {
            const k = q.len;
            q.sparse[pc] = k;
            q.dense[k] = pc;
            q.len += 1;
            return k;
        }
    };

    pub fn deinit(self: *Pike) void {
        for (&self.queues) |*q| {
            self.allocator.free(q.sparse);
            self.allocator.free(q.dense);
            self.allocator.free(q.caps);
        }
        self.stack.deinit(self.allocator);
        self.scratch.deinit(self.allocator);
    }

    fn prepare(self: *Pike, n: usize, nslots: usize) error{OutOfMemory}!void {
        for (&self.queues) |*q| {
            if (q.sparse.len < n) {
                self.allocator.free(q.sparse);
                self.allocator.free(q.dense);
                q.sparse = &.{};
                q.dense = &.{};
                q.sparse = try self.allocator.alloc(u32, n);
                q.dense = try self.allocator.alloc(u32, n);
                // Stale sparse entries are harmless (checked against dense), but keep them defined.
                @memset(q.sparse, 0);
            }
            if (q.caps.len < n * nslots) {
                self.allocator.free(q.caps);
                q.caps = &.{};
                q.caps = try self.allocator.alloc(usize, n * nslots);
            }
            q.len = 0;
        }
        try self.scratch.resize(self.allocator, nslots);
    }

    /// Leftmost-first search from `start` (only at `start` when `anchored`); fills `slots`
    /// (pairs of positions, `none` for an unset slot; at most `prog.nslots`) and returns whether
    /// a match was found.
    pub fn search(self: *Pike, prog: *const Prog, text: []const u8, start: usize, anchored: bool, slots: []usize) error{OutOfMemory}!bool {
        const nslots = slots.len;
        try self.prepare(prog.insts.len, nslots);
        var clist = &self.queues[0];
        var nlist = &self.queues[1];
        const tmp = self.scratch.items;
        var matched = false;
        var pos = start;
        while (true) : (pos += 1) {
            if (!matched and (!anchored or pos == start)) {
                @memset(tmp, none);
                try self.addThread(prog, clist, prog.start, text, pos, tmp);
            }
            if (clist.len == 0) break;
            const c: ?u8 = if (pos < text.len) text[pos] else null;
            for (clist.dense[0..clist.len], 0..) |pc, k| {
                const inst = prog.insts[pc];
                const caps = clist.caps[k * nslots ..][0..nslots];
                switch (inst.op) {
                    .match => {
                        @memcpy(slots, caps);
                        matched = true;
                        break; // lower-priority threads are cut (leftmost-first)
                    },
                    .byte, .set => {
                        const b = c orelse continue;
                        const ok = if (inst.op == .byte) b >= inst.lo and b <= inst.hi else prog.sets[inst.arg][b];
                        if (!ok) continue;
                        @memcpy(tmp, caps);
                        try self.addThread(prog, nlist, inst.out, text, pos + 1, tmp);
                    },
                    else => {},
                }
            }
            std.mem.swap(*Queue, &clist, &nlist);
            nlist.len = 0;
            if (pos >= text.len) break;
        }
        return matched;
    }

    /// Follow the empty transitions from `pc0` in priority order, adding the threads that
    /// consume a byte or match to `q` with the slots of the path.
    fn addThread(self: *Pike, prog: *const Prog, q: *Queue, pc0: u32, text: []const u8, pos: usize, tmp: []usize) error{OutOfMemory}!void {
        const nslots = tmp.len;
        try self.stack.append(self.allocator, .{ .pc = pc0 });
        while (self.stack.pop()) |e| {
            const pc = switch (e) {
                .restore => |r| {
                    tmp[r.slot] = r.value;
                    continue;
                },
                .pc => |pc| pc,
            };
            if (q.contains(pc)) continue;
            const k = q.insert(pc);
            const inst = prog.insts[pc];
            switch (inst.op) {
                .byte, .set, .match => @memcpy(q.caps[k * nslots ..][0..nslots], tmp),
                .split => {
                    try self.stack.append(self.allocator, .{ .pc = inst.arg });
                    try self.stack.append(self.allocator, .{ .pc = inst.out });
                },
                .save => {
                    if (inst.arg < nslots) {
                        try self.stack.append(self.allocator, .{ .restore = .{ .slot = inst.arg, .value = tmp[inst.arg] } });
                        tmp[inst.arg] = pos;
                    }
                    try self.stack.append(self.allocator, .{ .pc = inst.out });
                },
                .assert => if (assertHolds(@enumFromInt(inst.arg), text, pos)) try self.stack.append(self.allocator, .{ .pc = inst.out }),
                .nop => try self.stack.append(self.allocator, .{ .pc = inst.out }),
                .fail => {},
            }
        }
    }
};
