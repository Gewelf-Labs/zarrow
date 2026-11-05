const std = @import("std");
const core = @import("core.zig");
const signature_mod = @import("signature.zig");

const FunctionKind = core.FunctionKind;
const DataType = core.DataType;
const Datum = core.Datum;
const KernelError = core.KernelError;
const Options = core.Options;

pub const TypeCheckFn = signature_mod.TypeCheckFn;
pub const OptionsCheckFn = signature_mod.OptionsCheckFn;
pub const ResultTypeFn = signature_mod.ResultTypeFn;
pub const KernelSignature = signature_mod.KernelSignature;
pub const KernelExecFn = *const fn (ctx: *ExecContext, args: []const Datum, options: Options) KernelError!Datum;
pub const AggregateInitFn = *const fn (ctx: *ExecContext, options: Options) KernelError!*anyopaque;
pub const AggregateUpdateFn = *const fn (ctx: *ExecContext, state: *anyopaque, args: []const Datum, options: Options) KernelError!void;
pub const AggregateMergeFn = *const fn (ctx: *ExecContext, state: *anyopaque, other_state: *anyopaque, options: Options) KernelError!void;
pub const AggregateFinalizeFn = *const fn (ctx: *ExecContext, state: *anyopaque, options: Options) KernelError!Datum;
pub const AggregateDeinitFn = *const fn (ctx: *ExecContext, state: *anyopaque) void;

/// Stateful aggregate lifecycle callbacks used for incremental/grouped aggregation.
pub const AggregateLifecycle = struct {
    init: AggregateInitFn,
    update: AggregateUpdateFn,
    merge: AggregateMergeFn,
    finalize: AggregateFinalizeFn,
    deinit: AggregateDeinitFn,
};

pub const Kernel = struct {
    signature: KernelSignature,
    exec: KernelExecFn,
    aggregate_lifecycle: ?AggregateLifecycle = null,

    pub fn supportsAggregateLifecycle(self: Kernel) bool {
        return self.aggregate_lifecycle != null;
    }
};

/// Live aggregate state handle created from an aggregate kernel lifecycle.
pub const AggregateSession = struct {
    ctx: *ExecContext,
    kernel: *const Kernel,
    lifecycle: AggregateLifecycle,
    options: Options,
    state: *anyopaque,

    pub fn update(self: *AggregateSession, args: []const Datum) KernelError!void {
        if (!self.kernel.signature.matches(args)) return error.NoMatchingKernel;
        if (!self.kernel.signature.matchesOptions(self.options)) return error.InvalidOptions;
        return self.lifecycle.update(self.ctx, self.state, args, self.options);
    }

    pub fn merge(self: *AggregateSession, other: *AggregateSession) KernelError!void {
        if (self.kernel != other.kernel) return error.AggregateStateMismatch;
        return self.lifecycle.merge(self.ctx, self.state, other.state, self.options);
    }

    pub fn finalize(self: *AggregateSession) KernelError!Datum {
        return self.lifecycle.finalize(self.ctx, self.state, self.options);
    }

    pub fn deinit(self: *AggregateSession) void {
        self.lifecycle.deinit(self.ctx, self.state);
        self.* = undefined;
    }
};

const function_kind_count = std.enums.values(FunctionKind).len;
const FunctionIndexMap = std.StringHashMap(usize);

/// Implicit casts a function applies when no kernel matches the argument types (Arrow's
/// `Function::DispatchBest`).
pub const ImplicitCast = enum {
    /// Argument types must match a kernel exactly.
    none,
    /// Numeric arguments of different types are cast to their common numeric type first (see
    /// `commonNumericType`).
    numeric,
};

pub const Function = struct {
    allocator: std.mem.Allocator,
    name: []u8,
    kind: FunctionKind,
    kernels: std.ArrayList(Kernel),
    implicit_cast: ImplicitCast = .none,

    pub fn kernelCount(self: *const Function) usize {
        return self.kernels.items.len;
    }

    pub fn kernelsSlice(self: *const Function) []const Kernel {
        return self.kernels.items;
    }

    fn deinit(self: *Function) void {
        self.kernels.deinit(self.allocator);
        self.allocator.free(self.name);
    }
};

pub const FunctionRegistry = struct {
    allocator: std.mem.Allocator,
    functions: std.ArrayList(Function),
    function_index_by_kind: [function_kind_count]FunctionIndexMap,

    pub fn init(allocator: std.mem.Allocator) FunctionRegistry {
        return .{
            .allocator = allocator,
            .functions = .empty,
            .function_index_by_kind = initFunctionIndexMaps(allocator),
        };
    }

    pub fn deinit(self: *FunctionRegistry) void {
        for (&self.function_index_by_kind) |*index_map| {
            index_map.deinit();
        }
        for (self.functions.items) |*entry| {
            entry.deinit();
        }
        self.functions.deinit(self.allocator);
    }

    pub fn registerKernel(
        self: *FunctionRegistry,
        name: []const u8,
        kind: FunctionKind,
        kernel: Kernel,
    ) KernelError!void {
        if (!kernel.signature.isValidArityModel()) return error.InvalidInput;
        if (self.findFunctionIndex(name, kind)) |idx| {
            try self.functions.items[idx].kernels.append(self.allocator, kernel);
            return;
        }

        var entry = Function{
            .allocator = self.allocator,
            .name = try self.allocator.dupe(u8, name),
            .kind = kind,
            .kernels = .empty,
        };
        errdefer self.allocator.free(entry.name);
        try entry.kernels.append(self.allocator, kernel);
        try self.functions.append(self.allocator, entry);
        // Ownership of entry.name was value-copied into functions.items above.
        // Clear local so that the outer errdefer cannot double-free it if put() fails.
        entry.name = &.{};
        errdefer {
            var popped = self.functions.pop().?;
            popped.deinit();
        }
        const new_idx = self.functions.items.len - 1;
        try self.getIndexMap(kind).put(self.functions.items[new_idx].name, new_idx);
    }

    pub fn registerScalarKernel(self: *FunctionRegistry, name: []const u8, kernel: Kernel) KernelError!void {
        return self.registerKernel(name, .scalar, kernel);
    }

    pub fn registerVectorKernel(self: *FunctionRegistry, name: []const u8, kernel: Kernel) KernelError!void {
        return self.registerKernel(name, .vector, kernel);
    }

    pub fn registerAggregateKernel(self: *FunctionRegistry, name: []const u8, kernel: Kernel) KernelError!void {
        return self.registerKernel(name, .aggregate, kernel);
    }

    /// Set the implicit casts of a registered function.
    pub fn setImplicitCast(self: *FunctionRegistry, name: []const u8, kind: FunctionKind, mode: ImplicitCast) KernelError!void {
        const idx = self.findFunctionIndex(name, kind) orelse return error.FunctionNotFound;
        self.functions.items[idx].implicit_cast = mode;
    }

    pub fn containsFunction(self: *const FunctionRegistry, name: []const u8, kind: FunctionKind) bool {
        return self.findFunction(name, kind) != null;
    }

    pub fn functionCount(self: *const FunctionRegistry) usize {
        return self.functions.items.len;
    }

    pub fn functionAt(self: *const FunctionRegistry, index: usize) ?*const Function {
        if (index >= self.functions.items.len) return null;
        return &self.functions.items[index];
    }

    pub fn kernelCount(self: *const FunctionRegistry, name: []const u8, kind: FunctionKind) usize {
        const function = self.findFunction(name, kind) orelse return 0;
        return function.kernels.items.len;
    }

    pub fn findFunction(self: *const FunctionRegistry, name: []const u8, kind: FunctionKind) ?*const Function {
        const idx = self.findFunctionIndex(name, kind) orelse return null;
        return &self.functions.items[idx];
    }

    pub fn resolveKernel(
        self: *const FunctionRegistry,
        name: []const u8,
        kind: FunctionKind,
        args: []const Datum,
        options: Options,
    ) KernelError!*const Kernel {
        const function = self.findFunction(name, kind) orelse return error.FunctionNotFound;
        if (function.kernels.items.len == 0) return error.NoMatchingKernel;

        var saw_matching_arity = false;
        var saw_matching_type = false;
        var best_kernel: ?*const Kernel = null;
        var best_specificity: u8 = 0;
        for (function.kernels.items) |*kernel| {
            if (!kernel.signature.matchesArity(args.len)) continue;
            saw_matching_arity = true;
            if (!kernel.signature.matches(args)) continue;
            saw_matching_type = true;
            if (!kernel.signature.matchesOptions(options)) continue;

            const specificity = kernel.signature.aritySpecificityRank();
            if (best_kernel == null or specificity > best_specificity) {
                best_kernel = kernel;
                best_specificity = specificity;
            }
        }
        if (best_kernel) |kernel| return kernel;
        if (!saw_matching_arity) return error.InvalidArity;
        if (saw_matching_type) return error.InvalidOptions;
        return error.NoMatchingKernel;
    }

    /// Explain why kernel resolution would fail for the given call site.
    pub fn explainResolveKernelFailure(
        self: *const FunctionRegistry,
        name: []const u8,
        kind: FunctionKind,
        args: []const Datum,
        options: Options,
    ) []const u8 {
        const function = self.findFunction(name, kind) orelse return "function not found";
        if (function.kernels.items.len == 0) return "function has no registered kernels";

        var saw_matching_arity = false;
        var saw_matching_type = false;
        var best_specificity: u8 = 0;
        var saw_matching_kernel = false;
        var has_exact = false;
        var has_range = false;
        var has_at_least = false;
        var any_min_gt_arg = false;
        var any_range_excludes_arg = false;
        for (function.kernels.items) |*kernel| {
            switch (kernel.signature.arityModel()) {
                .exact => has_exact = true,
                .range => has_range = true,
                .at_least => has_at_least = true,
            }
            if (args.len < kernel.signature.arity) any_min_gt_arg = true;
            if (kernel.signature.arityModel() == .range and !kernel.signature.matchesArity(args.len)) {
                any_range_excludes_arg = true;
            }

            if (!kernel.signature.matchesArity(args.len)) continue;
            saw_matching_arity = true;
            if (!kernel.signature.matches(args)) continue;
            saw_matching_type = true;
            if (!kernel.signature.matchesOptions(options)) continue;
            const specificity = kernel.signature.aritySpecificityRank();
            if (!saw_matching_kernel or specificity > best_specificity) {
                best_specificity = specificity;
                saw_matching_kernel = true;
            }
        }
        if (saw_matching_kernel) return "kernel resolution should succeed";
        if (!saw_matching_arity) {
            if (has_at_least and any_min_gt_arg) return "no kernel matched minimum arity";
            if (has_range and any_range_excludes_arg) return "no kernel matched arity range";
            if (has_exact) return "no kernel matched exact arity";
            return "no kernel matched arity";
        }
        if (saw_matching_type) return "kernel matched args but options were invalid";
        return "no kernel matched argument types";
    }

    pub fn resolveResultType(
        self: *const FunctionRegistry,
        name: []const u8,
        kind: FunctionKind,
        args: []const Datum,
        options: Options,
    ) KernelError!DataType {
        const kernel = self.resolveKernel(name, kind, args, options) catch |err| {
            const target = self.implicitTarget(err, name, kind, args) orelse return err;
            // The result type of the call on arguments cast to the common type.
            var stack: [4]Datum = undefined;
            if (args.len > stack.len) return err;
            for (stack[0..args.len]) |*a| a.* = .{ .scalar = core.Scalar.init(target, .null) };
            return self.resolveResultType(name, kind, stack[0..args.len], options);
        };
        return kernel.signature.inferResultType(args, options);
    }

    /// The common numeric type to cast `args` to when resolution failed with `NoMatchingKernel`
    /// and the function casts numeric arguments implicitly; null otherwise.
    fn implicitTarget(self: *const FunctionRegistry, err: KernelError, name: []const u8, kind: FunctionKind, args: []const Datum) ?DataType {
        if (err != error.NoMatchingKernel) return null;
        const function = self.findFunction(name, kind) orelse return null;
        if (function.implicit_cast != .numeric) return null;
        return commonNumericType(args);
    }

    /// Explain why result-type inference would fail for the given call site.
    pub fn explainResolveResultTypeFailure(
        self: *const FunctionRegistry,
        name: []const u8,
        kind: FunctionKind,
        args: []const Datum,
        options: Options,
    ) []const u8 {
        const function = self.findFunction(name, kind) orelse return "cannot infer result type: function not found";
        if (function.kernels.items.len == 0) return "cannot infer result type: function has no kernels";
        for (function.kernels.items) |*kernel| {
            if (!kernel.signature.accepts(args, options)) continue;
            return kernel.signature.explainInferResultTypeFailure(args, options);
        }
        return self.explainResolveKernelFailure(name, kind, args, options);
    }

    pub fn invoke(
        self: *const FunctionRegistry,
        ctx: *ExecContext,
        name: []const u8,
        kind: FunctionKind,
        args: []const Datum,
        options: Options,
    ) KernelError!Datum {
        const kernel = self.resolveKernel(name, kind, args, options) catch |err| {
            const target = self.implicitTarget(err, name, kind, args) orelse return err;
            return self.invokeCast(ctx, name, kind, args, options, target);
        };
        _ = try kernel.signature.inferResultType(args, options);
        return kernel.exec(ctx, args, options);
    }

    /// Cast every argument that is not of type `target` with the `cast` function (safe unless
    /// `ExecConfig.safe_cast` is false, like Arrow's implicit casts), then invoke `name`.
    fn invokeCast(self: *const FunctionRegistry, ctx: *ExecContext, name: []const u8, kind: FunctionKind, args: []const Datum, options: Options, target: DataType) KernelError!Datum {
        const cast_args = ctx.allocator.alloc(Datum, args.len) catch return error.OutOfMemory;
        defer ctx.allocator.free(cast_args);
        const owned = ctx.allocator.alloc(bool, args.len) catch return error.OutOfMemory;
        defer ctx.allocator.free(owned);
        @memset(owned, false);
        defer for (cast_args, owned) |*a, o| if (o) a.release();
        for (args, cast_args, owned) |arg, *c, *o| {
            if (arg.dataType().eql(target)) {
                c.* = arg;
                continue;
            }
            c.* = try self.invoke(ctx, "cast", .scalar, &.{arg}, .{ .cast = .{ .to_type = target } });
            o.* = true;
        }
        const kernel = try self.resolveKernel(name, kind, cast_args, options);
        _ = try kernel.signature.inferResultType(cast_args, options);
        return kernel.exec(ctx, cast_args, options);
    }

    pub fn invokeScalar(
        self: *const FunctionRegistry,
        ctx: *ExecContext,
        name: []const u8,
        args: []const Datum,
        options: Options,
    ) KernelError!Datum {
        return self.invoke(ctx, name, .scalar, args, options);
    }

    pub fn invokeVector(
        self: *const FunctionRegistry,
        ctx: *ExecContext,
        name: []const u8,
        args: []const Datum,
        options: Options,
    ) KernelError!Datum {
        return self.invoke(ctx, name, .vector, args, options);
    }

    pub fn invokeAggregate(
        self: *const FunctionRegistry,
        ctx: *ExecContext,
        name: []const u8,
        args: []const Datum,
        options: Options,
    ) KernelError!Datum {
        return self.invoke(ctx, name, .aggregate, args, options);
    }

    /// Create a stateful aggregate session from an aggregate (or `hash_*` grouped aggregate)
    /// kernel lifecycle.
    pub fn beginAggregate(
        self: *const FunctionRegistry,
        ctx: *ExecContext,
        name: []const u8,
        prototype_args: []const Datum,
        options: Options,
    ) KernelError!AggregateSession {
        const kind: FunctionKind = if (self.findFunction(name, .aggregate) == null and self.findFunction(name, .hash_aggregate) != null) .hash_aggregate else .aggregate;
        const kernel = try self.resolveKernel(name, kind, prototype_args, options);
        const lifecycle = kernel.aggregate_lifecycle orelse return error.MissingLifecycle;
        const state = try lifecycle.init(ctx, options);
        return .{
            .ctx = ctx,
            .kernel = kernel,
            .lifecycle = lifecycle,
            .options = options,
            .state = state,
        };
    }

    fn findFunctionIndex(self: *const FunctionRegistry, name: []const u8, kind: FunctionKind) ?usize {
        const idx = self.getIndexMapConst(kind).get(name) orelse return null;
        if (idx >= self.functions.items.len) return null;
        return idx;
    }

    fn getIndexMap(self: *FunctionRegistry, kind: FunctionKind) *FunctionIndexMap {
        return &self.function_index_by_kind[@intFromEnum(kind)];
    }

    fn getIndexMapConst(self: *const FunctionRegistry, kind: FunctionKind) *const FunctionIndexMap {
        return &self.function_index_by_kind[@intFromEnum(kind)];
    }
};

/// Arrow's `CommonNumeric` over the argument types: null unless every argument is int8..int64,
/// uint8..uint64, float32 or float64 and two of them differ. Any float64 gives float64, else any
/// float32 gives float32; integers of one signedness give the widest; mixed signedness gives the
/// signed type wider than every unsigned one (uint64 with a signed type gives int64, as in Arrow).
pub fn commonNumericType(args: []const Datum) ?DataType {
    if (args.len < 2) return null;
    var differ = false;
    var float_bits: u8 = 0;
    var signed_bits: u8 = 0;
    var unsigned_bits: u8 = 0;
    for (args) |arg| {
        const dt = arg.dataType();
        if (!dt.eql(args[0].dataType())) differ = true;
        switch (dt) {
            .float => float_bits = @max(float_bits, 32),
            .double => float_bits = 64,
            .int8 => signed_bits = @max(signed_bits, 8),
            .int16 => signed_bits = @max(signed_bits, 16),
            .int32 => signed_bits = @max(signed_bits, 32),
            .int64 => signed_bits = 64,
            .uint8 => unsigned_bits = @max(unsigned_bits, 8),
            .uint16 => unsigned_bits = @max(unsigned_bits, 16),
            .uint32 => unsigned_bits = @max(unsigned_bits, 32),
            .uint64 => unsigned_bits = 64,
            else => return null,
        }
    }
    if (!differ) return null;
    if (float_bits == 64) return .{ .double = {} };
    if (float_bits == 32) return .{ .float = {} };
    if (signed_bits == 0) return switch (unsigned_bits) {
        8 => .{ .uint8 = {} },
        16 => .{ .uint16 = {} },
        32 => .{ .uint32 = {} },
        else => .{ .uint64 = {} },
    };
    if (signed_bits <= unsigned_bits) signed_bits = if (unsigned_bits >= 32) 64 else unsigned_bits * 2;
    return switch (signed_bits) {
        8 => .{ .int8 = {} },
        16 => .{ .int16 = {} },
        32 => .{ .int32 = {} },
        else => .{ .int64 = {} },
    };
}

fn initFunctionIndexMaps(allocator: std.mem.Allocator) [function_kind_count]FunctionIndexMap {
    var maps: [function_kind_count]FunctionIndexMap = undefined;
    inline for (0..function_kind_count) |i| {
        maps[i] = FunctionIndexMap.init(allocator);
    }
    return maps;
}

/// Overflow policy for arithmetic kernels.
pub const OverflowMode = enum {
    checked,
    wrapping,
    saturating,
};

/// Execution configuration shared by all kernel invocations in a context.
pub const ExecConfig = struct {
    /// Safe cast mode for cast kernels (fail on lossy casts when true).
    safe_cast: bool = true,
    /// Overflow policy for arithmetic kernels.
    overflow_mode: OverflowMode = .checked,
    /// Preferred thread count for vector/aggregate execution.
    threads: usize = 1,
    /// Optional arena-like allocator used for temporary/borrowed scalar payloads.
    arena_allocator: ?std.mem.Allocator = null,
};

pub const ExecContext = struct {
    allocator: std.mem.Allocator,
    registry: *const FunctionRegistry,
    config: ExecConfig,

    pub fn init(allocator: std.mem.Allocator, registry: *const FunctionRegistry) ExecContext {
        return initWithConfig(allocator, registry, .{});
    }

    pub fn initWithConfig(allocator: std.mem.Allocator, registry: *const FunctionRegistry, config: ExecConfig) ExecContext {
        var normalized = config;
        if (normalized.threads == 0) normalized.threads = 1;
        return .{
            .allocator = allocator,
            .registry = registry,
            .config = normalized,
        };
    }

    pub fn tempAllocator(self: *const ExecContext) std.mem.Allocator {
        return self.config.arena_allocator orelse self.allocator;
    }

    /// Duplicate UTF-8 bytes into the context temp allocator for scalar string payloads.
    pub fn dupScalarString(self: *const ExecContext, value: []const u8) KernelError![]const u8 {
        return self.tempAllocator().dupe(u8, value) catch error.OutOfMemory;
    }

    /// Duplicate raw bytes into the context temp allocator for scalar binary payloads.
    pub fn dupScalarBinary(self: *const ExecContext, value: []const u8) KernelError![]const u8 {
        return self.tempAllocator().dupe(u8, value) catch error.OutOfMemory;
    }

    pub fn safeCastEnabled(self: *const ExecContext) bool {
        return self.config.safe_cast;
    }

    pub fn overflowMode(self: *const ExecContext) OverflowMode {
        return self.config.overflow_mode;
    }

    pub fn threads(self: *const ExecContext) usize {
        return self.config.threads;
    }

    pub fn invoke(
        self: *ExecContext,
        name: []const u8,
        kind: FunctionKind,
        args: []const Datum,
        options: Options,
    ) KernelError!Datum {
        return self.registry.invoke(self, name, kind, args, options);
    }

    pub fn invokeScalar(
        self: *ExecContext,
        name: []const u8,
        args: []const Datum,
        options: Options,
    ) KernelError!Datum {
        return self.registry.invokeScalar(self, name, args, options);
    }

    pub fn invokeVector(
        self: *ExecContext,
        name: []const u8,
        args: []const Datum,
        options: Options,
    ) KernelError!Datum {
        return self.registry.invokeVector(self, name, args, options);
    }

    pub fn invokeAggregate(
        self: *ExecContext,
        name: []const u8,
        args: []const Datum,
        options: Options,
    ) KernelError!Datum {
        return self.registry.invokeAggregate(self, name, args, options);
    }

    pub fn beginAggregate(
        self: *ExecContext,
        name: []const u8,
        prototype_args: []const Datum,
        options: Options,
    ) KernelError!AggregateSession {
        return self.registry.beginAggregate(self, name, prototype_args, options);
    }
};
