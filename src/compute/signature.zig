const std = @import("std");
const core = @import("core.zig");

const DataType = core.DataType;
const Datum = core.Datum;
const KernelError = core.KernelError;
const Options = core.Options;

pub const TypeCheckFn = *const fn (args: []const Datum) bool;
pub const OptionsCheckFn = *const fn (options: Options) bool;
pub const ResultTypeFn = *const fn (args: []const Datum, options: Options) KernelError!DataType;
/// Result type callback whose type needs storage (a struct over the input's type, such as
/// min_max of a decimal or value_counts): nested types are allocated in `allocator` (an arena;
/// they may also borrow from `args`).
pub const ResultTypeAllocFn = *const fn (allocator: std.mem.Allocator, args: []const Datum, options: Options) KernelError!DataType;

/// Kernel signature metadata used for dispatch and type inference.
pub const KernelSignature = struct {
    /// Required argument count (exact arity or minimum arity for variadic kernels).
    arity: usize,
    /// Whether the signature accepts variadic arguments.
    variadic: bool = false,
    /// Optional maximum arity for bounded variadic signatures.
    max_arity: ?usize = null,
    /// Optional argument validator for logical type matching.
    type_check: ?TypeCheckFn = null,
    /// Optional options validator for type-safe options enforcement.
    options_check: ?OptionsCheckFn = null,
    /// Optional result type inference callback.
    result_type_fn: ?ResultTypeFn = null,
    /// Result type callback of a kernel whose result type needs storage; `inferResultType`
    /// then answers `NotImplemented` and `inferResultTypeAlloc` calls it.
    result_type_alloc_fn: ?ResultTypeAllocFn = null,

    pub fn any(arity: usize) KernelSignature {
        return .{
            .arity = arity,
            .variadic = false,
            .max_arity = null,
            .type_check = null,
            .options_check = null,
            .result_type_fn = null,
        };
    }

    pub fn atLeast(min_arity: usize) KernelSignature {
        return .{
            .arity = min_arity,
            .variadic = true,
            .max_arity = null,
            .type_check = null,
            .options_check = null,
            .result_type_fn = null,
        };
    }

    pub fn range(min_arity: usize, max_arity: usize) KernelSignature {
        std.debug.assert(max_arity >= min_arity);
        return .{
            .arity = min_arity,
            .variadic = true,
            .max_arity = max_arity,
            .type_check = null,
            .options_check = null,
            .result_type_fn = null,
        };
    }

    pub fn unary(type_check: ?TypeCheckFn) KernelSignature {
        return .{
            .arity = 1,
            .variadic = false,
            .max_arity = null,
            .type_check = type_check,
            .options_check = null,
            .result_type_fn = null,
        };
    }

    pub fn binary(type_check: ?TypeCheckFn) KernelSignature {
        return .{
            .arity = 2,
            .variadic = false,
            .max_arity = null,
            .type_check = type_check,
            .options_check = null,
            .result_type_fn = null,
        };
    }

    pub fn unaryWithResult(type_check: ?TypeCheckFn, result_type_fn: ResultTypeFn) KernelSignature {
        return .{
            .arity = 1,
            .variadic = false,
            .max_arity = null,
            .type_check = type_check,
            .options_check = null,
            .result_type_fn = result_type_fn,
        };
    }

    pub fn binaryWithResult(type_check: ?TypeCheckFn, result_type_fn: ResultTypeFn) KernelSignature {
        return .{
            .arity = 2,
            .variadic = false,
            .max_arity = null,
            .type_check = type_check,
            .options_check = null,
            .result_type_fn = result_type_fn,
        };
    }

    pub fn variadicWithResult(min_arity: usize, type_check: ?TypeCheckFn, result_type_fn: ResultTypeFn) KernelSignature {
        return .{
            .arity = min_arity,
            .variadic = true,
            .max_arity = null,
            .type_check = type_check,
            .options_check = null,
            .result_type_fn = result_type_fn,
        };
    }

    pub const ArityModel = enum {
        exact,
        at_least,
        range,
    };

    pub fn arityModel(self: KernelSignature) ArityModel {
        if (!self.variadic) return .exact;
        if (self.max_arity != null) return .range;
        return .at_least;
    }

    pub fn aritySpecificityRank(self: KernelSignature) u8 {
        return switch (self.arityModel()) {
            .exact => 3,
            .range => 2,
            .at_least => 1,
        };
    }

    fn hasValidArityModel(self: KernelSignature) bool {
        if (!self.variadic) return self.max_arity == null;
        if (self.max_arity) |max| return max >= self.arity;
        return true;
    }

    pub fn isValidArityModel(self: KernelSignature) bool {
        return self.hasValidArityModel();
    }

    pub fn matchesArity(self: KernelSignature, arg_count: usize) bool {
        std.debug.assert(self.hasValidArityModel());
        if (!self.hasValidArityModel()) return false;
        return switch (self.arityModel()) {
            .exact => arg_count == self.arity,
            .at_least => arg_count >= self.arity,
            .range => arg_count >= self.arity and arg_count <= self.max_arity.?,
        };
    }

    pub fn matches(self: KernelSignature, args: []const Datum) bool {
        if (!self.matchesArity(args.len)) return false;
        if (self.type_check) |check| return check(args);
        return true;
    }

    pub fn matchesOptions(self: KernelSignature, options: Options) bool {
        if (self.options_check) |check| return check(options);
        return true;
    }

    pub fn accepts(self: KernelSignature, args: []const Datum, options: Options) bool {
        return self.matches(args) and self.matchesOptions(options);
    }

    /// Human-readable mismatch reason for diagnostics and debugging.
    pub fn explainMismatch(self: KernelSignature, args: []const Datum, options: Options) []const u8 {
        if (!self.matchesArity(args.len)) {
            return switch (self.arityModel()) {
                .exact => "arity mismatch: argument count does not match exact kernel arity",
                .at_least => "arity mismatch: argument count is below minimum kernel arity",
                .range => "arity mismatch: argument count is outside kernel arity range",
            };
        }
        if (self.type_check) |check| {
            if (!check(args)) return "type mismatch: arguments did not satisfy kernel type_check";
        }
        if (self.options_check) |check| {
            if (!check(options)) return "options mismatch: options did not satisfy kernel options_check";
        }
        return "signature accepted";
    }

    pub fn inferResultType(self: KernelSignature, args: []const Datum, options: Options) KernelError!DataType {
        if (!self.matchesArity(args.len)) return error.InvalidArity;
        if (self.type_check) |check| {
            if (!check(args)) return error.NoMatchingKernel;
        }
        if (self.options_check) |check| {
            if (!check(options)) return error.InvalidOptions;
        }
        if (self.result_type_fn) |infer| {
            return infer(args, options);
        }
        if (self.result_type_alloc_fn != null) return error.NotImplemented;
        if (args.len == 0) return error.InvalidInput;
        return args[0].dataType();
    }

    /// `inferResultType`, with `allocator` (an arena) for a result type that needs storage.
    pub fn inferResultTypeAlloc(self: KernelSignature, allocator: std.mem.Allocator, args: []const Datum, options: Options) KernelError!DataType {
        const infer = self.result_type_alloc_fn orelse return self.inferResultType(args, options);
        if (!self.matchesArity(args.len)) return error.InvalidArity;
        if (self.type_check) |check| {
            if (!check(args)) return error.NoMatchingKernel;
        }
        if (self.options_check) |check| {
            if (!check(options)) return error.InvalidOptions;
        }
        return infer(allocator, args, options);
    }

    /// Check that the call has a result type (the type itself is dropped): a type that needs
    /// storage is built in a stack buffer.
    pub fn checkResultType(self: KernelSignature, args: []const Datum, options: Options) KernelError!void {
        var buf: [2048]u8 = undefined;
        var fba = std.heap.FixedBufferAllocator.init(&buf);
        _ = try self.inferResultTypeAlloc(fba.allocator(), args, options);
    }

    /// Human-readable reason for result-type inference failure.
    pub fn explainInferResultTypeFailure(self: KernelSignature, args: []const Datum, options: Options) []const u8 {
        if (!self.matchesArity(args.len)) return "cannot infer result type: invalid arity";
        if (self.type_check) |check| {
            if (!check(args)) return "cannot infer result type: argument type_check failed";
        }
        if (self.options_check) |check| {
            if (!check(options)) return "cannot infer result type: options_check failed";
        }
        if (self.result_type_fn == null and self.result_type_alloc_fn != null) return "cannot infer result type without an allocator: use inferResultTypeAlloc";
        if (self.result_type_fn == null and args.len == 0) return "cannot infer result type: no arguments and no result_type_fn";
        return "result type inference should succeed";
    }
};
