//! Kernel families shipped in this repository and `registerAll`, the equivalent of
//! `arrow::compute::Initialize()`. An application may instead call the `register` function of
//! only the families it uses.
const core = @import("core.zig");

pub const arithmetic = @import("arithmetic.zig");
pub const comparison = @import("comparison.zig");
pub const logical = @import("logical.zig");
pub const aggregate_basic = @import("aggregate_basic.zig");
pub const is_null = @import("is_null.zig");
pub const cast = @import("cast.zig");
pub const string = @import("string.zig");
pub const temporal = @import("temporal.zig");
pub const string_transform = @import("string_transform.zig");
pub const temporal_round = @import("temporal_round.zig");
pub const set_lookup = @import("set_lookup.zig");
pub const hash_aggregate = @import("hash_aggregate.zig");
pub const decimal = @import("decimal.zig");
pub const math = @import("math.zig");
pub const round = @import("round.zig");
pub const temporal_arith = @import("temporal_arith.zig");
pub const associative = @import("associative.zig");
pub const selection = @import("selection.zig");
pub const aggregate_stats = @import("aggregate_stats.zig");
pub const sort = @import("sort.zig");
pub const tdigest = @import("tdigest.zig");
pub const pivot = @import("pivot.zig");
pub const cumulative = @import("cumulative.zig");
pub const string_regex = @import("string_regex.zig");
pub const temporal_text = @import("temporal_text.zig");
pub const temporal_zone = @import("temporal_zone.zig");
pub const tz = @import("tz.zig");
pub const string_normalize = @import("string_normalize.zig");

/// Register every kernel family of this repository in `registry`.
pub fn registerAll(registry: *core.FunctionRegistry) core.KernelError!void {
    try arithmetic.register(registry);
    try comparison.register(registry);
    try logical.register(registry);
    try aggregate_basic.register(registry);
    try is_null.register(registry);
    try cast.register(registry);
    try string.register(registry);
    try temporal.register(registry);
    try string_transform.register(registry);
    try temporal_round.register(registry);
    try set_lookup.register(registry);
    try hash_aggregate.register(registry);
    try decimal.register(registry);
    try math.register(registry);
    try round.register(registry);
    try temporal_arith.register(registry);
    try associative.register(registry);
    try selection.register(registry);
    try aggregate_stats.register(registry);
    try sort.register(registry);
    try tdigest.register(registry);
    try pivot.register(registry);
    try cumulative.register(registry);
    try string_regex.register(registry);
    try temporal_text.register(registry);
    try temporal_zone.register(registry);
    try string_normalize.register(registry);
}
