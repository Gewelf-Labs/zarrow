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
}
