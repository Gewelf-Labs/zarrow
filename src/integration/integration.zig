//! The Arrow integration suite (docs/integration-suite.md): the integration JSON format, logical
//! comparison of batches and the golden-file check over apache/arrow-testing.
pub const json = @import("json.zig");
pub const compare = @import("compare.zig");
pub const golden = @import("golden.zig");
