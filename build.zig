const std = @import("std");
const builtin = @import("builtin");

// Enforce supported Zig version at compile time.
// master supports Zig 0.16.x and 0.17.x.
comptime {
    const ver = builtin.zig_version;
    if (ver.major != 0 or (ver.minor != 16 and ver.minor != 17)) {
        @compileError("zarrow master requires Zig 0.16.x or 0.17.x. For Zig 0.15.x use branch release/zig-0.15");
    }
}

const is_zig_0_17 = builtin.zig_version.minor >= 17;

/// Opens a build-root subdirectory for iteration (0.17: `b.root`, 0.16: `b.build_root.handle`).
fn openBuildDir(b: *std.Build, sub_path: []const u8) std.Io.Dir.OpenError!std.Io.Dir {
    const options: std.Io.Dir.OpenOptions = .{ .iterate = true };
    if (is_zig_0_17) return b.root.openDir(b.graph.io, sub_path, options);
    return b.build_root.handle.openDir(b.graph.io, sub_path, options);
}

/// Declares a configure-time directory dependency (0.17 only; 0.16 has no configure cache).
fn dependOnDir(b: *std.Build, sub_path: []const u8) void {
    if (is_zig_0_17) b.dependOnDirectoryContents(b.path(sub_path));
}

/// Forwards `zig build <step> -- args` to the run step.
fn addPassthru(run: *std.Build.Step.Run) void {
    if (is_zig_0_17) return run.addPassthruArgs();
    if (run.step.owner.args) |args| run.addArgs(args);
}

fn configureIpcCompression(b: *std.Build, step: *std.Build.Step.Compile, deps_check: *std.Build.Step) void {
    step.step.dependOn(deps_check);

    step.root_module.addIncludePath(b.path("vendor/zstd"));
    step.root_module.addIncludePath(b.path("vendor/lz4"));

    step.root_module.addCSourceFile(.{
        .file = b.path("vendor/zstd/zstd.c"),
        .flags = &.{"-DZSTD_LEGACY_SUPPORT=0"},
    });
    step.root_module.addCSourceFile(.{
        .file = b.path("vendor/lz4/lz4_all.c"),
        .flags = &.{},
    });

    step.root_module.link_libc = true;
}

// Configure the zarrow package as a reusable module plus a dedicated test step.
pub fn build(b: *std.Build) !void {
    // Allow downstream consumers and tests to select their own target and optimization mode.
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Expose zarrow as a reusable Zig module for downstream dependencies.
    const zarrow_module = b.addModule("zarrow", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
    });

    // Core-only module for downstream libraries that do not require IPC/FFI.
    _ = b.addModule("zarrow-core", .{
        .root_source_file = b.path("src/core.zig"),
        .target = target,
        .optimize = optimize,
    });

    // C ABI shared library exposing Arrow C Data/C Stream helpers for downstream C/C++/Rust.
    const zarrow_c_lib = b.addLibrary(.{
        .linkage = .dynamic,
        .name = "zarrow_c",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/c_api.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    zarrow_c_lib.root_module.link_libc = true;
    zarrow_c_lib.installHeader(b.path("include/zarrow_c_api.h"), "zarrow_c_api.h");
    b.installArtifact(zarrow_c_lib);
    const c_api_lib_step = b.step("c-api-lib", "Build and install zarrow C ABI shared library");
    c_api_lib_step.dependOn(b.getInstallStep());

    // C ABI smoke lanes (docs/interop-c-abi.md): the programs of the CI jobs of the same names,
    // built against this build's zarrow_c and run by tools/interop/c_abi_smoke.py (the Arrow C++
    // headers and library default to pyarrow's; the Rust lane needs cargo).
    const arrow_include = b.option([]const u8, "arrow-include", "Arrow C++ include directory for interop-c-abi-cpp-smoke (default: pyarrow's)");
    const arrow_lib = b.option([]const u8, "arrow-lib", "Arrow C++ library directory for interop-c-abi-cpp-smoke (default: pyarrow's)");
    const python = if (b.graph.host.result.os.tag == .windows) "python" else "python3";
    inline for (.{ "cpp", "rs" }, .{ "interop-c-abi-cpp-smoke", "interop-c-abi-rs-smoke" }, .{ "Arrow C++", "arrow-rs" }) |lane, name, other| {
        const smoke = b.addSystemCommand(&.{ python, "-P" });
        smoke.addFileArg(b.path("tools/interop/c_abi_smoke.py"));
        smoke.addArgs(&.{ lane, "--zig", b.graph.zig_exe });
        smoke.addPrefixedFileArg("--lib=", zarrow_c_lib.getEmittedBin());
        if (target.result.os.tag == .windows) smoke.addPrefixedFileArg("--implib=", zarrow_c_lib.getEmittedImplib());
        if (arrow_include) |dir| smoke.addArg(b.fmt("--arrow-include={s}", .{dir}));
        if (arrow_lib) |dir| smoke.addArg(b.fmt("--arrow-lib={s}", .{dir}));
        smoke.has_side_effects = true;
        b.step(name, "Run the C ABI smoke program zarrow_c <-> " ++ other).dependOn(&smoke.step);
    }

    const fbz_module = b.createModule(.{
        .root_source_file = b.path("src/fbs_runtime/lib.zig"),
    });

    const compression_deps_check_exe = b.addExecutable(.{
        .name = "check-ipc-compression-deps",
        .root_module = b.createModule(.{
            .root_source_file = b.path("tools/check_ipc_compression_deps.zig"),
            .target = b.graph.host,
            .optimize = .ReleaseSafe,
        }),
    });
    const run_compression_deps_check = b.addRunArtifact(compression_deps_check_exe);
    const compression_deps_check_step = b.step(
        "check-ipc-compression-deps",
        "Verify vendored zstd/lz4 sources required for IPC BodyCompression",
    );
    compression_deps_check_step.dependOn(&run_compression_deps_check.step);

    const arrow_fbs_module = b.createModule(.{
        .root_source_file = b.path("src/ipc_schema/lib.zig"),
        .imports = &.{.{ .name = "fbs_runtime", .module = fbz_module }},
    });

    zarrow_module.addImport("fbs_runtime", fbz_module);
    zarrow_module.addImport("ipc_schema", arrow_fbs_module);

    // Tests still build a runnable artifact, but the package itself does not.
    const test_module = b.createModule(.{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
    });
    test_module.addImport("fbs_runtime", fbz_module);
    test_module.addImport("ipc_schema", arrow_fbs_module);

    const tests = b.addTest(.{ .root_module = test_module });
    configureIpcCompression(b, tests, &run_compression_deps_check.step);

    // Discover example files in the `examples` directory and wire them into the build.
    const io = b.graph.io;
    dependOnDir(b, "examples");
    var dir = openBuildDir(b, "examples") catch |err| {
        std.debug.print("warning: failed to open examples directory: {s}\n", .{@errorName(err)});
        return;
    };
    defer dir.close(io);

    // Wire the test artifact into a named build step for `zig build test`.
    const run_tests = b.addRunArtifact(tests);
    const test_step = b.step("test", "Run zarrow unit tests");
    test_step.dependOn(&run_tests.step);

    // Fuzzing (docs/fuzzing.md): one harness exe for every target, plus std.testing.fuzz tests.
    const fuzz_module = b.createModule(.{
        .root_source_file = b.path("fuzz/fuzz.zig"),
        .target = target,
        .optimize = optimize,
    });
    fuzz_module.addImport("zarrow", zarrow_module);
    // zstd/lz4 in their own library: a `--fuzz` rebuild would instrument the C sources with
    // sanitizer-coverage calls that Zig's fuzzer runtime does not provide.
    const fuzz_codecs = b.addLibrary(.{
        .linkage = .static,
        .name = "zarrow-fuzz-codecs",
        .root_module = b.createModule(.{ .target = target, .optimize = optimize }),
    });
    configureIpcCompression(b, fuzz_codecs, &run_compression_deps_check.step);
    fuzz_module.linkLibrary(fuzz_codecs);
    fuzz_module.link_libc = true;
    const fuzz_exe = b.addExecutable(.{ .name = "zarrow-fuzz", .root_module = fuzz_module });

    const run_fuzz = b.addRunArtifact(fuzz_exe);
    addPassthru(run_fuzz);
    b.step("fuzz", "Fuzz harness: zig build fuzz -- <target> replay|run ...").dependOn(&run_fuzz.step);
    b.step("fuzz-install", "Install the fuzz harness (zig-out/bin/zarrow-fuzz) for tools/fuzz.py").dependOn(&b.addInstallArtifact(fuzz_exe, .{}).step);

    const run_gen_seeds = b.addRunArtifact(fuzz_exe);
    run_gen_seeds.addArg("gen-seeds");
    run_gen_seeds.addDirectoryArg(b.path("fuzz/corpus"));
    b.step("fuzz-gen-seeds", "Re-encode fuzz/corpus/ipc-stream with zarrow writers into gen-* seeds").dependOn(&run_gen_seeds.step);

    const fuzz_corpus_step = b.step("fuzz-corpus", "Replay the committed fuzz corpus through every target");
    const corpus_dirs = [_][2][]const u8{
        .{ "ipc-stream", "ipc-stream" },
        .{ "ipc-stream-zero-copy", "ipc-stream" },
        .{ "compute", "ipc-stream" },
        .{ "ipc-file", "ipc-file" },
        .{ "array-layout", "array-layout" },
        .{ "c-abi", "c-abi" },
        .{ "regex", "regex" },
        .{ "temporal", "temporal" },
        .{ "integration-json", "integration-json" },
    };
    for (corpus_dirs) |entry| {
        const run_replay = b.addRunArtifact(fuzz_exe);
        run_replay.addArgs(&.{ entry[0], "replay" });
        run_replay.addDirectoryArg(b.path(b.fmt("fuzz/corpus/{s}", .{entry[1]})));
        fuzz_corpus_step.dependOn(&run_replay.step);
    }

    // `zig build fuzz-test --fuzz=100K` runs the coverage-guided engine (Linux); `test` runs them once.
    const fuzz_tests = b.addTest(.{ .root_module = fuzz_module });
    const run_fuzz_tests = b.addRunArtifact(fuzz_tests);
    b.step("fuzz-test", "std.testing.fuzz targets (add --fuzz=N on Linux)").dependOn(&run_fuzz_tests.step);
    test_step.dependOn(&run_fuzz_tests.step);

    var run_default: ?*std.Build.Step = null;
    var run_all_step = b.step("examples", "Run all examples");
    var it = dir.iterate();
    while (it.next(io) catch null) |entry| {
        if (entry.kind != .file) continue;
        if (!std.mem.endsWith(u8, entry.name, ".zig")) continue;

        const base_name = entry.name[0 .. entry.name.len - 4];
        const example_path = b.fmt("examples/{s}", .{entry.name});
        const exe = b.addExecutable(.{
            .name = b.fmt("example-{s}", .{base_name}),
            .root_module = b.createModule(.{
                .root_source_file = b.path(example_path),
                .target = target,
                .optimize = optimize,
            }),
        });
        configureIpcCompression(b, exe, &run_compression_deps_check.step);

        exe.root_module.addImport("zarrow", b.modules.get("zarrow").?);

        const run_exe = b.addRunArtifact(exe);
        const run_step = b.step(b.fmt("example-{s}", .{base_name}), b.fmt("Run example {s}", .{base_name}));
        run_step.dependOn(&run_exe.step);
        run_all_step.dependOn(&run_exe.step);

        if (run_default == null) run_default = &run_exe.step;
    }

    if (run_default) |step| {
        const run_step = b.step("run", "Run the default example");
        run_step.dependOn(step);
    }

    // IPC interop helper tools used by CI compatibility matrix checks.
    const interop_writer_exe = b.addExecutable(.{
        .name = "interop-fixture-writer",
        .root_module = b.createModule(.{
            .root_source_file = b.path("tools/interop_fixture_writer.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    configureIpcCompression(b, interop_writer_exe, &run_compression_deps_check.step);
    interop_writer_exe.root_module.addImport("zarrow", b.modules.get("zarrow").?);
    // The pyarrow canonical extension fixture, rewritten by zarrow for `canonical-ext`.
    interop_writer_exe.root_module.addAnonymousImport("pyarrow_canonical", .{ .root_source_file = b.path("src/extension/testdata/pyarrow_canonical.arrows") });

    const run_interop_writer = b.addRunArtifact(interop_writer_exe);
    addPassthru(run_interop_writer);
    const interop_writer_step = b.step("interop-fixture-writer", "Write canonical IPC fixture with zarrow");
    interop_writer_step.dependOn(&run_interop_writer.step);

    const interop_check_exe = b.addExecutable(.{
        .name = "interop-fixture-check",
        .root_module = b.createModule(.{
            .root_source_file = b.path("tools/interop_fixture_check.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    configureIpcCompression(b, interop_check_exe, &run_compression_deps_check.step);
    interop_check_exe.root_module.addImport("zarrow", b.modules.get("zarrow").?);

    const run_interop_check = b.addRunArtifact(interop_check_exe);
    addPassthru(run_interop_check);
    const interop_check_step = b.step("interop-fixture-check", "Validate canonical IPC fixture with zarrow");
    interop_check_step.dependOn(&run_interop_check.step);

    // Arrow integration suite (docs/integration-suite.md): archery's integration tester commands
    // and the golden-file check over an apache/arrow-testing clone.
    const integration_exe = b.addExecutable(.{
        .name = "integration-runner",
        .root_module = b.createModule(.{
            .root_source_file = b.path("tools/integration_runner.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    configureIpcCompression(b, integration_exe, &run_compression_deps_check.step);
    integration_exe.root_module.addImport("zarrow", b.modules.get("zarrow").?);
    b.step("integration-runner", "Install the Arrow integration tester (zig-out/bin/integration-runner)").dependOn(&b.addInstallArtifact(integration_exe, .{}).step);
    const integration_golden_step = b.step("integration-golden", "Golden-file check over apache/arrow-testing (-Darrow-testing=<dir>)");
    if (b.option([]const u8, "arrow-testing", "apache/arrow-testing clone for integration-golden")) |testing_dir| {
        const run_golden = b.addRunArtifact(integration_exe);
        run_golden.addArg("golden");
        run_golden.addArg(testing_dir);
        run_golden.has_side_effects = true;
        integration_golden_step.dependOn(&run_golden.step);
    } else {
        integration_golden_step.dependOn(&b.addFail(
            \\integration-golden needs -Darrow-testing=<dir>: a clone of apache/arrow-testing at revision
            \\5a26d35785ff8761e56d79eebff971f6f737e18c, e.g.
            \\  git clone https://github.com/apache/arrow-testing && git -C arrow-testing checkout 5a26d35785ff8761e56d79eebff971f6f737e18c
        ).step);
    }

    // Compute compatibility runner used by PyArrow compute alignment checks.
    const compute_compat_exe = b.addExecutable(.{
        .name = "compute-compat-runner",
        .root_module = b.createModule(.{
            .root_source_file = b.path("tools/compute_compat_runner.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    compute_compat_exe.root_module.addImport("zarrow-core", b.modules.get("zarrow-core").?);

    const run_compute_compat = b.addRunArtifact(compute_compat_exe);
    addPassthru(run_compute_compat);
    const compute_compat_step = b.step("compute-compat-check", "Run compute compatibility check cases");
    compute_compat_step.dependOn(&run_compute_compat.step);
    // The runner's test block replays the committed pyarrow results (part of `zig build test`).
    const compute_compat_tests = b.addTest(.{ .root_module = compute_compat_exe.root_module });
    const run_compute_compat_tests = b.addRunArtifact(compute_compat_tests);
    b.step("compute-compat-test", "Replay the committed pyarrow compute compat results").dependOn(&run_compute_compat_tests.step);
    test_step.dependOn(&run_compute_compat_tests.step);

    // Discover benchmark files in `benchmarks` and wire dedicated run steps.
    dependOnDir(b, "benchmarks");
    var benches = openBuildDir(b, "benchmarks") catch |err| {
        std.debug.print("warning: failed to open benchmarks directory: {s}\n", .{@errorName(err)});
        return;
    };
    defer benches.close(io);

    var bench_all_step = b.step("benchmark", "Run all benchmarks (default mode)");
    var bench_smoke_step = b.step("benchmark-smoke", "Run all benchmarks in smoke mode");
    var bench_full_step = b.step("benchmark-full", "Run all benchmarks in full mode");
    var bench_matrix_step = b.step("benchmark-matrix", "Run all benchmarks in matrix CSV mode");
    var bench_ci_step = b.step("benchmark-ci", "Run all benchmarks in CI CSV mode");
    var matrix_header_done = false;
    var ci_header_done = false;
    var prev_matrix_run: ?*std.Build.Step = null;
    var prev_ci_run: ?*std.Build.Step = null;
    var bench_it = benches.iterate();
    while (bench_it.next(io) catch null) |entry| {
        if (entry.kind != .file) continue;
        if (!std.mem.endsWith(u8, entry.name, ".zig")) continue;
        if (!std.mem.endsWith(u8, entry.name, "_benchmark.zig")) continue;

        const base_name = entry.name[0 .. entry.name.len - 4];
        const bench_path = b.fmt("benchmarks/{s}", .{entry.name});
        const bench_exe = b.addExecutable(.{
            .name = b.fmt("benchmark-{s}", .{base_name}),
            .root_module = b.createModule(.{
                .root_source_file = b.path(bench_path),
                .target = target,
                .optimize = optimize,
            }),
        });
        configureIpcCompression(b, bench_exe, &run_compression_deps_check.step);
        bench_exe.root_module.addImport("zarrow", b.modules.get("zarrow").?);

        const run_bench = b.addRunArtifact(bench_exe);
        const bench_step = b.step(b.fmt("benchmark-{s}", .{base_name}), b.fmt("Run benchmark {s}", .{base_name}));
        bench_step.dependOn(&run_bench.step);
        bench_all_step.dependOn(&run_bench.step);

        const run_bench_smoke = b.addRunArtifact(bench_exe);
        run_bench_smoke.addArg("smoke");
        bench_smoke_step.dependOn(&run_bench_smoke.step);

        const run_bench_full = b.addRunArtifact(bench_exe);
        run_bench_full.addArg("full");
        bench_full_step.dependOn(&run_bench_full.step);

        const run_bench_matrix = b.addRunArtifact(bench_exe);
        if (prev_matrix_run) |prev| run_bench_matrix.step.dependOn(prev);
        if (!matrix_header_done) {
            run_bench_matrix.addArg("matrix");
            matrix_header_done = true;
        } else {
            run_bench_matrix.addArg("matrix-no-header");
        }
        prev_matrix_run = &run_bench_matrix.step;
        bench_matrix_step.dependOn(&run_bench_matrix.step);

        const run_bench_ci = b.addRunArtifact(bench_exe);
        if (prev_ci_run) |prev| run_bench_ci.step.dependOn(prev);
        if (!ci_header_done) {
            run_bench_ci.addArg("ci");
            ci_header_done = true;
        } else {
            run_bench_ci.addArg("ci-no-header");
        }
        prev_ci_run = &run_bench_ci.step;
        bench_ci_step.dependOn(&run_bench_ci.step);
    }
}
