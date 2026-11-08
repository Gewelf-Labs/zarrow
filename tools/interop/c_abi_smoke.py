#!/usr/bin/env python3
"""Build and run the C ABI smoke programs of the CI jobs interop-c-abi-cpp-smoke and
interop-c-abi-rs-smoke (tools/interop/cpp/c_abi_smoke.cpp, tools/interop/arrow-rs) against the
zarrow_c library of this build. Called by `zig build interop-c-abi-cpp-smoke` and
`zig build interop-c-abi-rs-smoke` (see docs/interop-c-abi.md); run it with `python -P` so the
user site-packages (pyarrow) stay visible.

C++: the Arrow C++ headers and library are `--arrow-include` / `--arrow-lib`, by default those of
the installed pyarrow. On Windows pyarrow's arrow.dll has the MSVC ABI, so the program is compiled
with `zig cc -target x86_64-windows-msvc` and the MSVC standard library (C++20, which pyarrow 26's
headers need); elsewhere with `$CXX` (default `c++`, the compiler that built Arrow, as in CI).
Rust: `cargo run` with `-L native=<zarrow_c directory>`.
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def run(cmd, env):
    print("+ " + " ".join(str(c) for c in cmd), flush=True)
    return subprocess.run([str(c) for c in cmd], env=env).returncode


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("lane", choices=["cpp", "rs"])
    p.add_argument("--zig", required=True, help="the zig executable")
    p.add_argument("--lib", required=True, help="the zarrow_c shared library")
    p.add_argument("--implib", help="its import library (Windows)")
    p.add_argument("--arrow-include", help="Arrow C++ include directory (default: pyarrow's)")
    p.add_argument("--arrow-lib", help="Arrow C++ library directory (default: pyarrow's)")
    a = p.parse_args()

    lib = Path(a.lib).resolve()
    link = Path(a.implib).resolve() if a.implib else lib
    windows = os.name == "nt"
    env = dict(os.environ)
    dll_dirs = [str(lib.parent)]

    if a.lane == "rs":
        env["RUSTFLAGS"] = f"-L native={link.parent}"
        cmd = ["cargo", "run", "--manifest-path", ROOT / "tools/interop/arrow-rs/Cargo.toml", "--bin", "c_abi_smoke"]
    else:
        include, lib_dirs = a.arrow_include, [a.arrow_lib] if a.arrow_lib else []
        if include is None or not lib_dirs:
            import pyarrow

            include = include or pyarrow.get_include()
            lib_dirs = lib_dirs or pyarrow.get_library_dirs()
        dll_dirs += lib_dirs
        out_dir = Path(tempfile.mkdtemp(prefix="zarrow-c-abi-smoke-"))
        exe = out_dir / ("c_abi_smoke.exe" if windows else "c_abi_smoke")
        src = ROOT / "tools/interop/cpp/c_abi_smoke.cpp"
        if windows:
            arrow_lib = next(Path(d) / "arrow.lib" for d in lib_dirs if (Path(d) / "arrow.lib").exists())
            compile_cmd = [a.zig, "cc", "-std=c++20", "-target", "x86_64-windows-msvc", "-nostdlib++",
                           f"-I{ROOT / 'include'}", f"-I{include}", "-x", "c++", src, "-x", "none",
                           arrow_lib, link, "-o", exe]
        else:
            compile_cmd = [env.get("CXX", "c++"), "-std=c++17", src, f"-I{ROOT / 'include'}", f"-I{include}",
                           f"-L{link.parent}", f"-Wl,-rpath,{link.parent}", "-lzarrow_c",
                           *(f"-L{d}" for d in lib_dirs), *(f"-Wl,-rpath,{d}" for d in lib_dirs), "-larrow", "-o", exe]
        cmd = [exe]

    key = "PATH" if windows else ("DYLD_LIBRARY_PATH" if sys.platform == "darwin" else "LD_LIBRARY_PATH")
    env[key] = os.pathsep.join(dll_dirs + ([env[key]] if env.get(key) else []))
    try:
        if a.lane == "cpp" and run(compile_cmd, env) != 0:
            return 1
        return run(cmd, env)
    finally:
        if a.lane == "cpp":
            shutil.rmtree(out_dir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
