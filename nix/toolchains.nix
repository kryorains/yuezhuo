{ pkgs }:

let
  inherit (pkgs) lib;

  qemuUserAvailable = lib.meta.availableOn pkgs.stdenv.hostPlatform pkgs.qemu-user;

  mkLinuxToolchain =
    {
      name,
      compatibilityPrefix,
      crossPkgs,
      qemuBinary,
      conventionalSysroot,
    }:
    let
      cc = crossPkgs.stdenv.cc;
      libc = cc.libc;
      compiler = pkgs.writeShellScriptBin "${compatibilityPrefix}gcc" ''
        exec ${cc}/bin/${cc.targetPrefix}gcc "$@"
      '';
      runner = pkgs.writeShellScriptBin qemuBinary ''
        if [ "$#" -ge 2 ] && [ "$1" = "-L" ] && [ "$2" = ${pkgs.lib.escapeShellArg conventionalSysroot} ]; then
          shift 2
          set -- -L ${libc} "$@"
        fi

        exec ${pkgs.qemu-user}/bin/${qemuBinary} "$@"
      '';
    in
    pkgs.symlinkJoin {
      name = "yuezhuo-${name}-linux-toolchain";
      # Keep the compiler derivation out of PATH so its setup hook cannot set
      # the shell-wide CC/CXX to whichever cross target happens to come last.
      paths = [ compiler ] ++ lib.optionals qemuUserAvailable [ runner ];
    };
in
{
  inherit qemuUserAvailable;

  x86_64 = mkLinuxToolchain {
    name = "x86_64";
    compatibilityPrefix = "x86_64-linux-gnu-";
    crossPkgs = pkgs.pkgsCross.gnu64;
    qemuBinary = "qemu-x86_64";
    conventionalSysroot = "/usr/x86_64-linux-gnu";
  };

  aarch64 = mkLinuxToolchain {
    name = "aarch64";
    compatibilityPrefix = "aarch64-linux-gnu-";
    crossPkgs = pkgs.pkgsCross.aarch64-multiplatform;
    qemuBinary = "qemu-aarch64";
    conventionalSysroot = "/usr/aarch64-linux-gnu";
  };

  riscv64 = mkLinuxToolchain {
    name = "riscv64";
    compatibilityPrefix = "riscv64-linux-gnu-";
    crossPkgs = pkgs.pkgsCross.riscv64;
    qemuBinary = "qemu-riscv64";
    conventionalSysroot = "/usr/riscv64-linux-gnu";
  };
}
