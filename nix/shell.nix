{
  pkgs,
  gitIdentity,
  linuxVm,
}:

let
  inherit (pkgs) lib;

  git = import ./git.nix { inherit lib; };
  toolchains = import ./toolchains.nix { inherit pkgs; };
  rustupAvailable = lib.meta.availableOn pkgs.stdenv.hostPlatform pkgs.rustup;
in
pkgs.mkShell {
  packages = [
    pkgs.bash
    pkgs.cargo
    pkgs.clippy
    pkgs.coreutils
    pkgs.diffutils
    pkgs.findutils
    pkgs.gawk
    pkgs.git
    pkgs.gnugrep
    pkgs.gnused
    pkgs.nixfmt
    pkgs.perl
    pkgs.rust-analyzer
    pkgs.rustc
    pkgs.rustfmt
    pkgs.stdenv.cc
    linuxVm
    toolchains.x86_64
    toolchains.riscv64
  ]
  ++ lib.optionals rustupAvailable [ pkgs.rustup ];

  RUST_BACKTRACE = "1";
  RUST_SRC_PATH = "${pkgs.rustPlatform.rustLibSrc}";

  shellHook = ''
    ${git.mkShellHook gitIdentity}

    # mkShell and cross-compiler setup hooks set CC/CXX globally. The test
    # scripts need an unset CC in order to select the compiler for TARGET.
    unset CC CXX

    echo "Rust: $(rustc --version)"
    echo "Linux cross-compilers: x86_64, riscv64"
    ${lib.optionalString toolchains.qemuUserAvailable ''
      echo "QEMU user-mode runners: x86_64, riscv64"
    ''}
    ${lib.optionalString (!toolchains.qemuUserAvailable) ''
      echo "QEMU user-mode runners: use yuezhuo-vm on ${pkgs.stdenv.hostPlatform.system}"
    ''}
  '';
}
