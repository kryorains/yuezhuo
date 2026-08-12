{
  description = "Yuezhuo development environments";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs =
    { nixpkgs, ... }:
    let
      supportedSystems = [
        "aarch64-darwin"
        "aarch64-linux"
        "x86_64-linux"
      ];

      forAllSystems = nixpkgs.lib.genAttrs supportedSystems;
    in
    {
      devShells = forAllSystems (
        system:
        let
          pkgs = import nixpkgs { inherit system; };
          linuxPkgs = import nixpkgs { system = "aarch64-linux"; };
          git = import ./nix/git.nix { lib = pkgs.lib; };
          linuxVm = import ./nix/vm.nix {
            inherit pkgs linuxPkgs;
          };
          mkDevShell =
            gitIdentity:
            import ./nix/shell.nix {
              inherit pkgs gitIdentity linuxVm;
            };
        in
        rec {
          kryorains = mkDevShell git.identities.kryorains;

          dev = kryorains;
          default = kryorains;
        }
      );

      packages = forAllSystems (
        system:
        let
          pkgs = import nixpkgs { inherit system; };
          linuxPkgs = import nixpkgs { system = "aarch64-linux"; };
        in
        {
          linux-vm = import ./nix/vm.nix {
            inherit pkgs linuxPkgs;
          };
        }
      );

      formatter = forAllSystems (
        system:
        let
          pkgs = import nixpkgs { inherit system; };
        in
        pkgs.nixfmt
      );
    };
}
