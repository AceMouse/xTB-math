{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    xtb.url = "github:grimme-lab/xtb";
    xtb.flake = false;
  };

  outputs = { self, nixpkgs, ... }: {
    checks."x86_64-linux" = let
      pkgs = nixpkgs.legacyPackages."x86_64-linux";
    in {
      "cmp-impls" = pkgs.runCommand "cmp-impls" {
        nativeBuildInputs = with pkgs; [
          (self.packages."x86_64-linux".xtb.overrideAttrs (finalAttrs: previousAttrs: {
            patches = [./test_output.patch];
          }))
          (pkgs.python3.withPackages (python-pkgs: with python-pkgs; [
            numpy
            scipy
          ]))
        ];
      } ''
        xtb ${./caffeine.xyz}
        mv calls $out
        PYTHONPATH=${./.} python ${./cmp_impls.py} $out/get_multiints/
      '';
    };

    packages."x86_64-linux" = let
      pkgs = nixpkgs.legacyPackages."x86_64-linux";
    in rec {
      xtb = pkgs.callPackage ./xtb.nix { inherit cpx numsa; };
      cpx = pkgs.callPackage ./cpx.nix { inherit numsa; };
      numsa = pkgs.callPackage ./numsa.nix {};
    };

    devShells."x86_64-linux".default = let
      pkgs = nixpkgs.legacyPackages."x86_64-linux";
    in pkgs.mkShell {
      packages = with pkgs; [
        pkgs.pyright
        pkgs.texlab
        (pkgs.python3.withPackages (python-pkgs: with python-pkgs; [
          numpy
          scipy
        ]))
      ];
    };
  };
}
