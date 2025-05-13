{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    xtb.url = "github:grimme-lab/xtb";
    xtb.flake = false;
  };

  outputs = { self, nixpkgs, ... }: {
    apps."x86_64-linux" = let
      pkgs = nixpkgs.legacyPackages."x86_64-linux";
      xtb = (self.packages."x86_64-linux".xtb.overrideAttrs (finalAttrs: previousAttrs: {
        patches = [./test_output.patch];
      }));
      xtb_test_data = builtins.derivation {
        name = "xtb-test-data";
        system = "x86_64-linux";
        builder = "${pkgs.bash}/bin/bash";
        src = ./caffeine.xyz;
        args = ["-c" ''
          PATH=$PATH:${pkgs.coreutils}/bin
          ${xtb}/bin/xtb $src
          mv calls $out
        ''];
      };
    in {
      "cmp-impls" = let
        python = (pkgs.python3.withPackages (python-pkgs: with python-pkgs; [
          numpy
          scipy
        ]));
      in {
        type = "app";
        program = toString (pkgs.writeShellScript "cmp-impls" ''
          PYTHONPATH=${pkgs.lib.cleanSource ./.} exec ${python}/bin/python \
            ${./cmp_impls.py} ${xtb_test_data}
        '');
      };
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
