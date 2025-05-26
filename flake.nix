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
        patches = [./patches/xtb/log_args_and_outputs.patch];
      }));
      multicharge = (pkgs.multicharge.overrideAttrs (finalAttrs: previousAttrs: rec {
        version = "0.4.0";
        src = pkgs.fetchFromGitHub {
          owner = "grimme-lab";
          repo = "multicharge";
          rev = "v${version}";
          hash = "sha256-53u3eiDQruiemtu+6fB39eP9PxP9UdTYn1Jmo5RGHjk=";
        };
      }));
      dftd4 = (pkgs.dftd4.overrideAttrs (finalAttrs: previousAttrs: {
        src = pkgs.fetchFromGitHub {
          owner = "dftd4";
          repo = "dftd4";
          rev = "502d7c59bf88beec7c90a71c4ecf80029794bd5e";
          hash = "sha256-FEABtBAZK0xQ1P/Pbj5gUuvKf8/ZLITXaXYB+btAY/8=";
        };
        buildInputs = [ multicharge ] ++ previousAttrs.buildInputs;
        doCheck = false;
        patches = previousAttrs.patches ++ [
          ./patches/dftd4/use_gfn2.patch
          ./patches/dftd4/log_args_and_outputs.patch
        ];
      }));
      xtb_test_data = builtins.derivation {
        name = "xtb-test-data";
        system = "x86_64-linux";
        builder = "${pkgs.bash}/bin/bash";
        src = ./caffeine.xyz;
        args = ["-c" ''
          PATH=$PATH:${pkgs.coreutils}/bin
          ${xtb}/bin/xtb $src
          ${dftd4}/bin/dftd4 $src
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
