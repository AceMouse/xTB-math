{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
  };

  outputs = { self, nixpkgs, ... }: {
    apps."x86_64-linux" = let
      pkgs = nixpkgs.legacyPackages."x86_64-linux";
      xtb = (self.packages."x86_64-linux".xtb.overrideAttrs (finalAttrs: previousAttrs: {
        patches = [
          ./nix/patches/xtb/log_utils.patch
          ./nix/patches/xtb/log_args_and_outputs.patch
        ];
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
          ./nix/patches/dftd4/use_gfn2.patch
          ./nix/patches/dftd4/log_args_and_outputs.patch
        ];
      }));
      xtb_test_data = builtins.derivation {
        name = "xtb-test-data";
        system = "x86_64-linux";
        builder = "${pkgs.bash}/bin/bash";
        src = ./xtb-python/data/C200.xyz;
        args = ["-c" ''
          PATH=$PATH:${pkgs.coreutils}/bin
          mkdir -p ./calls/{build_SDQH0,coordination_number,dim_basis,dtrf2,electro,form_product,get_multiints,h0scal,horizontal_shift,multipole_3d,newBasisset,olapp}
          ${xtb}/bin/xtb $src
          ${dftd4}/bin/dftd4 $src
          mv calls $out
        ''];
      };

      electro_data = let
        xtb = (self.packages."x86_64-linux".xtb.overrideAttrs (finalAttrs: previousAttrs: {
          patches = [
            ./nix/patches/xtb/log_utils.patch
            ./nix/patches/xtb/log_electro.patch
          ];
        }));
      in builtins.derivation {
        name = "xtb-electro-data";
        system = "x86_64-linux";
        builder = "${pkgs.bash}/bin/bash";
        src = ./bin2xyz;
        args = ["-c" ''
          PATH=$PATH:${pkgs.coreutils}/bin:${pkgs.clang}/bin
          cp $src/* .
          clang++ -o bin2xyz bin2xyz.cpp -O3
          ./bin2xyz ./C200_10000_fullerenes.float64 10000

          count=0
          for file in ./output/*; do
            count=$((count + 1))
            ${xtb}/bin/xtb "$file" > /dev/null
            echo "[$count/10000] Processing: $file"
          done

          mv calls $out
        ''];
      };
    in {
      "cmp-impls" = let
        python = (pkgs.python3.withPackages (python-pkgs: with python-pkgs; [
          numpy
          scipy
          cvxopt
        ]));
      in {
        type = "app";
        program = toString (pkgs.writeShellScript "cmp-impls" ''
          PYTHONPATH=${pkgs.lib.cleanSource ./xtb-python} exec ${python}/bin/python \
            ${./xtb-python/cmp_impls.py} ${xtb_test_data}
        '');
      };

      "check-electro" = let
        python = (pkgs.python3.withPackages (python-pkgs: with python-pkgs; [
          numpy
          scipy
          cvxopt
        ]));
        electro_test_data = let
          xtb = (self.packages."x86_64-linux".xtb.overrideAttrs (finalAttrs: previousAttrs: {
            patches = [
              ./nix/patches/xtb/log_utils.patch
              ./nix/patches/xtb/electro_correctness_check.patch
            ];
          }));
        in builtins.derivation {
          name = "electro-test-data";
          system = "x86_64-linux";
          builder = "${pkgs.bash}/bin/bash";
          src = ./xtb-python/data/C100-IPR;
          args = ["-c" ''
            PATH=$PATH:${pkgs.coreutils}/bin

            for f in $src/*.xyz; do
              base="$(basename "''${f%.*}")"
              mkdir -p $base/calls/electro

              pushd $base
              ${xtb}/bin/xtb $f
              popd
            done
            mv * $out
          ''];
        };
      in {
        type = "app";
        program = toString (pkgs.writeShellScript "cmp-impls" ''
          c=0
          for dir in ${electro_test_data}/*/; do
            [ -d "$dir" ] || continue

            PYTHONPATH=${pkgs.lib.cleanSource ./xtb-python} exec ${python}/bin/python \
              ${./xtb-python/cmp_impls.py} "$dir"
            c=$((c + 1))
          done
        '');
      };

      "bench-electro" = {
        type = "app";
        program = toString (pkgs.writeShellScript "bench-electro" ''
          echo ${electro_data}
        '');
      };
    };

    packages."x86_64-linux" = let
      pkgs = nixpkgs.legacyPackages."x86_64-linux";
    in rec {
      xtb = pkgs.callPackage ./nix/xtb.nix { inherit cpx numsa; };
      cpx = pkgs.callPackage ./nix/cpx.nix { inherit numsa; };
      numsa = pkgs.callPackage ./nix/numsa.nix {};
    };

    devShells."x86_64-linux".default = let
      pkgs = nixpkgs.legacyPackages."x86_64-linux";
    in pkgs.mkShell.override { stdenv = pkgs.clangStdenv; } {
      packages = with pkgs; [
        pkgs.pyright
        pkgs.texlab
        (pkgs.python3.withPackages (python-pkgs: with python-pkgs; [
          numpy
          scipy
          cvxopt
        ]))
      ];
    };
  };
}
