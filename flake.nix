{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    xtb.url = "github:grimme-lab/xtb";
    xtb.flake = false;
  };

  outputs = inputs @ { self, nixpkgs, ... }: {
    #checks."x86_64-linux"."compare-implementations" = let
    #  pkgs = nixpkgs.legacyPackages."x86_64-linux";
    #in pkgs.stdenvNoCC.mkDerivation {
    #  name = "compare-implementations";
    #  srcs = [./. inputs.xtb];

    #  patchPhase = ''
    #    runHook prePatch

    #    git -C "''${srcs[1]}" apply 

    #    runHook postPatch
    #  '';
    #} ''
    #  for src in $srcs;
    #  do
    #    ls $src
    #  done
    #  mkdir "$out"
    #'';

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
        #pkgs.pyright
        #pkgs.texlab
        #(pkgs.python3.withPackages (python-pkgs: with python-pkgs; [
        #  numpy
        #  scipy
        #]))
        meson
        ninja
        gfortran
        pkg-config
        cmake
        lapack
        python3
        blas
        asciidoctor
        tblite
        go
        toml-f
        test-drive
        simple-dftd3
        mstore
        multicharge
        dftd4
        json-fortran
        mctc-lib
        #self.packages.x86_64-linux.cpx
      ];
    };
  };
}
