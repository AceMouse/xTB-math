{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
  };

  outputs = { self, nixpkgs }: {
    devShells."x86_64-linux".default = let
      pkgs = nixpkgs.legacyPackages."x86_64-linux";
    in pkgs.mkShell {
      packages = [
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
