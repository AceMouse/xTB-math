let
  pkgs = import <nixpkgs> {};
in pkgs.mkShell {
  packages = [
    pkgs.pyright
    pkgs.texlab
    (pkgs.python3.withPackages (python-pkgs: with python-pkgs; [
      numpy
      scipy
    ]))
  ];
}

