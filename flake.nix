{
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.05";

  outputs = {nixpkgs, ...}: let
    system = "x86_64-linux";
    pkgs = import nixpkgs {
      inherit system;
    };

    py = pkgs.python313;

    zarr = py.pkgs.buildPythonPackage rec {
      pname = "zarr";
      version = "3.0.8";
      src = pkgs.fetchPypi {
        inherit pname version;
        sha256 = "sha256-iFBdCVr4maiK6KxNsC9GUO8IAdL/b2W20fCkXc92Cm0=";
      };
      format = "pyproject";
      nativeBuildInputs = with py.pkgs; [hatchling hatch-vcs];
      propagatedBuildInputs = with py.pkgs; [numcodecs donfig];
      doCheck = false;
    };

    pyzfn = py.pkgs.buildPythonPackage {
      pname = "pyzfn";
      version = "1.0.2";
      src = ./.;
      format = "pyproject";
      nativeBuildInputs = with py.pkgs; [setuptools];
      propagatedBuildInputs = with py.pkgs; [
        matplotlib
        numpy
        zarr
        typing-extensions
        crc32c
        rich
      ];
      doCheck = false;
    };
  in {
    packages.${system}.default = pyzfn;
    devShells.${system}.default = pkgs.mkShell {buildInputs = [pkgs.uv];};
  };
}
