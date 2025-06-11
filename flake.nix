{
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.05";

  outputs = {nixpkgs, ...}: let
    system = "x86_64-linux";
    pkgs = import nixpkgs {
      inherit system;
    };

    py = pkgs.python313;

    ipytree = py.pkgs.buildPythonPackage rec {
      pname = "ipytree";
      version = "0.2.2";
      src = pkgs.fetchPypi {
        inherit pname version;
        sha256 = "sha256-1T1zm7qqRUFXM80G4NxCCirz0XNDhhfbRypRe8emHlY=";
      };
      nativeBuildInputs = with py.pkgs; [setuptools wheel jupyter-packaging];
      doCheck = false;
    };

    cmocean = py.pkgs.buildPythonPackage rec {
      pname = "cmocean";
      version = "4.0.3";
      src = pkgs.fetchPypi {
        inherit pname version;
        sha256 = "sha256-N4aDmftfQbTqxZbmmAP5v66kmUZRTfsuf0iIaFQlDXw=";
      };
      format = "pyproject";
      nativeBuildInputs = with py.pkgs; [setuptools wheel];
      propagatedBuildInputs = with py.pkgs; [numpy matplotlib];
      doCheck = false;
    };

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
      propagatedBuildInputs = with py.pkgs; [
        ipympl
        matplotlib
        numpy
        psutil
        tqdm
        zarr
        matplotx
        cmocean
        ipytree
        typing-extensions
        crc32c
        rich
      ];
      doCheck = false;
    };
  in {
    packages.${system}.default = pyzfn;
    devShells.${system}.default = pkgs.mkShell {buildInputs = with py.pkgs; [pyzfn pip pytest pytest-cov];};
  };
}
