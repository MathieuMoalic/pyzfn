{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    fix-python.url = "github:GuillaumeDesforges/fix-python";
  };

  outputs = {
    nixpkgs,
    flake-utils,
    fix-python,
    ...
  }:
    flake-utils.lib.eachDefaultSystem (
      system: let
        pkgs = import nixpkgs {inherit system;};
        devEnv = pkgs.mkShell {
          buildInputs = with pkgs; [
            fix-python.packages.${system}.default
            python311
            ruff
            ruff-lsp
            ffmpeg
          ];

          shellHook = ''
            set -euo pipefail
            test -d .venv || (${pkgs.python3.interpreter} -m venv .venv && source .venv/bin/activate && pip install -e . && fix-python --venv .venv && echo "use flake" > .envrc && direnv allow)
            source .venv/bin/activate
          '';
        };
      in {
        devShell = devEnv;
      }
    );
}
