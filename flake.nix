{
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.05";

  outputs = {nixpkgs, ...}: let
    system = "x86_64-linux";
    pkgs = import nixpkgs {
      inherit system;
    };
  in {
    devShells.${system}.default = pkgs.mkShell {
      buildInputs = [pkgs.uv];

      shellHook = ''
        # Create .venv if it doesn't exist
        if [ ! -d ".venv" ]; then
          echo "Creating virtual environment with uv..."
          uv venv .venv
        fi

        source .venv/bin/activate
      '';
    };
  };
}
