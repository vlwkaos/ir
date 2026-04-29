{
  description = "ir - local markdown semantic search with hybrid BM25+vector retrieval and LLM reranking";

  inputs = {
    flake-parts.url = "github:hercules-ci/flake-parts";
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs =
    inputs@{ flake-parts, ... }:
    flake-parts.lib.mkFlake { inherit inputs; } {
      systems = [
        "x86_64-linux"
        "aarch64-linux"
        "aarch64-darwin"
        "x86_64-darwin"
      ];
      perSystem =
        {
          pkgs,
          system,
          config,
          ...
        }:
        {
          _module.args.pkgs = import inputs.nixpkgs {
            inherit system;
            overlays = [
              (import inputs.rust-overlay)
            ];
            config = { };
          };
          packages.default = pkgs.rustPlatform.buildRustPackage {
            pname = "ir";
            version = (builtins.fromTOML (builtins.readFile ./Cargo.toml)).package.version;
            src = ./.;
            cargoLock.lockFile = ./Cargo.lock;

            nativeBuildInputs = [
              pkgs.cmake
              pkgs.python3 # used by preprocess sentinel tests
            ];

            buildInputs = [ pkgs.llvmPackages.openmp ];

            # ggml is statically linked into the Rust binary; the final link
            # step needs -lomp because cmake's OpenMP detection only affects
            # the static archive, not the cargo link line.
            env.NIX_LDFLAGS = "-lomp";

            CMAKE_GENERATOR = "Unix Makefiles";
          };
          devShells.default = pkgs.mkShell {
            inputsFrom = [ config.packages.default ];
            nativeBuildInputs = [
              (pkgs.rust-bin.stable."1.95.0".default.override {
                extensions = [ "rust-src" ];
              })
            ];
          };
          formatter = pkgs.nixfmt-rfc-style;
        };
    };
}
