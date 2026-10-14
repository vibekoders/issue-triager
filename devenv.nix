{ pkgs, ... }:

{
  dotenv.enable = true;

  packages = with pkgs; [
    cargo-audit
    cargo-deny
    cargo-dist
    cargo-release
    cargo-watch

    sqlite
  ];

  languages = {
    rust = {
      enable = true;

      lld.enable = true;
    };

    python = {
      enable = true;
      directory = "./evals";

      uv = {
        enable = true;
        sync.enable = true;
      };
    };
  };
}
