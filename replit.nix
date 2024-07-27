{pkgs}: {
  deps = [
    pkgs.glibcLocales
    pkgs.rustc
    pkgs.libiconv
    pkgs.cargo
  ];
}
