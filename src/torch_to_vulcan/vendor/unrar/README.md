# Bundled UnRAR decoder

Torch to Vulcan bundles the Windows x64 `UnRAR.exe` decoder so RAR imports work
without a separate WinRAR installation.

- Product: UnRAR 7.23 x64
- Copyright: Alexander Roshal
- Publisher: win.rar GmbH
- SHA-256: `0D3715001790F0FD18D3E850F947B540530B2D2DEB9A2E6A9E84F2ED7B234235`
- Upstream: <https://www.rarlab.com/rar_add.htm>

The original license is stored beside the executable as
`LICENSE.UnRAR.txt`. Its redistribution terms explicitly permit UnRAR to be
distributed inside other software packages. The decoder must not be used to
recreate the proprietary RAR compression algorithm.

The executable is used only to read RAR archives. Torch to Vulcan does not use
it to create archives and does not modify it.

On non-Windows platforms, install `unrar` and make it available on `PATH`, or
set `TTV_UNRAR` to the decoder path.
