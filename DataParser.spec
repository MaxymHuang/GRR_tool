# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the standalone data parser (GUI via cli.py with no -f)."""
import os

from PyInstaller.utils.hooks import collect_all

_spec_dir = os.path.dirname(os.path.abspath(SPEC))
_repo_root = _spec_dir
_pkg_parent = os.path.join(_repo_root, 'data_parser_app')
_entry = os.path.join(_pkg_parent, 'data_parser_app', 'cli.py')

datas = []
binaries = []
hiddenimports = [
    'data_parser_app',
    'data_parser_app.cli',
    'data_parser_app.gui',
    'data_parser_app.parser',
]

tmp_ret = collect_all('PySide6')
datas += tmp_ret[0]
binaries += tmp_ret[1]
hiddenimports += tmp_ret[2]

tmp_ret = collect_all('pandas')
datas += tmp_ret[0]
binaries += tmp_ret[1]
hiddenimports += tmp_ret[2]

a = Analysis(
    [_entry],
    pathex=[_pkg_parent],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='DataParser',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
