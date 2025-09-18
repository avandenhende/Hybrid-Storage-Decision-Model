# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.building.build_main import Tree


a = Analysis(
    ['app.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('be/templates', 'be/templates'),
        ('be/static', 'be/static'),
        ('.env', '.')
    ],
    hiddenimports=[],
    hookspath=['hooks'],
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
    name='Hybrid Storage Decision Model',
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