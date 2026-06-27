# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ["FaceSlim_v1.py"],
    pathex=[],
    binaries=[],
    datas=[
        ("icon.png", "."),
        ("icon.ico", "."),
    ],
    hiddenimports=[
        "mediapipe",
        "mediapipe.tasks",
        "mediapipe.tasks.python",
        "mediapipe.tasks.python.vision",
        "onnxruntime",
        "PIL",
        "scipy.interpolate",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=["runtime_hook_mp.py"],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="FaceSlim",
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
    icon="icon.ico",
)
