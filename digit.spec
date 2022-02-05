# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_submodules
hidden_imports = collect_submodules('tensorflow.python.keras.engine.*') + collect_submodules('tensorflow.python.keras.engine.base_layer_v1')


block_cipher = None


a = Analysis(['digit.py'],
             pathex=['C:\\Users\\inder\\Desktop\\Projects\\Project - Handwritten Digit'],
             binaries=[],
             datas=[
                 ('model','models'),
                 ('tkinter_model','models')
                 ],
             hiddenimports=hidden_imports,
             hookspath=[],
             hooksconfig={},
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)

exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,  
          [],
          name='digit',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          upx_exclude=[],
          runtime_tmpdir=None,
          console=False,
          disable_windowed_traceback=False,
          target_arch=None,
          codesign_identity=None,
          entitlements_file=None )
