# -*- mode: python -*-
''' Creates a stand alone distribution for tba.
'''

import sys
import os
import testbeam_analysis

mod_path = os.path.dirname(testbeam_analysis.__file__)
gui_path = os.path.join(mod_path, 'gui/main.py')

# Fix missing qt.conf issue
with open('qt.conf', 'w') as out:
    out.write('[Paths]\n')
    out.write('plugins = PyQt5/Qt/plugins')

block_cipher = None

a = Analysis([gui_path],
             # pathex=['/home/davidlp/git/testbeam_analysis/testbeam_analysis/gui'],
             # for tables
             binaries=[(os.path.join(sys.prefix, 'Library/bin/lzo2.dll'), '.')],
             datas=[ (os.path.join(mod_path, 'gui/dut_types.yaml'), '.'),
                     ('qt.conf', '.') ],
             hiddenimports=['setuptools.msvc',
                            'pixel_clusterizer.clusterizer',
                            'numpydoc',
                            'progressbar'],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          exclude_binaries=True,
          name='tba',
          debug=False,
          strip=False,
          upx=False,
          console=True,
          # Missing runtime c includes that pyinstaller cannot see
          # cythonized analysis part
          resources=[('analysis_functions.pyd,dll,analysis_functions.pyd')]) 
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=False,
               name='tba')

