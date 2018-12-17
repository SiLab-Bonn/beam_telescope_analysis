# -*- mode: python -*-
''' Creates a stand alone distribution for tba.
To create a packed installer use makeself:
makeself --bzip2 dist/tba/ tba.bz2.run "Installer for testbeam analysis" tba
'''

import sys
import os
import beam_telescope_analysis

mod_path = os.path.dirname(beam_telescope_analysis.__file__)
gui_path = os.path.join(mod_path, 'gui/main.py')

block_cipher = None

a = Analysis([gui_path],
             pathex=['/home/davidlp/git/beam_telescope_analysis/beam_telescope_analysis/gui'],
             # for libmkl
             binaries=[(os.path.join(sys.prefix, 'lib/libiomp5.so'), '.')],
             datas=[ (os.path.join(mod_path, 'gui/dut_types.yaml'), '.') ],
             hiddenimports=['setuptools.msvc',
                            'pixel_clusterizer.cluster_functions',
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
          resources=['analysis_functions.so,dll,analysis_functions.so'])
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=False,
               name='tba')
