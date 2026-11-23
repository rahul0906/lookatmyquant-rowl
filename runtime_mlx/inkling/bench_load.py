"""Experiment with memory mapping uncompressed NPZ model arrays; use only local model paths."""
from __future__ import annotations
import argparse
import json
import struct
import time
import zipfile
from pathlib import Path
import numpy as np


class MappedNpz:
    def __init__(self, path, direct=False):
        self.path = Path(path)
        self.direct = direct
        self.zip = zipfile.ZipFile(path)
        self.entries = {item.filename[:-4]: item for item in self.zip.infolist() if item.filename.endswith('.npy')}
        self.files = list(self.entries)
        if any(item.compress_type != zipfile.ZIP_STORED for item in self.entries.values()):
            raise ValueError('compressed archive cannot be mapped')

    def __contains__(self, key):
        return key in self.entries

    def __getitem__(self, key):
        info = self.entries[key]
        with self.path.open('rb') as f:
            f.seek(info.header_offset)
            header = f.read(30)
            signature, _, flags, compression, _, _, _, _, _, name_len, extra_len = struct.unpack('<IHHHHHIIIHH', header)
            if signature != 0x04034b50 or compression != 0 or flags & 1:
                raise ValueError('unsupported ZIP entry')
            f.seek(name_len + extra_len, 1)
            version = np.lib.format.read_magic(f)
            if version == (1, 0):
                shape, fortran, dtype = np.lib.format.read_array_header_1_0(f)
            elif version in ((2, 0), (3, 0)):
                shape, fortran, dtype = np.lib.format.read_array_header_2_0(f)
            else:
                raise ValueError(f'unsupported NPY version {version}')
            if dtype.hasobject:
                raise ValueError('object arrays are not supported')
            offset = f.tell()
        if self.direct:
            with self.path.open('rb') as f:
                f.seek(offset)
                array = np.fromfile(f, dtype=dtype, count=int(np.prod(shape)))
            return array.reshape(shape, order='F' if fortran else 'C')
        return np.memmap(self.path, mode='r', dtype=dtype, offset=offset, shape=shape, order='F' if fortran else 'C')

    def close(self):
        self.zip.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('model', type=Path)
    p.add_argument('drafter', type=Path)
    p.add_argument('--mapped', action='store_true')
    p.add_argument('--direct', action='store_true')
    p.add_argument('--out', type=Path, required=True)
    a=p.parse_args()
    if not a.model.is_dir() or not a.drafter.exists():
        p.error('local model and drafter required')
    import mlx.core as mx
    from runtime_mlx import Runtime
    original=np.load
    if a.mapped and a.direct:
        p.error('choose mapped or direct')
    if a.mapped or a.direct:
        def mapped_load(file, *args, **kwargs):
            if isinstance(file, (str, Path)) and str(file).endswith('.npz') and Path(file).parent == a.model:
                return MappedNpz(file, direct=a.direct)
            return original(file, *args, **kwargs)
        np.load=mapped_load
    start=time.perf_counter()
    rt=Runtime.load(a.model,drafter=a.drafter)
    load=time.perf_counter()-start
    result=rt.generate('What is 17 times 23? Show the steps.',max_tokens=128)
    data={'mapped':a.mapped,'direct':a.direct,'model':str(a.model),'drafter':str(a.drafter),'load_s':load,'tokens':result.token_ids,
          'finish_reason':result.finish_reason,'generation_tps':result.generation_tps,'active_bytes':mx.get_active_memory(),'peak_bytes':mx.get_peak_memory()}
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(data,indent=2))
    print(json.dumps(data))

if __name__=='__main__':
    main()
