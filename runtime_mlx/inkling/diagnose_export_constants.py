import sys
from pathlib import Path
import mlx.core as mx
from runtime_mlx import Runtime
from runtime_mlx.inkling.fast_kernels import qkv_rel_prep_sliding_fused
if len(sys.argv) != 2:
    raise SystemExit('usage: python -m runtime_mlx.inkling.diagnose_export_constants MODEL_DIR')
m=Runtime.load(Path(sys.argv[1])).model.inner
l=m.layers[0];mx.random.seed(96);y=mx.random.normal((6,8576)).astype(mx.float32);k=mx.zeros((1,3,1024),mx.float32);v=mx.zeros((1,3,1024),mx.float32)
def f(y,k,v):return qkv_rel_prep_sliding_fused(y,l.q_scale,l.rel_scale,k,l.k_conv_w,v,l.v_conv_w,6,m.cfg.qk_l2_clip_threshold,m.cfg.rel_l2_clip_threshold)
a=f(y,k,v);mx.eval(*a)
Path('/private/tmp/inkling-native-layer').mkdir(parents=True,exist_ok=True)
mx.export_function('/private/tmp/inkling-native-layer/multi.mlxfn',f,y,k,v)
b=mx.import_function('/private/tmp/inkling-native-layer/multi.mlxfn')(y,k,v);mx.eval(*b)
for i,(x,z) in enumerate(zip(a,b)):
 print(i,x.shape,'same',bool(mx.array_equal(x,z).item()),'max_abs',float(mx.max(mx.abs(x.astype(mx.float32)-z.astype(mx.float32)))),flush=True)

import numpy as np
for k in range(2):
 x=np.array(a[2])[k];z=np.array(b[2])[k];print('half',k,'ref mean/std',float(x.mean()),float(x.std()),'export mean/std',float(z.mean()),float(z.std()),'first',x.flatten()[:5],z.flatten()[:5])
def weights(x): return (l.k_conv_w+x, l.v_conv_w+x)
z=mx.array(0.,dtype=mx.float32);c=weights(z);mx.eval(*c);mx.export_function('/private/tmp/inkling-native-layer/weights.mlxfn',weights,z);e=mx.import_function('/private/tmp/inkling-native-layer/weights.mlxfn')(z);mx.eval(*e)
for i,(x,w) in enumerate(zip(c,e)): print('weights',i,'same',bool(mx.array_equal(x,w).item()),'max',float(mx.max(mx.abs(x-w))))
