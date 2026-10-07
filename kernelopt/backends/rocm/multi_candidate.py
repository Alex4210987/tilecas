import tilelang
import tilelang.language as T
@tilelang.jit(out_idx=[1], target='hip', execution_backend='cython')
def add(N=256):
    @T.prim_func
    def main(A:T.Tensor((N,), 'float32'), B:T.Tensor((N,), 'float32')):
        with T.Kernel(T.ceildiv(N,128), threads=128) as bx:
            for i in T.Parallel(128):
                B[bx*128+i] = A[bx*128+i]+1
    return main
class ModelNew:
    def __init__(self): self.kernel=add()
    def to(self,*a,**k): return self
    def __call__(self,*a): return self.forward(*a)
    def forward(self,a): return self.kernel(self.kernel(a))
