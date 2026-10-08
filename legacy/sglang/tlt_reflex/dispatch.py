"""Setup-only cost tables: same interpolation/argmin policy as SpecNaacl.

Device lookup is exact for integer active counts, including nonmonotone GEMM
regions. Tables are built before capture; generation does no tuning or D2H.
"""
import bisect
import math
import torch
from tlt_reflex.ported.profiles import _between


class DispatchTables:
    def __init__(self,profile,vocab,max_rows,device):
        self.vocab=vocab;self.profile=profile;self.max_rows=max_rows
        self.can_gemm=False;self.sparse_capacity=0
        if profile is None:
            self.breakpoints=torch.tensor([0,vocab//8,max(vocab//8+1,vocab)],device=device,dtype=torch.int32)
            self.costs=None;self.points=0
            self.sparse_capacity=max(1,vocab//8)
            return
        # Union of active-count knots retains piecewise linear costs exactly.
        points=sorted({0,vocab,*[min(vocab,s) for slots,_ in profile.buckets.values() for s in slots if s<=vocab]})
        self.points=len(points);self.breakpoints=torch.tensor(points,device=device,dtype=torch.int32)
        # Context interpolation follows the source policy in log(contexts).
        costs=torch.tensor([[profile.costs(n,s) for s in points] for n in range(1,max_rows+1)],dtype=torch.float64)
        self.costs=costs.to(device)
        # Workspace bounds must include every integer count where an
        # interpolated cost line can win (not just measured endpoint winners).
        for row in costs.tolist():
            for i,(lo,hi) in enumerate(zip(points,points[1:])):
                left,right=row[i],row[i+1];cuts={0.,1.}
                for a in range(3):
                    for b in range(a+1,3):
                        d0=left[a]-left[b];d1=right[a]-right[b]
                        if d0!=d1:
                            t=-d0/(d1-d0)
                            if 0<t<1:cuts.add(t)
                cuts=sorted(cuts)
                for a,b in zip(cuts,cuts[1:]):
                    t=(a+b)/2
                    winner=min(range(3),key=lambda j:left[j]+(right[j]-left[j])*t)
                    if winner==2:self.can_gemm=True
                    if winner==0:self.sparse_capacity=max(self.sparse_capacity,min(vocab,math.ceil(lo+(hi-lo)*b)+1))
                for t in cuts:
                    winner=min(range(3),key=lambda j:left[j]+(right[j]-left[j])*t)
                    if winner==2:self.can_gemm=True
                    if winner==0:self.sparse_capacity=max(self.sparse_capacity,min(vocab,math.ceil(lo+(hi-lo)*t)+1))
        self.sparse_capacity=max(1,self.sparse_capacity)
