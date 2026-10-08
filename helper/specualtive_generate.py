"""Production dispatcher: both methods use TLT scheduling on FastGRPO."""
from helper.method_config import resolve_method
from helper.fastgrpo_generate import sampling

def speculative_generate(*args,method='tlt',**kwargs):
    resolve_method(method)
    from helper.tlt_generate import speculative_generate as generate
    return generate(*args,method=method,**kwargs)
