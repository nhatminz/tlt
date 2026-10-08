"""TLT adaptive rollout + FastGRPO drafter, optional source ReflexOPD."""
METHOD_TO_REFLEX_MODE={'tlt':'off','tlt_opd_reflex':'opd_reflex'}
def resolve_method(method):
    value=str(method).strip().lower()
    if value not in METHOD_TO_REFLEX_MODE: raise ValueError('METHOD must be tlt or tlt_opd_reflex')
    return value,METHOD_TO_REFLEX_MODE[value]
