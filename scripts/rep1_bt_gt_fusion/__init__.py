"""Rep1 six-endpoint B/T extension and two production-only Geometry+Type fusions."""

SCHEMA = 'rep1-bt-gt-science-v1'
EXISTING = ('B0', 'BASE_MATCH', 'V2_MORPH', 'P2_BASE', 'A2_GT', 'P2_MORPH')
NEW = ('P2_BASE_GT', 'V2_MORPH_GT')
PARENTS = dict(zip(NEW, ('P2_BASE', 'V2_MORPH')))
ASSIGNED = {'P2_BASE_GT': 21654424, 'V2_MORPH_GT': 21449617}
NODE_ORDER = ('BT_EXISTING6', 'P2_BASE_GT_PRODUCE', 'V2_MORPH_GT_PRODUCE', 'PRODUCTION_VALIDATE')
N_CELLS = 107955
N_GENES = 313
ELIGIBLE = 34442716
APPLICABLE_TYPES = [0, 2, 3, 4, 5, 6, 7, 9, 10, 13]


def comparisons(methods=EXISTING + NEW):
    """Prespecified directed comparisons; candidate minus control, never adaptive."""
    result = []
    def add(a, b):
        if a != b and a in methods and b in methods and (a, b) not in result:
            result.append((a, b))
    for a in EXISTING[1:]:
        for b in ('B0', 'BASE_MATCH'):
            add(a, b)
    add('P2_BASE', 'BASE_MATCH')
    add('P2_MORPH', 'V2_MORPH')
    for a in NEW:
        for b in (PARENTS[a], 'A2_GT', 'BASE_MATCH', 'B0', 'P2_MORPH'):
            add(a, b)
    add('P2_BASE_GT', 'V2_MORPH_GT')
    return result
