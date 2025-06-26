import sys
import types
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Provide dummy statsmodels modules if not available
if 'statsmodels' not in sys.modules:
    sm = types.ModuleType('statsmodels')
    sm_api = types.ModuleType('statsmodels.api')
    sm_formula_pkg = types.ModuleType('statsmodels.formula')
    sm_formula_api = types.ModuleType('statsmodels.formula.api')
    sm_stats_pkg = types.ModuleType('statsmodels.stats')
    sm_multitest = types.ModuleType('statsmodels.stats.multitest')
    def multipletests(*args, **kwargs):
        return ([], [], [], [])
    sm_multitest.multipletests = multipletests
    sm_stats_pkg.multitest = sm_multitest
    sm.api = sm_api
    sm.formula = sm_formula_pkg
    sm.stats = sm_stats_pkg
    sm_formula_pkg.api = sm_formula_api
    sys.modules['statsmodels'] = sm
    sys.modules['statsmodels.api'] = sm_api
    sys.modules['statsmodels.formula'] = sm_formula_pkg
    sys.modules['statsmodels.formula.api'] = sm_formula_api
    sys.modules['statsmodels.stats'] = sm_stats_pkg
    sys.modules['statsmodels.stats.multitest'] = sm_multitest

# Provide dummy upsetplot if not installed
if 'upsetplot' not in sys.modules:
    up = types.ModuleType('upsetplot')
    up.plot = lambda *a, **k: plt.figure()
    sys.modules['upsetplot'] = up
