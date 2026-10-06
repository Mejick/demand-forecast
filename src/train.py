"""train one model on the CV folds

python train.py features
python train.py rules
python train.py ets
python train.py theta
python train.py glm_offset
python train.py lgbm
python train.py lgbm_noitem
python train.py catboost
python train.py nhits
python train.py deepar
"""
import sys

import pandas as pd

from common import FEAT, ROOT, build_table
from cv import load_matrices


def main(what):
    if what == "features":
        return build_table()
    M = load_matrices()
    if what == "rules":
        from models import rules
        return rules.run(M)
    if what in ("ets", "theta"):
        from models import statistical
        r = statistical.run(M, "ets" if what == "ets" else "more")
    elif what in ("nhits", "deepar"):
        from models import neural
        r = neural.run(M, what)
    else:
        t = pd.read_parquet(FEAT)
        if what in ("glm", "glm_offset"):
            from models import linear
            r = linear.run(t, M, offset=what == "glm_offset")
        elif what in ("lgbm", "lgbm_noitem"):
            from models import lgbm
            r = lgbm.run(t, M, drop=("item_id",) if what == "lgbm_noitem" else (), name=what)
        elif what == "catboost":
            from models import catboost_model
            r = catboost_model.run(t, M)
        else:
            raise SystemExit(__doc__)
    r.to_csv(ROOT / "reports" / f"{what}_folds.csv", index=False)
    print(r[["WRMSSE", "RMSSE_mean", "MAE", "bias"]].mean().round(4).to_dict())


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "")
