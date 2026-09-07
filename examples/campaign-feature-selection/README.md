# Campaign feature selection snapshot

Vendor copy of the campaign LGBM + SHAP selectors used to compare against
fmlib LightGBM (`vote` / `min_set_share: 1.0`, threshold 0.85).

This is **not** part of the fmlib pipeline. Import from here only in comparison
notebooks:

```python
from utils.feature_selection import LGMFeatureSelection, ShapFeatureSelection
```

`utils/__init__.py` also imports `uplift_clf_metrics`, which needs `scikit-uplift`
(`import sklift`). For selection only, import `utils.feature_selection` after the
package is on `sys.path`, or load that module by file path.
