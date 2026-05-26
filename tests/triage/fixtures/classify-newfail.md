# ❌ Regression failed

- **1223** tests passed
- **455** tests failed (447 expected, 8 new)

## ❌ New failures

These tests failed and are NOT in `.github/known-failures-d2m.txt`.
Fix the regression or, if expected, add to the allowlist.

### `test_tms.py::test_arange[ttmetal-bf16-shape0-0-1]`

Reproduce locally:
```bash
pytest tt-mlir/test/python/golden/test_tms.py::test_arange[ttmetal-bf16-shape0-0-1]
```

### `test_tms.py::test_arange[ttmetal-bf16-shape1-32-2]`

Reproduce locally:
```bash
pytest tt-mlir/test/python/golden/test_tms.py::test_arange[ttmetal-bf16-shape1-32-2]
```
