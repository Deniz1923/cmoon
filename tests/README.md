# Tests

Contract tests belong here once implementation starts.

Initial coverage should assert:
- market data schema
- UTC-aware monotonic indexes
- feature columns prefixed with `feat_`
- label and feature alignment
- purged walk-forward split gaps
- no date slicing outside `src/data_loader.py`
- next-bar-open backtest fills
