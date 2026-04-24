# Tests

Contract tests belong here once implementation starts.

Initial coverage should assert:
- market data schema
- auto-discovered symbol registry
- UTC-aware monotonic indexes
- unique `symbol + timestamp` rows
- feature columns prefixed with `feat_`
- label and feature alignment
- purged walk-forward split gaps
- no date slicing outside `src/data_loader.py`
- no hardcoded competition coin names outside config fixtures
- next-bar-open backtest fills
