from scripts.smoke import make_synthetic_data
from submission import strateji


def test_submission_contract_first_and_last_window() -> None:
    data = make_synthetic_data(rows=80)
    strateji.fit(data)

    for window in (1, 80):
        orders = strateji.predict({asset: frame.iloc[:window] for asset, frame in data.items()})
        assert len(orders) == 3
        assert sum(float(order["oran"]) for order in orders) <= 1.000001
        for order in orders:
            assert set(order) == {"sinyal", "oran", "kaldirac"}
            assert order["sinyal"] in {-1, 0, 1}
            assert 0.0 <= float(order["oran"]) <= 1.0
            assert order["kaldirac"] in {2, 3, 5, 10}
