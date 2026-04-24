import numpy as np
from sklearn.ensemble import RandomForestClassifier
from cnlib.base_strategy import BaseStrategy
from cnlib import backtest

class MLStratejim(BaseStrategy):
    def __init__(self):
        super().__init__()
        self.modeller = {}

    def egit(self):
        for coin, df in self.coin_data.items():
            X, y = self._ozellikler(df)
            model = RandomForestClassifier(random_state=42)
            model.fit(X[:-1], y[:-1])
            self.modeller[coin] = model

    def _ozellikler(self, df):
        closes = df["Close"]
        ma5 = closes.rolling(5).mean()
        ma20 = closes.rolling(20).mean()
        X = np.column_stack([ma5, ma20, closes.pct_change()])
        y = (closes.shift(-1) > closes).astype(int)
        mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
        return X[mask], y[mask]

    def predict(self, data: dict) -> list[dict]:
        decisions = []
        coins = ["kapcoin-usd_train", "metucoin-usd_train", "tamcoin-usd_train"]

        for coin in coins:
            if coin not in data or coin not in self.modeller or len(data[coin]) < 21:
                decisions.append({"coin": coin, "signal": 0, "allocation": 0.0, "leverage": 1})
                continue

            df = data[coin]
            X, _ = self._ozellikler(df)
            tahmin = self.modeller[coin].predict([X[-1]])[0]

            if tahmin == 1:
                decisions.append({"coin": coin, "signal": 1, "allocation": 0.3, "leverage": 1})
            else:
                decisions.append({"coin": coin, "signal": 0, "allocation": 0.0, "leverage": 1})

        return decisions

if __name__ == "__main__":
    strategy = MLStratejim()
    strategy.get_data()
    strategy.egit()
    result = backtest.run(strategy=strategy, initial_capital=3000.0)
    result.print_summary()