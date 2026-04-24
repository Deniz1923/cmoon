from cnlib.base_strategy import BaseStrategy

class MyStrategy(BaseStrategy):
    def predict(self, data):
        closes = data["kapcoin-usd_train"]["Close"]

        if closes.iloc[-1] > closes.iloc[-2]:
            signal = 1   # price went up → go long
        else:
            signal = -1  # price went down → go short

        return [
            {"coin": "kapcoin-usd_train",  "signal": signal, "allocation": 0.5, "leverage": 2},
            {"coin": "metucoin-usd_train", "signal": 0,      "allocation": 0.0, "leverage": 1},
            {"coin": "tamcoin-usd_train",  "signal": 0,      "allocation": 0.0, "leverage": 1},
        ]