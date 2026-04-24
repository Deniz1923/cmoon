from cnlib import backtest
from zart import MyStrategy

result = backtest.run(MyStrategy(), initial_capital=3000.0)
result.print_summary()