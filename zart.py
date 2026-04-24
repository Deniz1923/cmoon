import pandas

print(pandas.read_parquet(".venv/lib/python3.14/site-packages/cnlib/data/kapcoin-usd_train.parquet").to_string())
print(pandas.read_parquet(".venv/lib/python3.14/site-packages/cnlib/data/tamcoin-usd_train.parquet").to_string())
print(pandas.read_parquet(".venv/lib/python3.14/site-packages/cnlib/data/metucoin-usd_train.parquet").to_string())

