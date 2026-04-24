# Yarismaci Starter (Jenerik Mimari)

Bu repo, yarismaci olarak strateji gelistirmen icin tek paketli ve jenerik bir yapi sunar. Kurallar degisse bile kodun bozulmadan devam etmesi icin kural-seti `configs/competition.json` dosyasindan okunur.

## Kurulum (sadece uv)

```bash
uv sync
```

## Mimari

```text
.
├── configs/
│   └── competition.json          # Yarismaya ozel kurallar
├── data/
│   ├── train/*.parquet
│   └── validation/*.parquet
├── scripts/
│   └── generate_dummy_data.py    # Rules'a gore dummy veri uretir
├── src/
│   ├── competition.py            # Rule model + JSON loader
│   ├── strategy_base.py          # BaseStrategy
│   ├── validation.py             # predict() cikti dogrulama
│   ├── data_loader.py            # Parquet yukleme
│   ├── backtest.py               # PnL/fee/liquidation engine
│   ├── evaluation.py             # Metrik hesaplama
│   ├── output_writer.py          # Rapor/CSV yazimi
│   ├── runner.py                 # Uctan uca local run
│   ├── strategy_loader.py        # Strateji class yukleyici
│   ├── indicators.py             # SMA/EMA/RSI yardimcilari
│   ├── model_types.py            # Ortak tipler
│   └── constants.py              # Defaults
├── strategy.py                   # Senin stratejin
└── run_validation.py             # CLI entrypoint
```

## Separation of Concerns (3 kisilik ekip)

### Track A — Kurallar ve Domain Sozlesmeleri
- `competition.py`
- `model_types.py`
- `strategy_base.py`
- `validation.py`

Sorumluluk:
- Yarismaya ozel kural modelini tanimlamak
- Strateji cikti kontratini korumak
- Signal dogrulamayi merkezi tutmak

### Track B — Veri ve Entegrasyon
- `data_loader.py`
- `strategy_loader.py`
- `runner.py`
- `scripts/generate_dummy_data.py`
- `run_validation.py`

Sorumluluk:
- Veri yukleme ve format tutarliligi
- Stratejinin dinamik import edilmesi
- Uctan uca pipeline orkestrasyonu

### Track C — Simulasyon ve Raporlama
- `backtest.py`
- `evaluation.py`
- `output_writer.py`
- `indicators.py`

Sorumluluk:
- Simulasyon mekanigi (PnL, fee, liquidation)
- Performans metriklerinin hesaplanmasi
- Sonuc artefaktlarinin yazimi

### Katman Kurali
- `runner.py` disindaki moduller sadece kendi concern'ine ait is yapar.
- Simulasyon katmani (`backtest.py`) dosya yazmaz.
- Raporlama katmani (`output_writer.py`) PnL hesaplamaz.
- Domain kurallari (`competition.py`, `validation.py`) IO yapmaz.

## Kural Dosyasi (competition.json)

Temel alanlar:

- `coins`
- `allowed_leverages`
- `allow_long`, `allow_short`
- `max_total_ratio`, `max_ratio_per_coin`
- `initial_equity`, `fee_rate`, `min_history`

Bu dosya degistiginde motor yeni kurallarla calisir.

## Veri Formati

Her coin icin bir parquet dosyasi:

- `data/train/<coin>.parquet`
- `data/validation/<coin>.parquet`

Beklenen kolonlar:

- `open`, `high`, `low`, `close`, `volume`

Opsiyonel index kolonlari:

- `timestamp` veya `time_index`

## Strateji Arayuzu

`strategy.py` icinde `BaseStrategy` sinifini extend et:

```python
class MyStrategy(BaseStrategy):
    def fit(self, train_data):
        ...

    def predict(self, current_window, state):
        return [
            {"coin": "Varlik_A", "signal": 1, "ratio": 0.3, "leverage": 3},
            {"coin": "Varlik_B", "signal": 0, "ratio": 0.0, "leverage": 2},
            {"coin": "Varlik_C", "signal": -1, "ratio": 0.2, "leverage": 5}
        ]
```

Kurallar:

- `signal`: `1` long, `-1` short, `0` hold
- `ratio`: coin'e ayirilan sermaye orani
- toplam ratio `<= max_total_ratio`
- `leverage`: `allowed_leverages` icinden olmali

## Calistirma

1) Dummy veri uret (gercek veri yoksa):

```bash
uv run python scripts/generate_dummy_data.py --rules configs/competition.json
```

2) Validation backtest:

```bash
uv run python run_validation.py \
  --rules configs/competition.json \
  --strategy-file strategy.py \
  --class-name MyStrategy \
  --train-dir data/train \
  --validation-dir data/validation \
  --out-dir outputs
```

## Ciktilar

- `outputs/report.json`
- `outputs/equity_curve.csv`
- `outputs/step_log.csv`

`report.json` hem performans metriklerini hem de kosulan kural/config bilgisini yazar.
