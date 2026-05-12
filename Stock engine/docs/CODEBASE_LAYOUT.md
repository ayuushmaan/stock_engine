# Codebase Layout

This project is now split into three clear areas so the historical-training flow and the live-prediction flow stay separate.

## Folders

### `SRC/past_data/`
- Historical data generation
- Model training on past sentiment + price behavior
- Backtesting, research evaluation, and paper tables

Files:
- `historical_data_generator.py`
- `ml_model_trainer.py`
- `backtester.py`
- `research_evaluation.py`
- `paper_tables.py`

### `SRC/live_data/`
- Daily news fetch
- Live sentiment scoring
- Daily market price fetch
- Live prediction/evaluation pipeline

Files:
- `Sentiment.py`
- `Scanner.py`
- `data_loader2.py`
- `Predictions.py`

### `SRC/common/`
- Shared paths
- Shared database schema
- Shared scoring utilities
- Shared trained-parameter loader used by live prediction

Files:
- `paths.py`
- `db_schema.py`
- `scoring_engine.py`
- `model_config.py`

## Entry Points

- `app.py`: live daily pipeline
- `train_ml_model.py`: historical training pipeline
- `quick_demo_ml.py`: small historical training demo
- `scripts/init_v3.py`: shared DB/scoring setup

## Flow

1. Historical training runs in `SRC/past_data/` and writes learned parameters to `ml_trained_config.py`.
2. Live prediction runs in `SRC/live_data/`.
3. The live pipeline loads trained values from `ml_trained_config.py` through `SRC/common/model_config.py`.
4. This keeps the future dashboard path clean: train on the past, apply on live data.
