# Enhanced Solar Energy Intelligence Suite

This repository bundles multiple deep-learning and classical ML pipelines for satellite-aware solar analytics. It focuses on turning real-time weather feeds, NASA/NOAA irradiance data, and Google Earth Engine (GEE) imagery into accurate solar irradiance and energy predictions, plus cloud detection tooling to improve production estimates.

## Highlights
- **Full solar workflow**: irradiance prediction, energy output forecasting, cloud detection/forecasting, and Google Earth Engine integrations.
- **Hybrid data**: live APIs (OpenWeatherMap, NREL, NASA POWER), synthetic data generators, and Sentinel/Landsat imagery through GEE.
- **Production-oriented**: configurable CLI prompts, realistic defaults for South/Southeast Asia, automatic fallbacks when APIs are unavailable.
- **Explainability and reporting**: charts, confidence distributions, daily breakdowns, and contextual recommendations per location.

## Repository Structure

| File | Purpose |
| --- | --- |
| `solar_energy_output_prediction.py` | Flagship LSTM + attention model with interactive CLI for end-to-end forecasting. |
| `solar_irradiance_prediction.py` | Gradient boosted tree models for irradiance estimation. |
| `irradiance_prediction.py` | Weather-driven baselines for irradiance forecasting. |
| `cloud_detection.py` | U-Net segmentation pipeline for satellite cloud masks. |
| `cloud_forecasting.py` | ConvLSTM-based short-term cloud motion forecaster. |
| `gee_config.py` / `gee_config_simple.py` | Centralized Google Earth Engine initialization utilities. |
| `test_*.py` | Smoke tests and integration checks for the main modules. |
| `requirements.txt` | Locked dependency list for local development. |

## Requirements
- Python 3.10+ (tested on Windows 10)
- Optional CUDA-enabled GPU for faster training
- GEE-enabled Google Cloud project (default: `instant-text-459407-v4`)
- API keys if you want real weather/irradiance data:
  - OpenWeatherMap (One Call 3.0)
  - NREL NSRDB
  - NASA POWER (no key required, but rate limiting applies)

## Environment Setup
```bash
python -m venv .venv
.venv\Scripts\activate  # On PowerShell
pip install --upgrade pip
pip install -r requirements.txt
```

### Google Earth Engine Authentication
```bash
# Service account (recommended for automation)
set GOOGLE_APPLICATION_CREDENTIALS=path\to\service-account.json

# User account (interactive)
earthengine authenticate
```

## Running the Solar Energy Output Predictor
```bash
python solar_energy_output_prediction.py
```

Interactive prompts guide you through:
1. **Custom location** – decimal-degree coordinates and optional labels, with climate zone detection.
2. **Time configuration** – training window, forecast horizon (1–168 h), and specific-date target.
3. **System parameters** – capacity (kW), panel area (m²), training epochs, learning rate.
4. **API keys** – supply OpenWeatherMap/NREL keys or rely on realistic synthetic generation.

The script then:
- Retrieves or synthesizes weather + irradiance histories.
- Engineers irradiance, weather, temporal, and panel metadata features.
- Trains an attention-enhanced LSTM on sliding time-window datasets.
- Generates forward irradiance projections and energy forecasts.
- Prints human-readable summaries (daily totals, confidence histograms, weather context, revenue estimates).

### Key Internals
- `SolarPanelSpecs`, `EnergyPrediction`, `SystemPerformance`, and `TrainingDataPoint` dataclasses keep metadata consistent.
- `FeatureEngineer` normalizes datasets and derives panel-adjusted features (temperature coefficients, soiling factors, tilt/azimuth effects).
- `LSTMPredictor` uses attention pooling to emphasize the most informative time steps.
- Real-data integration happens through `RealDataAPIs`, abstracting OpenWeatherMap, NREL NSRDB, and NASA POWER calls with synthetic fallbacks.

## Other Notable Pipelines
- **`solar_irradiance_prediction.py`**: Offers regression models (XGBoost/Random Forest) trained on API + satellite-derived irradiance features.
- **`cloud_detection.py`**: U-Net segmentation for satellite imagery; supports training with GEE-sourced chips plus local augmentations.
- **`cloud_forecasting.py`**: ConvLSTM sequences for short-term cloud movement, enabling irradiance nowcasting.
- **`cloud_irradiance` utilities** (in `cloud_*` / `cloud_segmentation.py` if present) feed cloud-cover percentages back into energy predictors for dynamic derates.

## Configuration & Customization
- `gee_config.py`: set your Google Cloud project ID and dataset defaults.
- `setup_gee.py`: helper for bootstrapping Earth Engine in fresh environments.
- `requirements.txt`: includes PyTorch, scikit-learn, pandas, structlog, geemap, Google API clients, etc.
- Dark-theme UI preference is already respected in interactive prompts for main/login/register screens (per project conventions).

## Testing
Basic test entry points:
```bash
python -m pytest test_all_models.py
python test_cloud_detection.py
python test_gee_integration.py
```
Tests validate data loaders, model initialization, and GEE configuration fallbacks. Expand with scenario-specific cases before deployment (e.g., long-horizon inference, missing sensor data).

## Troubleshooting
- **Authentication errors**: run `earthengine authenticate --clear`, then re-auth.
- **API quota exceedance**: reduce polling frequency or schedule jobs off-peak.
- **Slow training**: lower `sequence_length`, disable attention, or leverage GPU acceleration.
- **Unstable forecasts**: ensure input datasets include required columns (`irradiance`, `temperature`, `humidity`, `wind_speed`). The scripts auto-fill defaults and log warnings when they patch missing features.

## Roadmap Ideas
- Export-ready JSON/CSV summaries from CLI runs.
- RESTful microservice wrapper for `SolarEnergyPredictor`.
- Integration with weather forecast providers (ECMWF/GFS) for longer horizons.
- Automated model retraining with new GEE imagery batches.

## License & Support
Distributed under the MIT License. For GEE usage questions, see the [official docs](https://developers.google.com/earth-engine) and community forum. For model-specific issues, open an issue or consult the individual scripts for implementation notes.

---
Feel free to adapt the CLI prompts, model architectures, or data feeds to fit your geography, plant design, or deployment environment.
"# GIS-ML-Intelligence" 
