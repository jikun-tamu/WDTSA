# Power Outage Prediction Model (WDTSA)

This repository contains the implementation of the Wide-and-Deep-Based Time Sequence Algorithm (WDTSA) for predicting power outage severity caused by extreme winter storms. The model combines temporal and static feature processing through specialized pathways to achieve high-accuracy outage forecasting.

## Contents

- **[`train_data_reconstructed.csv`](train_data_reconstructed.csv )**: Processed dataset containing time-lagged weather measurements and outage percentages for Texas counties during February 2021.
- **[`main.py`](main.py )**: Core implementation of the WDTSA model architecture, feature routing, and evaluation frameworks.
- **[`model_train.py`](model_train.py )**: Training pipeline with reproducible experiment settings and model persistence.
- **[`linear.py`](linear.py )**: Implementation of linear baseline models for comparison.
- **[`widedeep_K10_run6_acc0_9519.pt`](widedeep_K10_run6_acc0_9519.pt )**: Pre-trained WDTSA model with 10 outage severity classes (95.19% validation accuracy).
- **[`all_predictions_K10_run6_acc0_9519.csv`](all_predictions_K10_run6_acc0_9519.csv )**: Model predictions for Winter Storm Uri period.

## Model Architecture

The WDTSA architecture features two specialized pathways:

1. **Deep Path**: Processes temporal features through a bidirectional LSTM (16 units per direction)
2. **Wide Path**: Handles static features via dense layers (64 units)

These pathways are fused through concatenation and additional dense layers to produce outage severity classifications.

## Usage
### Training a New Model

```python
from model_train import train_and_save_model

# Train model with 10 severity classes
train_and_save_model(n_runs=5, n_classes=10)
```

### Making Predictions

```python
import torch
from main import WideAndDeepClassifier, prepare_model_inputs, get_routed_columns, FEATURE_GROUPS

# Load pre-trained model
model = WideAndDeepClassifier(time_steps=10, deep_features=5, wide_features=20, 
                             hidden_size=64, lstm_units=16, num_classes=10)
model.load_state_dict(torch.load('widedeep_K10_run6_acc0_9519.pt'))
model.eval()

# Load and preprocess data
df = pd.read_csv('new_data.csv')
deep_cols, wide_cols = get_routed_columns(df.columns, FEATURE_GROUPS)
x_deep, x_wide = prepare_model_inputs(df, deep_cols, wide_cols)

# Make predictions
with torch.no_grad():
    x_deep_tensor = torch.tensor(x_deep, dtype=torch.float32)
    x_wide_tensor = torch.tensor(x_wide, dtype=torch.float32)
    outputs = model(x_deep_tensor, x_wide_tensor)
    _, predicted = torch.max(outputs, 1)
```

## Performance

The included pre-trained model (`widedeep_K10_run6_acc0_9519.pt`) achieves **95.19% validation accuracy** with 10 severity classes. Performance across various model configurations:

| Classes | WDTSA Accuracy | Linear Model | Advantage |
|---------|---------------|--------------|-----------|
| 3       | 99.4%         | 99.0%        | 0.4%      |
| 5       | 97.0%         | 95.8%        | 1.2%      |
| 7       | 93.4%         | 89.0%        | 4.4%      |
| 10      | 89.6%         | 77.2%        | 12.4%     |
| 15      | 84.4%         | 68.7%        | 15.7%     |

## Feature Routing

Features are routed to either the deep or wide pathway based on their temporal characteristics:

- **Deep features**: Temperature, humidity, outage history, dew point
- **Wide features**: Barometric pressure, wind gust, latitude/longitude

## Citation

If you use this code in your research, please cite:

```
@article{liu2025widedeep,
  title={A Wide-and-Deep-Based Time Sequence Model for Predicting Power Outages Caused by Extreme Winter Storms},
  author={Liu, Jikun and Cheng, Yuhan and Lee, Jangjae and Paal, Stephanie and Li, Diya and Zhang, Zhe},
  journal={},
  year={2025}
}
```

## Data Availability

The processed dataset (`train_data_reconstructed.csv`) was derived from raw power outage data obtained from the DOE Eagle-I system and historical weather data retrieved from the Weather API (weatherapi.com). Researchers wishing to access the original raw data should refer to the Eagle-I system and Weather API documentation for data retrieval protocols.

## License

MIT License