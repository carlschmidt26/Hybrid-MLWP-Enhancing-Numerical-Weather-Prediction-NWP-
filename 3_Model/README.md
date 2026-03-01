# Model Definition and Evaluation

**[Link to Notebook](model_definition_evaluation.ipynb)**

## Model Selection
- **Architecture:** Decoder-only Transformer with Causal Masking.
- **Output Layer:** Depends on the experiment at hand: Either a linear output layer matching the output variables (zonal and meridional wind) or a probabilistic head predicting Gaussian distribution parameters (μ, σ).
- **Rationale:** The Transformer's self-attention mechanism allows for capturing long-range dependencies in weather data. The different experiments are conducted to analyze the sensitivity of the model to various architectural choices or to account for inherent features of the input data (e.g., predicting a Gaussian distribution to model the uncertainty in atmospheric processes).

## Hyperparameters
| Parameter | Value |
| :--- | :--- |
| Context Window | 24h, 72h, 168h, 720h (depending on experiment) |
| Prediction Horizon | 1h (Next Token Prediction) |
| Attention Heads | 1, 4, 8, 16 (depending on experiment) |
| Loss Function | MSE or Gaussian Negative Log-Likelihood (NLL) (depending on experiment) |
| Optimizer | Adam with Learning Rate Scheduler |

## Experiments
| Experiment | Description |
| :--- | :--- |
| **Experiment A:** (TBD) | Value vs. Delta Forecasting |
| **Experiment B:** | Enhancement of the Feature Space |
| **Experiment C:** | Single-Head vs. Multi-Head |
| **Experiment D:** | Context Window Length |
| **Experiment E:** | Sequence-to-One vs. Sequence-to-Sequence | 
| **Experiment F:** | Point Forecasting (single value) vs. Probabilistic Forecasting (distribution) |

## Evaluation Metrics
- **Primary Metric:** MSE (for comparison with Baseline).
- **Secondary Metric:** MAE and Negative Log-Likelihood (NLL) for uncertainty quantification.

## Comparative Analysis
- **Baseline (Persistence) MSE:** 0.017396 (using normalized data)
- **Transformer Model MSE:** TBD
- **Improvement:** TBD
