# Baseline Model

**[Notebook](baseline_model.ipynb)**

## Baseline Model Results

### Model Selection
- **Baseline Model Type:** Persistence
- **Rationale:** The rationale behind using persistence as the baseline for evaluating MLWP models is greatly summarize by the following two quotes:</br>
  * _"Weather and atmospheric patterns are often persistent. The simplest weather forecasting method is the so-called persistence model, which assumes that the future state of a system will be similar (or equal) to the present state. Machine learning (ML) models [...] need to be compared to the persistence model to analyse whether they provide a competitive solution to the problem at hand."_</br>[Perenz-Ortiz et al., 2018](https://ieeexplore.ieee.org/document/8489179)
  * _"[...] It serves as a benchmark for evaluating the accuracy of other forecasting models, with its reliability depending on stable weather conditions."_</br>(AI generated definition based on: [Renewable and Sustainable Energy Reviews, 2018](https://www.sciencedirect.com/science/article/abs/pii/S1364032117311620) obtained via [ScienceDirect - Persistence Model](https://www.sciencedirect.com/topics/engineering/persistence-model))
### Model Performance
- **Evaluation Metric:** MSE
- **Performance Score:** 
- **Cross-Validation Score:** _-no CV applied-_

### Evaluation Methodology
- **Data Split:** _-no split applied-_
- **Evaluation Metrics:** MSE (see above)

### Metric Practical Relevance
The persistence model represents the simplest baseline in time-series forecasting. It assumes that the weather at $t+1$ is identical to the weather at $t$ ($ŷ_{t+1} = x_t$).</br>
Comparing this with the original data $y_{t+1}\equiv x_{t+1}$, yields the performance of the persistence model, e.g., in the form of $MSE = \frac{1}{N-1}\sum_{i=1}^{N}\left(y_{i}-ŷ_{i}\right)^{2}$.
Despite its simplicity, the persistence model is used as the common baseline in the climate science community. Therefore, it provides a sensible reference for the experiments conducted here, as any machine learning model must significantly outperform this baseline to prove it has captured physical dynamics rather than just temporal autocorrelation. The persistent nature of the underlying data highlighted by the high autocorrelation values found in our Exploratory Data Analysis (EDA) underscores why this specific baseline is the most challenging and relevant benchmark for this project

## Next Steps
While the persistence model provides a strong baseline for short-term forecasts, it lacks the ability to capture non-linear interactions and long-term trends. In the next phase, [Model Definition and Evaluation](../3_Model/README.md), the Transformer-based architecture is introduced. 
The primary objective is to demonstrate that the self-attention mechanism can extract meaningful physical patterns that go beyond simple temporal persistence, effectively "beating" this baseline even in highly autocorrelated weather data.
