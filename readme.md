# Time-series Forecasting: Are (Cross-)Attentions Necessary?

## *Authors*

* Paweł Gacek
* Dawid Wołek


# Introduction
Time series forecasting is a fundamental problem in machine learning and statistics, with applications spanning finance, energy management, environmental monitoring, healthcare, and beyond. The goal is to predict future values of a sequence based on historical observations, capturing temporal dependencies, trends, seasonal patterns, and complex non-linear relationships within the data. As datasets grow larger and forecasting horizons extend further into the future, the challenge intensifies—requiring models that can generalize across diverse domains while remaining computationally efficient.

## The Rise of Transformer-Based Approaches

In recent years, Transformer architectures, originally designed for natural language processing, have been adapted to time series forecasting with considerable success. Models such as **Autoformer**, **FEDformer**, and **PatchTST** leverage self-attention mechanisms to capture long-range dependencies by computing relationships between all time steps in a sequence. These architectures often incorporate domain-specific modifications.

The appeal of Transformers lies in their ability to model complex, non-local interactions in data. However, this expressiveness comes at a cost: increased computational overhead, larger memory footprints, and greater susceptibility to overfitting, particularly when training data is limited.

## Challenging the Necessity of Attention: Simple Linear Models

In contrast to the complexity of Transformers, **DLinear** presents a radically simpler alternative. DLinear avoids attention mechanisms entirely, instead employing a decomposition-based approach where the input time series is split into trend and seasonal components using a moving average filter. Each component is then processed by a single-layer linear model, and the results are summed to produce the final forecast. Because of its simplicity, DLinear has demonstrated surprisingly competitive performance on standard benchmarks, often matching or exceeding Transformer-based models while requiring orders of magnitude fewer parameters and less computation.

This raises a provocative question: **Are attention mechanisms truly necessary for effective long-term time series forecasting?** DLinear's success suggests that explicit temporal modeling through linear projections, combined with principled decomposition, may be sufficient to capture the dependencies needed for accurate predictions.


# Datasets

To evaluate the long-term forecasting capabilities of state-of-the-art models such as DLinear, PatchTST, CATS, and others, we conduct experiments on six diverse and under-explored real-world time series datasets. These datasets cover a range of domains, including renewable energy, financial markets, industrial sales, and household utility usage, presenting varied temporal dynamics, sampling resolutions, noise characteristics, and forecasting challenges. Using less commonly benchmarked datasets helps assess how well modern forecasting models generalize beyond standard public benchmarks.

Below, we describe each dataset individually, outlining its source, key attributes, and the forecasting setup used in our experiments.

## Air Pollution

The air pollution dataset we use contains hourly measurements of air quality indicators, most notably PM2.5 concentration, which is a key metric for particulate pollution. This dataset comes from a Kaggle collection designed for forecasting tasks and captures real-world temporal dynamics in atmospheric pollution levels with 20,976 samples. It exhibits strong seasonal and diurnal patterns, and challenges models with noise, missing values, and varied autocorrelation structures common to environmental data. Because pollution levels respond to both local emissions and weather conditions, this dataset tests each model's ability to capture both short-term variability and long-term trends.

**Source:** [Kaggle - LSTM Datasets Multivariate Univariate](https://www.kaggle.com/datasets/rupakroy/lstm-datasets-multivariate-univariate/data)

## Wind Power Generation

The wind power generation dataset records power output measurements from wind turbines,  sampled at hourly intervals, comprising 43,800 samples. It provides a realistic example of energy time series where forecasting is critical for grid planning and renewable integration. This dataset is multivariate and contains different operational and environmental variables (e.g., wind speed) that drive power production. Long-term forecasting here is challenging due to the combination of natural variability in wind patterns and engineered system responses, making it an ideal benchmark for models designed to capture complex temporal dependencies.

**Source:** [Kaggle - Wind Power Generation Data Forecasting](https://www.kaggle.com/datasets/mubashirrahim/wind-power-generation-data-forecasting)

## Microsoft Stock

The Microsoft stock dataset contains historical financial time series data for MSFT shares, including prices (open, high, low, close) and volume, with 9,083 samples. Unlike many benchmark finance datasets that focus on popular indices, this dataset allows evaluation of forecasting models in a noisy, volatile domain where patterns are subtle and influenced by market dynamics. Financial time series are typically non-stationary, with changing volatility, trends, and shocks, challenging models to generalize beyond short-term dependencies.

**Source:** [Kaggle - Microsoft Stock Data](https://www.kaggle.com/datasets/varpit94/microsoft-stock-data)

## Household Power Consumption (HPC)

The Household Power Consumption dataset is a classic multivariate time series from the UCI Machine Learning Repository that records electrical usage and related variables for a single household. With measurements such as global active power, voltage, and sub-metering, sampled at a high (minute-level) frequency over multiple years, it provides a rich testbed for forecasting models capable of leveraging long sequences and intra-day patterns. The dataset contains 34,589 samples and its high resolution and real-world missing values make it a strong candidate for evaluating long-horizon forecasting performance.

**Source:** [UCI Machine Learning Repository - Individual Household Electric Power Consumption](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption)

## QPS 

The QPS dataset (from the Kaggle collection of multivariate time series forecasting datasets) includes several real-world examples curated for forecasting tasks across domains. In our study, we use the portion labeled QPS, which contains multiple signals with varying temporal correlations and comprises 30,240 samples, enabling assessment of how models handle multivariate interactions and cross-series dependencies. This dataset helps probe whether models can exploit inter-feature relationships effectively for long-term forecasting.

**Source:** [Kaggle - Datasets for Multivariate Time Series Forecasting](https://www.kaggle.com/datasets/limpidcloud/datasets-for-multivariate-time-series-forecasting)

## Sales (Pharma Sales)

The pharmaceutical sales dataset contains hourly sales records for products, offering a fine-grained view of demand patterns over time with 50,532 samples. It combines regular seasonal effects (e.g., daily or weekly purchase cycles) with random fluctuations in sales volume, which is characteristic of many real retail time series. Because forecasting future sales accurately can have direct business impact (e.g., inventory planning), this dataset provides a practical setting for evaluating model performance in economic demand forecasting.

**Source:** [Kaggle - Pharma Sales Data](https://www.kaggle.com/datasets/milanzdravkovic/pharma-sales-data?select=saleshourly.csv)


# Models

We evaluate a diverse set of time series forecasting models, ranging from simple statistical baselines to recent deep learning architectures specifically designed for long-term forecasting. This selection allows us to analyze model performance across different levels of complexity and inductive biases, and to better understand the trade-offs between simplicity, interpretability, and predictive accuracy.

Our experiments include a naive baseline, which serves as a reference point for measuring the added value of more sophisticated approaches, as well as several state-of-the-art neural forecasting models that have demonstrated strong performance on long-horizon time series tasks. The chosen models represent different modeling paradigms, including decomposition-based methods, linear forecasting, convolutional architectures, and Transformer-based approaches. 

Below, we briefly describe each model used in our study.

## Naive Last Value (Repeat-C)
The Naive Last Value baseline, also referred to as Closest Repeat (Repeat-C), is a simple forecasting method that predicts all future time steps by repeating the last observed value from the look-back window. Despite its simplicity, this baseline provides a strong reference point for evaluating long-term forecasting models.

## Autoformer
Autoformer is a Transformer-based model specifically designed for long-term time series forecasting. It introduces a series decomposition mechanism that explicitly separates the input sequence into trend and seasonal components, allowing the model to better capture long-term patterns. In addition, Autoformer replaces standard self-attention with an auto-correlation mechanism, which focuses on discovering periodic dependencies in the time series while reducing computational complexity.
In our experiments, Autoformer is also used as a baseline for DLinear, as DLinear adopts the same decomposition strategy while replacing the nonlinear forecasting modules with simple linear projections.

## FEDformer
FEDformer extends Autoformer by performing time series modeling in the frequency domain, enabling more efficient and expressive long-term forecasting. It applies frequency-enhanced decomposition and leverages Fourier- or wavelet-based representations to capture global temporal patterns while reducing redundancy in the attention mechanism. By modeling dominant frequency components, FEDformer improves both computational efficiency and forecasting accuracy on long sequences. Similar to Autoformer, it builds upon series decomposition principles, making it a strong Transformer-based baseline for long-horizon forecasting.

## N-HiTS

N-HiTS (Neural Hierarchical Interpolation for Time Series Forecasting) is a deep learning architecture designed to address the challenges of long-horizon forecasting by processing information at multiple temporal scales. It evolves from the N-BEATS architecture, utilizing a hierarchical structure of neural blocks that decompose the signal into different frequencies. The model employs a multi-rate sampling technique to reduce the dimensionality of the input and a hierarchical interpolation mechanism to ensure that each block focuses on a specific scale—ranging from coarse long-term trends to fine-grained short-term fluctuations. This approach significantly reduces computational costs while mitigating the "volatility" often found in long-term point predictions, allowing N-HiTS to outperform many Transformer-based models in both accuracy and efficiency.

## DLinear
DLinear is a lightweight linear model for long-term time series forecasting that challenges the necessity of complex Transformer-based architectures. While many forecasting models rely on self-attention mechanisms to capture long-range dependencies, DLinear demonstrates that explicit temporal modeling via simple linear projections can be sufficient for long-horizon forecasting. It adopts the series decomposition framework introduced in Autoformer and FEDformer, separating the input into trend and seasonal components using a moving average filter. Each component is then processed by a one-layer linear model, and the results are summed to form the final prediction

## CATS
CATS (Cross-Attention-only Time Series transformer) is a streamlined architecture that rethinks the role of attention in forecasting by eliminating self-attention entirely. The model addresses "temporal information loss" caused by the permutation-invariant nature of self-attention, which can hinder the capture of precise temporal orders. Instead, CATS establishes future horizons as learnable queries and treats historical data as keys and values within a cross-attention-only framework. By leveraging parameter sharing across horizons and a unique query-adaptive masking technique, CATS significantly reduces memory usage and parameter counts while outperforming both complex Transformers and lightweight models like DLinear.

## PatchTST
PatchTST is a novel time-series forecasting model that leverages patch-based attention mechanisms to capture temporal dependencies and improve prediction accuracy. It efficiently processes large datasets by dividing them into smaller patches, allowing for better handling of complex patterns and trends in time-series data.
This approach not only enhances the model's ability to learn from diverse temporal patterns but also significantly reduces computational overhead. By focusing on localized patches, PatchTST can adaptively learn from varying time scales, making it particularly effective for datasets with irregular sampling rates. The model's architecture also allows for integration with other forecasting techniques.

## Temporal Fusion Transformer (TFT)
TFT combines sequence-to-sequence LSTMs with attention and gating to fuse static covariates, known future inputs, and observed history. It uses variable selection networks to pick useful signals per time step, a temporal attention module to focus on relevant horizons, and gating layers to skip components when they are not helpful, aiming for interpretability and flexible handling of mixed inputs.

We briefly tested TFT onn few datasets and its MSE/MAE were several times worse than even the naive baseline, and training took hours. Because of the poor accuracy and prohibitive runtime, we stopped further experiments and did not include TFT in the reported results or ranking tables.


# Setup and Used Metrics

## Experimental Environment

To ensure reproducibility and consistency across all experiments, we conducted our evaluation using Google Colab with an NVIDIA Tesla T4 GPU as the primary computational environment. This cloud-based setup provides standardized hardware access and eliminates variability due to local system configurations.

For each model, we utilized the official implementations provided by the original authors in their respective repositories. This approach maximizes reproducibility and ensures that our results reflect the models as intended by their creators, rather than introducing biases through custom reimplementations. Each model was run in its own environment with dependencies and configurations specified by the authors.

## Training Configuration

We adhered to the default hyperparameters and training procedures recommended in each model's original paper and codebase. This includes learning rates, batch sizes, number of training epochs, optimizer settings, and any model-specific design choices (e.g., number of layers, hidden dimensions, attention heads).

## Forecasting Task Setup

For all experiments, we performed multivariate-to-multivariate forecasting, meaning that the models take as input multiple variables from the historical window and predict multiple variables for the future horizon.

We evaluated each model across four standard forecasting horizons:
- **96 time steps**
- **192 time steps**
- **336 time steps**
- **720 time steps**


## Evaluation Metrics

We use two widely adopted metrics for quantitative evaluation:

### Mean Squared Error (MSE)

MSE measures the average squared difference between predicted and actual values:

$$
\text{MSE} = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2
$$

where $y_i$ is the true value and $\hat{y}_i$ is the predicted value. MSE is sensitive to large errors, making it useful for identifying models that produce occasional large mispredictions.

### Mean Absolute Error (MAE)

MAE measures the average absolute difference between predicted and actual values:

$$
\text{MAE} = \frac{1}{N} \sum_{i=1}^{N} |y_i - \hat{y}_i|
$$

MAE is more robust to outliers than MSE and provides an interpretable measure of average prediction error in the same units as the original data.


## Reproducibility

All experiments are fully reproducible using the provided Jupyter notebooks and shell scripts included in this repository. Each model's setup, dataset preprocessing, training, and evaluation steps are documented in the **"How to reproduce?"** section of this README.


# Results

The following table presents a comprehensive comparative analysis of various long-term time series forecasting models across diverse real-world datasets.

| Dataset               | Metric | MSE (DLinear) | MAE (DLinear) | MSE (CATS) | MAE (CATS) | MSE (PatchTST) | MAE (PatchTST) | MSE (NHITS) | MAE (NHITS) | MSE (FEDformer) | MAE (FEDformer) | MSE (Autoformer) | MAE (Autoformer) | MSE (Naive  Last) | MAE (NaiveLast Value) |
|-----------------------|--------|---------------|---------------|------------|------------|----------------|----------------|-------------|-------------|-----------------|-----------------|------------------|------------------|------------:|------------:|
| Air Pollution         | 96     | 0.122         | 0.250         | 0.122      | 0.243      | 0.131          | 0.255          | 0.158       | 0.284       | 0.151           | 0.287           | 0.188            | 0.319            | 0.448       | 0.461       |
|                       | 192    | 0.149         | 0.277         | 0.152      | 0.273      | 0.163          | 0.283          | 0.214       | 0.325       | 0.182           | 0.313           | 0.194            | 0.319            | 0.482       | 0.484       |
|                       | 336    | 0.175         | 0.306         | 0.177      | 0.293      | 0.194          | 0.307          | 0.232       | 0.337       | 0.211           | 0.341           | 0.268            | 0.386            | 0.516       | 0.505       |
|                       | 720    | 0.229         | 0.356         | 0.247      | 0.346      | 0.267          | 0.356          | 0.281       | 0.392       | 0.236           | 0.361           | 0.281            | 0.396            | 0.579       | 0.538       |
| Microsoft Stock       | 96     | 2.468         | 1.013         | 0.797      | 0.572      | 1.248          | 0.692          | 0.926       | 0.703       | 2.361           | 1.009           | 2.440            | 1.016            | 1.218       | 0.688       |
|                       | 192    | 3.470         | 1.215         | 3.270      | 1.168      | 3.088          | 1.082          | 1.246       | 0.819       | 4.311           | 1.372           | 4.677            | 1.415            | 3.109       | 1.109       |
|                       | 336    | 7.631         | 1.693         | 11.031     | 2.256      | 5.894          | 1.531          | 1.343       | 0.835       | 8.188           | 1.887           | 8.249            | 1.844            | 6.606       | 1.659       |
|                       | 720    | 21.068        | 2.944         | 22.730     | 3.265      | 18.673         | 2.834          | 1.410       | 0.884       | 19.593          | 2.955           | 20.534           | 2.996            | 19.435      | 2.926       |
| Wind Power Generation | 96     | 0.702         | 0.651         | 0.696      | 0.643      | 0.760          | 0.674          | 0.831       | 0.691       | 0.762           | 0.677           | 0.783            | 0.691            | 1.325       | 0.852       |
|                       | 192    | 0.736         | 0.675         | 0.736      | 0.672      | 0.810          | 0.703          | 0.898       | 0.734       | 0.819           | 0.707           | 0.848            | 0.720            | 1.425       | 0.904       |
|                       | 336    | 0.754         | 0.688         | 0.754      | 0.684      | 0.851          | 0.725          | 0.917       | 0.745       | 0.865           | 0.729           | 0.850            | 0.726            | 1.496       | 0.937       |
|                       | 720    | 0.775         | 0.705         | 0.777      | 0.704      | 0.878          | 0.743          | 0.867       | 0.730       | 0.875           | 0.737           | 0.867            | 0.736            | 1.528       | 0.957       |
| HPC                   | 96     | 0.608         | 0.513         | 0.606      | 0.508      | 0.638          | 0.524          | 0.658       | 0.482       | 0.633           | 0.535           | 0.672            | 0.567            | 1.340       | 0.715       |
|                       | 192    | 0.622         | 0.523         | 0.631      | 0.524      | 0.665          | 0.538          | 0.689       | 0.494       | 0.656           | 0.556           | 0.702            | 0.580            | 1.363       | 0.727       |
|                       | 336    | 0.627         | 0.527         | 0.642      | 0.532      | 0.681          | 0.544          | 0.695       | 0.499       | 0.664           | 0.556           | 0.699            | 0.583            | 1.376       | 0.734       |
|                       | 720    | 0.626         | 0.530         | 0.643      | 0.536      | 0.679          | 0.551          | 0.710       | 0.508       | 0.667           | 0.562           | 0.687            | 0.569            | 1.359       | 0.734       |
| QPS                   | 96     | 0.090         | 0.198         | 0.041      | 0.116      | 0.063          | 0.158          | 0.043       | 0.119       | 0.086           | 0.218           | 0.363            | 0.432            | 0.153       | 0.252       |
|                       | 192    | 0.230         | 0.340         | 0.085      | 0.172      | 0.207          | 0.296          | 0.084       | 0.175       | 0.185           | 0.319           | 0.308            | 0.419            | 0.426       | 0.661       |
|                       | 336    | 0.470         | 0.502         | 0.184      | 0.261      | 0.536          | 0.496          | 0.083       | 0.157       | 0.263           | 0.398           | 0.533            | 0.567            | 0.826       | 0.638       |
|                       | 720    | 0.759         | 0.651         | 0.352      | 0.380      | 1.116          | 0.812          | 0.088       | 0.184       | 0.250           | 0.385           | 0.655            | 0.644            | 1.555       | 0.979       |
| Sales                 | 96     | 1.050         | 0.488         | 1.041      | 0.476      | 1.068          | 0.478          | 1.174       | 0.350       | 1.050           | 0.517           | 1.071            | 0.521            | 2.279       | 0.599       |
|                       | 192    | 1.052         | 0.489         | 1.043      | 0.477      | 1.068          | 0.481          | 1.178       | 0.356       | 1.051           | 0.516           | 1.065            | 0.514            | 2.280       | 0.560       |
|                       | 336    | 1.054         | 0.490         | 1.045      | 0.478      | 1.070          | 0.478          | 1.190       | 0.357       | 1.052           | 0.516           | 1.059            | 0.510            | 2.282       | 0.600       |
|                       | 720    | 1.063         | 0.493         | 1.053      | 0.477      | 1.080          | 0.481          | 1.205       | 0.394       | 1.058           | 0.518           | 1.066            | 0.515            | 2.230       | 0.602       |

**Legend**
- HPC - Household Power Consumption.
- QPS - Queries Per Second - different sources of system loads.

# Average Rank

To compare model performance across different datasets and horizons, we calculate the average rank for each model. For each dataset-horizon combination, models are ranked by their MSE (rank 1 = best/lowest MSE). These ranks are then averaged to provide an overall performance indicator.

## Average Rank per Horizon

The following table shows the average rank for each model across all six datasets at each forecasting horizon:

| Horizon | DLinear | CATS | PatchTST | NHITS | FEDformer | Autoformer | Naive |
|---------|---------|------|----------|-------|-----------|------------|-------|
| 96 | 3.17 | 1.17 | 3.50 | 4.33 | 3.83 | 5.83 | 6.17 |
| 192 | 2.67 | 2.17 | 3.50 | 4.17 | 3.67 | 5.50 | 6.33 |
| 336 | 2.33 | 2.67 | 4.00 | 4.00 | 3.67 | 5.00 | 6.33 |
| 720 | 2.83 | 3.00 | 4.50 | 3.67 | 3.00 | 4.67 | 6.33 |

## Overall Average Rank

The overall average rank across all datasets and horizons:

- **CATS**: 2.25
- **DLinear**: 2.75
- **FEDformer**: 3.54
- **PatchTST**: 3.88
- **NHITS**: 4.04
- **Autoformer**: 5.25
- **Naive**: 6.29

CATS and DLinear demonstrate the best overall performance, with CATS showing particularly strong consistency at shorter horizons (96 and 192 steps). Notably, PatchTST (rank 3.88), despite claiming superior performance over DLinear in its original paper, ranks lower in our evaluation across these diverse datasets. NHITS (rank 4.04) also underperforms compared to FEDformer (rank 3.54), though it shows competitive performance at longer horizons, particularly on the QPS dataset. The Naive baseline, as expected, ranks last across all horizons.

# Conclusion

Our evaluation across six diverse real-world datasets provides a nuanced answer to the question: **Are (Cross-)Attentions Necessary?** While attention mechanisms can be beneficial, they are not strictly necessary for competitive long-term forecasting.

The top two performers—CATS (rank 2.25) and DLinear (rank 2.75)—represent fundamentally different approaches. CATS demonstrates that a streamlined cross-attention-only design can be highly effective, while DLinear achieves competitive performance using no attention mechanisms whatsoever, relying instead on simple linear projections with series decomposition. This challenges the assumption that complex Transformer architectures are necessary for capturing long-range dependencies in time series.

Notably, PatchTST's underperformance relative to its original paper claims, and NHITS ranking below FEDformer, suggest that architectural complexity does not consistently translate to better performance across diverse domains. Our results indicate that simpler models like DLinear often provide comparable results with significantly lower computational overhead, while CATS offers an efficient middle ground when attention is deemed beneficial.

These findings emphasize the importance of evaluating models on diverse, under-explored datasets rather than relying solely on standard benchmarks, as no single architecture dominates universally across all forecasting scenarios.

# How to reproduce?

This section describes how to reproduce our results for each model.

## DLinear

Our DLinear experiments use the official implementation from the authors' `LTSF-Linear/` repository. Results are reproduced using the provided Jupyter notebook [`DLinear.ipynb`](DLinear.ipynb). The simplest and recommended way to run this is directly on Google Colab.

### What the notebook does:

1. Downloads LTSF-Linear repository from GitHub
2. Installs dependencies (PyTorch, pandas, scikit-learn, etc.)
3. Runs experiments on each dataset across all prediction horizons (96, 192, 336, 720 steps)
4. Saves and prints results

To run DLinear on all six datasets, execute:

```python
!bash dlinear.sh
```

To print results for all six datasets, execute:

```python
!bash show_metrics.sh ../results
```

## Naive, Autoformer, FEDformer

These models are also implemented in the LTSF-Linear repository. Reproduction follows the same process as DLinear, but with different scripts:

### Naive Baseline

To run naive baseline on all six datasets, execute :

```python
!bash naive.sh
```

### Autoformer & FEDformer

To run Autoformer and  FEDformer on all six datasets, execute :

```python
!bash transformers.sh
```

## N-HiTS

Our N-HiTS experiments use the official implementation from the authors' repository. Results are reproduced using the provided Jupyter notebook [`N-HiTS.ipynb`](N-HiTS.ipynb). The simplest and recommended way to run this is directly on Google Colab.

### What the notebook does:

1. Downloads N-HiTS repository from GitHub
2. Installs dependencies
3. Transforms datasets into N-HiTS format using the provided transform utility
4. Runs experiments on each dataset across all prediction horizons (96, 192, 336, 720 steps)
5. Saves and prints results

To run N-HiTS on all six datasets, execute in a notebook cell:

```python
!bash nhits.sh
```

To print results for all six datasets, execute:

```python
!bash evaluate.sh
```

## CATS

Our CATS experiments use the official implementation from the authors' repository. Results are reproduced using the provided Jupyter notebook [`CATS.ipynb`](CATS.ipynb). The simplest and recommended way to run this is directly on Google Colab.

### What the notebook does:

1. Downloads CATS repository from GitHub
2. Installs dependencies
3. Runs experiments on each dataset across all prediction horizons (96, 192, 336, 720 steps)
4. Saves and prints results

To run CATS on all six datasets, execute in a notebook cell:

```python
!bash cats.sh
```

## PatchTST

Our PatchTST experiments use the implementation from the TSLib repository which contains official implementation. Results are reproduced using the provided Jupyter notebook [`PatchTST.ipynb`](PatchTST.ipynb). The recommended way to run this is directly on Google Colab.

### What the notebook does:

1. Set up proper python version
2. Download TSLib repository from GitHub
3. Install dependencies
4. Run experiments on each dataset across all prediction horizons
5. Save and prints results

To run PatchTST on all six datasets, execute in a notebook cell:

```python
!bash patch_tst_runner.sh
```


# References

- Kim, D., Park, J., Lee, J., Kim, H. (2024). *Are Self-Attentions Effective for Time Series Forecasting?* NeurIPS 2024. https://arxiv.org/abs/2405.16877
- Zeng, A., Chen, M., Zhang, L., Xu, Q. (2022). *Are Transformers Effective for Time Series Forecasting?* NeurIPS 2022. https://arxiv.org/abs/2205.13504
- Nie, Y., Nguyen, N. H., Sinthong, P., Kalagnanam, J. (2023). *A Time Series is Worth 64 Words: Long-term Forecasting with Transformers.* ICLR 2023. https://arxiv.org/abs/2211.14730
- Wu, H., Xu, J., Wang, J., Long, M. (2021). *Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting.* NeurIPS 2021. https://arxiv.org/abs/2106.13008
- Zhou, T., Ma, Z., Wen, Q., Wang, X., Sun, L., Jin, R. (2022). *FEDformer: Frequency Enhanced Decomposed Transformer for Long-term Series Forecasting.* ICML 2022. https://arxiv.org/abs/2201.12740
- Challu, C., Olivares, K. G., Oreshkin, B. N., Garza, F., Mergenthaler-Canseco, M., Dubrawski, A. (2023). *N-HiTS: Neural Hierarchical Interpolation for Time Series Forecasting.* AAAI 2023. https://arxiv.org/abs/2201.12886
- Wang, Y., Wu, H., Dong, J., Liu, Y., Long, M., Wang, J. (2024). *Deep Time Series Models: A Comprehensive Survey and Benchmark.* arXiv preprint. https://arxiv.org/abs/2407.13278