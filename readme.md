# Time-series Forecasting: Are (Cross-)Attentions Necessary?

# Authors

* Paweł Gacek
* Dawid Wołek

# Datasets

To evaluate the long-term forecasting capabilities of state-of-the-art models such as DLinear, PatchTST, CATS, and others, we conduct experiments on six diverse and under-explored real-world time series datasets. These datasets cover a range of domains, including renewable energy, financial markets, industrial sales, and household utility usage, presenting varied temporal dynamics, sampling resolutions, noise characteristics, and forecasting challenges. Using less commonly benchmarked datasets helps assess how well modern forecasting models generalize beyond standard public benchmarks.

Below, we describe each dataset individually, outlining its source, key attributes, and the forecasting setup used in our experiments.

## Air Pollution

The air pollution dataset we use contains hourly measurements of air quality indicators, most notably PM2.5 concentration, which is a key metric for particulate pollution. This dataset comes from a Kaggle collection designed for forecasting tasks and captures real-world temporal dynamics in atmospheric pollution levels with 20,976 samples. It exhibits strong seasonal and diurnal patterns, and challenges models with noise, missing values, and varied autocorrelation structures common to environmental data. Because pollution levels respond to both local emissions and weather conditions, this dataset tests each model's ability to capture both short-term variability and long-term trends.

**Source:** [Kaggle - LSTM Datasets Multivariate Univariate](https://www.kaggle.com/datasets/rupakroy/lstm-datasets-multivariate-univariate/data)

## Wind Power Generation (Wind-Power-Consumption)

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

# Results Summary

The following table presents a comprehensive comparative analysis of various long-term time series forecasting models across diverse real-world datasets. We evaluate performance using Mean Squared Error (MSE) and Mean Absolute Error (MAE) across four standard forecasting horizons: 96, 192, 336, and 720 time steps.

| Dataset               | Metric | MSE (DLinear) | MAE (DLinear) | MSE (CATS) | MAE (CATS) | MSE (PatchTST) | MAE (PatchTST) | MSE (NHITS) | MAE (NHITS) | MSE (TFT) | MAE (TFT) | MSE (FEDformer) | MAE (FEDformer) | MSE (Autoformer) | MAE (Autoformer) | MSE (Naive  Last) | MAE (NaiveLast Value) |
|-----------------------|--------|---------------|---------------|------------|------------|----------------|----------------|-------------|-------------|-----------|-----------|-----------------|-----------------|------------------|------------------|------------:|------------:|
| Air Pollution         | 96     | 0.122         | 0.250         | 0.122      | 0.243      | 0.131          | 0.255          | 0.158       | 0.284       |           |           | 0.151           | 0.287           | 0.188            | 0.319            | 0.448       | 0.461       |
|                       | 192    | 0.149         | 0.277         | 0.152      | 0.273      | 0.163          | 0.283          | 0.214       | 0.325       |           |           | 0.182           | 0.313           | 0.194            | 0.319            | 0.482       | 0.484       |
|                       | 336    | 0.175         | 0.306         | 0.177      | 0.293      | 0.194          | 0.307          | 0.232       | 0.337       |           |           | 0.211           | 0.341           | 0.268            | 0.386            | 0.516       | 0.505       |
|                       | 720    | 0.229         | 0.356         | 0.247      | 0.346      | 0.267          | 0.356          | 0.281       | 0.392       |           |           | 0.236           | 0.361           | 0.281            | 0.396            | 0.579       | 0.538       |
| Microsoft Stock       | 96     | 2.468         | 1.013         | 0.797      | 0.572      | 1.248          | 0.692          | 0.926       | 0.703       |           |           | 2.361           | 1.009           | 2.440            | 1.016            | 1.218       | 0.688       |
|                       | 192    | 3.470         | 1.215         | 3.270      | 1.168      | 3.088          | 1.082          | 1.246       | 0.819       |           |           | 4.311           | 1.372           | 4.677            | 1.415            | 3.109       | 1.109       |
|                       | 336    | 7.631         | 1.693         | 11.031     | 2.256      | 5.894          | 1.531          | 1.343       | 0.835       |           |           | 8.188           | 1.887           | 8.249            | 1.844            | 6.606       | 1.659       |
|                       | 720    | 21.068        | 2.944         | 22.730     | 3.265      | 18.673         | 2.834          | 1.410       | 0.884       |           |           | 19.593          | 2.955           | 20.534           | 2.996            | 19.435      | 2.926       |
| Wind Power Generation | 96     | 0.702         | 0.651         | 0.696      | 0.643      | 0.760          | 0.674          | 0.831       | 0.691       |           |           | 0.762           | 0.677           | 0.783            | 0.691            | 1.325       | 0.852       |
|                       | 192    | 0.736         | 0.675         | 0.736      | 0.672      | 0.810          | 0.703          | 0.898       | 0.734       |           |           | 0.819           | 0.707           | 0.848            | 0.720            | 1.425       | 0.904       |
|                       | 336    | 0.754         | 0.688         | 0.754      | 0.684      | 0.851          | 0.725          | 0.917       | 0.745       |           |           | 0.865           | 0.729           | 0.850            | 0.726            | 1.496       | 0.937       |
|                       | 720    | 0.775         | 0.705         | 0.777      | 0.704      | 0.878          | 0.743          | 0.867       | 0.730       |           |           | 0.875           | 0.737           | 0.867            | 0.736            | 1.528       | 0.957       |
| HPC                   | 96     | 0.608         | 0.513         | 0.606      | 0.508      | 0.638          | 0.524          | 0.658       | 0.482       | 16.172    | 1.868     | 0.633           | 0.535           | 0.672            | 0.567            | 1.340       | 0.715       |
|                       | 192    | 0.622         | 0.523         | 0.631      | 0.524      | 0.665          | 0.538          | 0.689       | 0.494       | 14.759    | 1.819     | 0.656           | 0.556           | 0.702            | 0.580            | 1.363       | 0.727       |
|                       | 336    | 0.627         | 0.527         | 0.642      | 0.532      | 0.681          | 0.544          | 0.695       | 0.499       | 17.440    | 1.925     | 0.664           | 0.556           | 0.699            | 0.583            | 1.376       | 0.734       |
|                       | 720    | 0.626         | 0.530         | 0.643      | 0.536      | 0.679          | 0.551          | 0.710       | 0.508       | 14.931    | 1.805     | 0.667           | 0.562           | 0.687            | 0.569            | 1.359       | 0.734       |
| QPS                   | 96     | 0.090         | 0.198         | 0.041      | 0.116      | 0.063          | 0.158          | 0.043       | 0.119       |           |           | 0.086           | 0.218           | 0.363            | 0.432            | 0.153       | 0.252       |
|                       | 192    | 0.230         | 0.340         | 0.085      | 0.172      | 0.207          | 0.296          | 0.084       | 0.175       |           |           | 0.185           | 0.319           | 0.308            | 0.419            | 0.426       | 0.661       |
|                       | 336    | 0.470         | 0.502         | 0.184      | 0.261      | 0.536          | 0.496          | 0.083       | 0.157       |           |           | 0.263           | 0.398           | 0.533            | 0.567            | 0.826       | 0.638       |
|                       | 720    | 0.759         | 0.651         | 0.352      | 0.380      | 1.116          | 0.812          | 0.088       | 0.184       |           |           | 0.250           | 0.385           | 0.655            | 0.644            | 1.555       | 0.979       |
| Sales                 | 96     | 1.050         | 0.488         | 1.041      | 0.476      | 1.068          | 0.478          | 1.174       | 0.350       |           |           | 1.050           | 0.517           | 1.071            | 0.521            | 2.279       | 0.599       |
|                       | 192    | 1.052         | 0.489         | 1.043      | 0.477      | 1.068          | 0.481          | 1.178       | 0.356       |           |           | 1.051           | 0.516           | 1.065            | 0.514            | 2.280       | 0.560       |
|                       | 336    | 1.054         | 0.490         | 1.045      | 0.478      | 1.070          | 0.478          | 1.190       | 0.357       |           |           | 1.052           | 0.516           | 1.059            | 0.510            | 2.282       | 0.600       |
|                       | 720    | 1.063         | 0.493         | 1.053      | 0.477      | 1.080          | 0.481          | 1.205       | 0.394       |           |           | 1.058           | 0.518           | 1.066            | 0.515            | 2.230       | 0.602       |

**Legend**
- HPC - Household Power Consumption.
- QPS - Queries Per Second - different sources of system loads.