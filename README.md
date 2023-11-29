# spatiotemporal-lstnet-for-markets

 We aim to explore the use of attention as an automatic feature selector for multivariate time series forecasting models. We hope that attention can help use determine both spatial (feature to feature) and temporal (timestep to timestep) dependencies that can be used to weigh features based on importance.

 To test our hypothesis, we will attempt to predict soybeans futures prices based on other futures prices and other possibly related variables.

 As part of our work, we hope to attempt to replicate the architecture described in the paper [Spatiotemporal Self-Attention-Based LSTNet for Multivariate Time Series Prediction](https://www.hindawi.com/journals/ijis/2023/9523230/) by Dezheng Wang and Congyan Chen.

 ## Data

 For all data, we employed a preprocessing pipeline that removed invalid values, created train and test sets, sequenced the data with a sliding window approach, designated the target timestep and variable, and performed z-score normalization.

 ### Nasdaq100
 We used the nasdaq100 dataset, found [here](https://cseweb.ucsd.edu/~yaq007/NASDAQ100_stock_data.html), for initial testing, as this was the dataset used in Wang and Chen's paper.

 ### Futures
 Later, we built our own dataset of futures prices for 18 commodities by pulling data from Yahoo Finance through the Yahoo Finance API.

 ![alt text](https://github.com/sebiancoder/spatiotemporal-lstnet-for-markets/blob/seb_branch/assorted_images/yf_futures_performance.png?raw=true)

 ## Models

 We have so far implemented the following models for our project. Performance is detailed below.

 ### Baseline Models

 These models are traditional models that have been previously used for Time Series Forecasting, and we included them here as benchmarks to which we will compare our Attention based models.

| Model   | Nasdaq100 Performance (MAE)  | Futures Dataset Performance (MAE) |
| :------------ |:---------------:| :-----:|
| LSTM   | 1.836 | 823.53 |
| TCN  | 64.47 | Pending |
| MLP | 66.79 | Pending |
| ARIMA | Pending | 117.43 |

### Attention Based Models

Here we have our two models that have attention mechanisms: the AttentionLSTM and Wang and Chen's LSTNet. The AttentionLSTM simply involves spatiotemporal self-attention as detailed in Wang and Chen's paper, followed by a LSTM layer.

| Model   | Nasdaq100 Performance (MAE)  | Futures Dataset Performance (MAE) |
| :------------ |:---------------:| :-----:|
| AttentionLSTM   | 6.82 | Pending |
| LSTNet  | 8.57 | Pending |

## Further Explanation of our Work

For a better look at how we built and trained our models, please take a look at the notebooks entitled "main.ipynb" for the main training loop code for most models. In the "Models" folder, you can also find a training log of all the models we trained.

