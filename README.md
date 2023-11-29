# spatiotemporal-lstnet-for-markets

 We aim to explore the use of attention as an automatic feature selector for multivariate time series forecasting models. We hope that attention can help use determine both spatial (feature to feature) and temporal (timestep to timestep) dependencies that can be used to weigh features based on importance.

 To test our hypothesis, we will attempt to predict soybeans futures prices based on other futures prices and other possibly related variables.

 As part of our work, we hope to attempt to replicate the architecture described in the paper [Spatiotemporal Self-Attention-Based LSTNet for Multivariate Time Series Prediction](https://www.hindawi.com/journals/ijis/2023/9523230/) by Dezheng Wang and Congyan Chen.

 Implementing [this](https://www.hindawi.com/journals/ijis/2023/9523230/) paper for applications in markets

 ## Data

 For all data, we employed a preprocessing pipeline that removed invalid values, created train and test sets, sequenced the data with a sliding window approach, designated the target timestep and variable, and performed z-score normalization.

 ### Nasdaq100
 We used the nasdaq100 dataset, found here, for initial testing, as this was the dataset used in Wang and Chen's paper.

 ### Futures
 Later, we built our own dataset of futures prices for 18 commodities by pulling data from Yahoo Finance through the Yahoo Finance API

## Setup

Once cloned, run 

`conda env create -f environment.yml`

to create the conda environment and install dependencies.

Then run `pip install -r requirements.txt` to install all pip dependencies.
