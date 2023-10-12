import pandas as pd
import numpy as np

class DataProcessor:

    def sliding_window_sequence(data, seq_len):
        '''
        Given a time series, use a sliding window approach to generate sequences
        '''
    
        sequences = [data.iloc[i:i + seq_len] for i in range(0,len(data) - seq_len + 1, 1)]

        for sequence in sequences:

            sequence.reset_index(drop=True, inplace=True)

        seqs = pd.concat(sequences, axis=0, keys=(list(range(len(sequences)))))
        
        return seqs

    def sequence_target_split(data : pd.DataFrame, target_size : int = 1):
        '''
        Split a sequence into sequences and the target (value attempting to be predicted). Useful for time series prediction

        :param data: the data to be split
        :param target_size: number of datapoints to be the target
        :return: multiindex pandas array containing sequence and target
        :NOTE at this point, setting a target_size != 1 will cause problems with the lstm, as it is not setup to handle a 2D output shape
        '''
        
        seq_len = len(data.xs(0, level=0))
        seq_mask = data.index.get_level_values(1) < (seq_len - 1)
        
        sequences = data[seq_mask]
        targets = data.xs((seq_len - 1), level=1, drop_level=False)

        return sequences, targets

    def get_zscore_params(data):
        '''
        Get z-score parameters (mean and standard deviation) for every column
        Throws error if non-numeric data present

        :param data: pandas dataFrame containing data to be standardized
        :return:
        '''
        
        if not data.applymap(np.isreal).values.all():

            raise Exception("Cannot calculate zscore parameters for non-numeric data")

        
        means = data.mean()
        std_devs = data.std()

        return means, std_devs

    def fold_sequences(data):
        '''
        Given a dataframe containing multilevel indices, fold the dataframe into a 3d numpy array where
        level 0 multi-index -> dim 0
        level 1 multi-index -> dim 1
        level 2 multi-index -> dim 2

        :return: a numpy array as described above
        '''
        
        sequences = [data.xs(first_level_index, level=0).values for first_level_index in \
            data.index.get_level_values(0).unique()]

        return np.squeeze(np.stack(sequences))
    
    def zscore_standardization(data):
        '''
        standardize data into a normal distribution

        :param data: pandas dataFrame containing data to be standardized
        :return: pandas dataFrame with standardized data
        '''

        if type(data) == pd.Series:

            data = data.to_frame()
        
        numeric_data = data.select_dtypes([np.number])
        
        mu, sigma = DataProcessor.get_zscore_params(numeric_data)
        
        return (numeric_data - mu) / sigma

    def calc_log_returns(data):
        '''
        Calculate log returns for a multivate time series
        R = ln(CLOSE_t / CLOSE_(t-1))
        '''

        return np.log(data / data.shift(periods=1))