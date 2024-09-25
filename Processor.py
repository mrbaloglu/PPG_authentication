import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MultiLabelBinarizer
from scipy import signal
import pywt
from torch.utils.data import Dataset, DataLoader

import torch
from typing import Literal, List, Optional, Tuple, Union


class Processor:
    def __init__(self, 
                 train_data: pd.DataFrame, 
                 test_data: Optional[pd.DataFrame], 
                 target_col: str,
                 non_ppg_cols: List[str], 
                 mother_wavelets: List[str], 
                 scaler: Literal["standard", "minmax"] = "standard"
        ):

        self.train_data = self.arrange_data(train_data)

        if test_data is not None:
            self.test_data = self.arrange_data(test_data)

        self.target_col = target_col
        self.non_ppg_cols = non_ppg_cols
        self.ppg_cols = [c for c in self.train_data.columns if c not in non_ppg_cols]
        self.mother_wavelets = mother_wavelets

        assert len(self.mother_wavelets) > 1

        if scaler == "standard":
            self.scaler = StandardScaler()
        elif scaler == "minmax":
            self.scaler = MinMaxScaler()
        else:
            raise ValueError("")
        
        self.binarizer = MultiLabelBinarizer()

    def arrange_data(self, df):
        df = df.T
        df = df.rename(columns={301: "Serial", 303: "ID", 305: "Total"})
        df = df.reset_index()
        df = df.rename(columns={"index": -1})
        df = df.drop([299, 300, 302, 304], axis=1)
        return df
    

    def butter_bandpass(self, lowcut, highcut, fs, order=4):
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = signal.butter(order, [low, high], btype='band')
        return b, a

    def apply_bandpass_filter(self, signals, lowcut, highcut, fs):
        b, a = self.butter_bandpass(lowcut, highcut, fs)
        filtered_signals = signal.filtfilt(b, a, signals, axis=1)
        return filtered_signals 
    
    def apply_cwt(self, ppg_data, wavelet='mexh', num_scales=64):
        """
        Apply Continuous Wavelet Transform (CWT) to the preprocessed PPG data.

        Parameters:
        - ppg_data: numpy array of shape (num_samples, num_points), the preprocessed PPG signal
        - wavelet: string, type of wavelet to use (default: 'morl' for Morlet wavelet)
        - num_scales: integer, number of scales for the CWT (default: 32)

        Returns:
        - cwt_coeffs: numpy array of CWT coefficients of shape (num_samples, num_scales, num_points)
        """
        
        num_samples, num_points = ppg_data.shape
        cwt_coeffs = []

        # Define scales
        scales = np.arange(1, num_scales + 1)

        # Apply CWT to each sample
        for sample in range(num_samples):
            # Perform CWT for the current sample
            coeffs, _ = pywt.cwt(ppg_data[sample], scales, wavelet)
            cwt_coeffs.append(coeffs)
        
        # Convert to numpy array: shape will be (num_samples, num_scales, num_points)
        cwt_coeffs = np.array(cwt_coeffs)
        
        return cwt_coeffs
    
    def prepare_training_data(self, 
                              return_tensors: Literal["np", "pt"] = "np"
        ) -> Tuple[Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor]]:

        df = self.train_data.copy()

        df_y = df[self.target_col].astype(int).values
        df_y = self.binarizer.fit_transform(df_y.reshape(-1, 1))

        df_x = df[self.ppg_cols]
        df_x = df_x.astype(float)
        df_x = signal.detrend(df_x.values, axis=1)
        df_x = self.apply_bandpass_filter(df_x, lowcut=0.5, highcut=5, fs=50)
        df_x = self.scaler.fit_transform(df_x.T).T  # Transpose to standardize each signal independently

        df_x_wavelets = []
        for wavelet in self.mother_wavelets:
            df_x_wv = self.apply_cwt(df_x, wavelet=wavelet)
            df_x_wavelets.append(df_x_wv)
        
        df_x_wavelets = np.stack(df_x_wavelets, axis=1)

        if return_tensors == "pt":
            df_x_wavelets = torch.FloatTensor(df_x_wavelets)
            df_y = torch.FloatTensor(df_y)
        
        return df_x_wavelets, df_y
    
    
    def prepare_test_data(self, 
                              return_tensors: Literal["np", "pt"] = "np"
        ) -> Tuple[Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor]]:

        df = self.test_data.copy()
        
        df_y = df[self.target_col].values
        df_y = self.binarizer.transform(df_y.reshape(-1, 1))

        df_x = df[self.ppg_cols]
        df_x = df_x.astype(float)
        df_x = signal.detrend(df_x.values, axis=1)
        df_x = self.apply_bandpass_filter(df_x, lowcut=0.5, highcut=5, fs=50)
        df_x = self.scaler.fit_transform(df_x.T).T  # Transpose to standardize each signal independently

        df_x_wavelets = []
        for wavelet in self.mother_wavelets:
            df_x_wv = self.apply_cwt(df_x, wavelet=wavelet)
            df_x_wavelets.append(df_x_wv)
        
        df_x_wavelets = np.stack(df_x_wavelets, axis=1)

        if return_tensors == "pt":
            df_x_wavelets = torch.FloatTensor(df_x_wavelets)
            df_y = torch.FloatTensor(df_y)
        
        return df_x_wavelets, df_y
    






    