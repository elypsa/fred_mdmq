import pandas as pd
import os
import requests
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

class Fred:
    def __init__(self, path=None, url=None):
        if not path and not url:
            raise ValueError("Either 'path' or 'url' must be provided.")
        self.path = path
        self.url = url
        self.df = None
        self.transform_codes = None
        self.df_raw = None

    def __repr__(self):
        return f"Fred(path={self.path})"

    def read(self):
        if self.path is None:
            self.read_url()
        # read csv file with pandas
        self.df = pd.read_csv(self.path, skiprows=[1], parse_dates=['sasdate'])
        # order by sasdate
        self.df = self.df.sort_values(by='sasdate')
        # read the first row of the csv file to get the transform codes
        self.transform_codes = pd.read_csv(self.path, nrows=1, usecols=np.arange(1, self.df.shape[1]))
        # convert transform_codes to a dictionary
        self.transform_codes = self.transform_codes.to_dict(orient='records')[0]

    def transform(self, user_codes=None):
        if self.df is None or self.transform_codes is None:
            raise ValueError("Data has not been read. Please run the 'read' method first.")
        # copy df to a new dataframe
        if self.df_raw is None:
            self.df_raw = self.df.copy()
        # if user_codes is not None, change the transform codes
        if user_codes is not None:
            self._change_transform_code(user_codes)
        # apply the transformation codes
        for col, code in self.transform_codes.items():
            match code:
                case 1:  # no transformation
                    self.df[col] = self.df_raw[col]
                case 2:  # first difference
                    self.df[col] = self.df_raw[col].diff()
                case 3:  # second difference
                    self.df[col] = self.df_raw[col].diff().diff()
                case 4:  # log transformation
                    self.df[col] = np.log(self.df_raw[col])
                case 5:  # first difference of log
                    self.df[col] = np.log(self.df_raw[col]).diff()
                case 6:  # second difference of log
                    self.df[col] = np.log(self.df_raw[col]).diff().diff()
                case 7:  # percentage change
                    self.df[col] = self.df_raw[col].pct_change()
                case _:
                    raise ValueError(f"Unknown transformation code: {code}")

    def read_url(self):
        # download the csv file from the url
        self._download(self.url)
        # read the csv file
        local_name = self.url.split("/")[-1]
        self.path = local_name
        self.read()

    def to_pandas(self):
        # return the transformed dataframe
        if self.df is None:
            raise ValueError("No data have been read.")
        return self.df
    
    def em(self):
        pass

    def plot_missing(self, index=None, columns=None):
        # check if the dataframe is empty
        if self.df is None:
            raise ValueError("No data have been read.")
        
        if index is not None and columns is not None:
            raise ValueError("Provide either 'index' or 'columns', not both.")
        
        if index is not None:
            mask = self.df.iloc[:, index].notna().astype(int)
            column_names = self.df.columns[index]
        elif columns is not None:
            mask = self.df[columns].notna().astype(int)
            column_names = columns
        else:
            raise ValueError("Either 'index' or 'columns' must be provided.")
        
        # convert the mask to a numpy array
        mask = mask.to_numpy()
        cmap = ListedColormap(['gray', 'white'])
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(mask, aspect='auto', cmap=cmap, interpolation='nearest')
        # set the ticks and labels
        ax.set_xticks(np.arange(len(column_names)))
        ax.set_xticklabels(column_names)
        years = self.df['sasdate'].dt.year
        unique_years, year_indices = np.unique(years, return_index=True)
        unique_years = unique_years[::2]
        year_indices = year_indices[::2]
        ax.set_yticks(year_indices)
        ax.set_yticklabels(unique_years)
        plt.xlabel('Variables')
        plt.ylabel('Date Index')
        plt.title('Data Completeness Mask')
        plt.colorbar(im, ticks=[0, 1], label='0 = Missing (gray), 1 = Present (white)')
        plt.xticks(rotation=45)
        plt.show()
        

    def _change_transform_code(self, dict):
        # change the transform codes in the dictionary
        for key, value in dict.items():
            if key in self.transform_codes:
                self.transform_codes[key] = value
            else:
                raise KeyError(f"{key} not found in transform codes")
            
    def _download(self, url):
        # download the csv file from the url
        local_name = url.split("/")[-1]
        with open(local_name, "wb") as f:
            response = requests.get(url)
            if response.status_code == 200:
                f.write(response.content)
            else:
                raise ValueError(f"Failed to download file from {url}")
        
            

if __name__ == "__main__":
    fred = Fred(path="current.csv")
    fred.read()
    fred.transform()
    fred.plot_missing(index=np.arange(50,60))
    fred.plot_missing(columns=['RPI', 'USCONS'])
 
  
