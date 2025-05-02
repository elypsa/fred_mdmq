import pandas as pd
from datetime import datetime
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
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
        self.df_appendix = None
        self.outliers = None
        self.df_imp = None
        print("Fred object created.")

    def __repr__(self):
        return f"Fred(path={self.path})"

    def read(self, start_date=None, end_date=None):
        if self.path is None:
            self.read_url()
        # read csv file with pandas
        self.df = pd.read_csv(self.path, skiprows=[1], parse_dates=['sasdate'])
        self.df = self.df.sort_values(by='sasdate')
        self.df.set_index('sasdate', inplace=True)
        self.df.index = self.df.index.to_period('M')
        if start_date is not None:
            self.df = self.df[self.df.index >= start_date]
        if end_date is not None:
            self.df = self.df[self.df.index <= end_date]
        # read the first row of the csv file to get the transform codes
        self.transform_codes = pd.read_csv(self.path, nrows=1, usecols=np.arange(1, self.df.shape[1]))
        # convert transform_codes to a dictionary
        self.transform_codes = self.transform_codes.to_dict(orient='records')[0]
        # read appendix csv
        self.df_appendix = pd.read_csv("appendix/FRED-MD_updated_appendix.csv", usecols=["fred", "description", "group"])
        self.df_appendix.set_index("fred", inplace=True)
        # hierarchical indexing
        self.df.columns = pd.MultiIndex.from_arrays(
            [fred.df.columns.tolist(), fred.df_appendix.group.tolist()],
            names=["column", "group"]
        )
        print("Data read successfully.")
        print(f"\tData shape: {self.df.shape}")


    def transform(self, user_codes=None, remove_outliers=True):
        if self.df is None or self.transform_codes is None:
            raise ValueError("Data has not been read. Please run the 'read' method first.")
        # copy df to a new dataframe
        if self.df_raw is None:
            self.df_raw = self.df.copy()
        # if user_codes is not None, change the transform codes
        if user_codes is not None:
            self._change_transform_code(user_codes)
        # apply the transformation codes
        init_na_mask = self.df.isna()
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
                case 7:  # first difference of percentage change
                    self.df[col] = self.df_raw[col].pct_change().diff()
                case _:
                    raise ValueError(f"Unknown transformation code: {code}")
            # remove outliers
            if remove_outliers:
                # remove outliers that deviate from median by more than 10 interquartile ranges
                q1 = self.df[col].quantile(0.25)
                q3 = self.df[col].quantile(0.75)
                iqr = q3 - q1
                self.df[col] = np.where(
                    (self.df[col]-self.df[col].median()).abs() > 10*iqr,
                    np.nan,
                    self.df[col]
                )
        # remove first two rows
        self.df = self.df.iloc[2:]
        print("Transformation completed.")
        if remove_outliers:
            self.outliers = pd.concat([init_na_mask.sum(), self.df.isna().sum()], axis=1, keys=["Initial NA", "Current NA"])
            self.outliers['Difference'] = self.outliers['Current NA'] - self.outliers['Initial NA']
            self.outliers.sort_values(by='Difference', ascending=False, inplace=True)
            # print("Outliers removed (top10):")
            # print(self.outliers.head(10))
            n_outliers = self.outliers['Difference'].sum()
            print(f"\tRemoved {n_outliers} outliers from the data.")

    def read_url(self):
        # download the csv file from the url
        self._download(self.url)
        # read the csv file
        local_name = self.url.split("/")[-1]
        self.path = local_name
        self.read()

    # def to_pandas(self):
    #     # return the transformed dataframe
    #     if self.df is None:
    #         raise ValueError("No data have been read.")
    #     return self.df
    
    def ask_appendix(self, index=None):
        # check if the dataframe is empty
        if self.df_appendix is None:
            raise ValueError("No appendix data have been read.")
        
        if index is not None:
            print(self.df_appendix.loc[index])
        else:
            print(self.df_appendix)

    def em(self, max_iter=5000, tol=1e-6):
        # not ready yet
        # todo:
        # implement search for the best number of components (needs to be done in the pipeline!)
        # find out which distance metric to use

        if self.df_raw is None:
            raise ValueError("Transformation has not been applied. Please run the 'transform' method first.")
        
        df = self.df.copy()
        na_mask = df.isna()
        df.fillna(df.mean(), inplace=True) # initialize missing values with mean of the column
        df = df.to_numpy(copy=True)
        
        # create pipeline
        pipeline = Pipeline([
            ('scale', StandardScaler()),
            ('svd', TruncatedSVD(n_components=8, algorithm='arpack'))
        ])
        iter = 0
        while iter < max_iter:
            # fit the pipeline to the data
            F = pipeline.fit_transform(df)
            # transform the data back to the original space
            df_ = pipeline.inverse_transform(F)
            # Check if the factor estimates have converged
            if np.allclose(df[na_mask], df_[na_mask], atol=tol):
                break
            df[na_mask] = df_[na_mask]
            iter += 1

        # Print results
        if iter == max_iter:
            print(
                f"EM alogrithm failed to converge after Maximum iterations of {max_iter}.")
        else:
            print(f"EM algorithm converged after {iter} iterations")

        # df[na_mask] = initial_scale.inverse_transform(F)[na_mask]
        df = pd.DataFrame(df, columns=self.df.columns, index=self.df.index)
        self.df_imp = df
        print("EM algorithm completed.")
    
    def plot_imputed(self, column):
        # check that EM algorithm has been run
        if self.df_imp is None:
            raise ValueError("EM algorithm has not been run. Please run the 'em' method first.")
        # check that the column exists in the dataframe
        if column not in self.df_imp.columns:
            raise ValueError(f"Column {column} not found in the dataframe.")
        # plot the original and imputed data
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(self.df.index.to_timestamp(), self.df[column], label='Original', color='blue', alpha=0.5)
        ax.plot(self.df_imp.index.to_timestamp(), self.df_imp[column], label='Imputed', color='red', alpha=0.1)
        ax.set_xlabel('Date')
        ax.set_ylabel('Value')
        ax.set_title(f'Original vs Imputed Data for {column}')
        ax.legend()
        plt.show()

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
        years = self.df.index.year
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
    
    @staticmethod
    def _download(url='https://www.stlouisfed.org/-/media/project/frbstl/stlouisfed/research/fred-md/monthly/current.csv'):
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
    fred.transform(remove_outliers=True)
    # print(fred.ask_appendix(index=fred.outliers.index.get_level_values('column')[:5]))



    # fred.plot_missing(columns=fred.outliers.index.get_level_values('column')[:5])
    # print(fred.df.loc["2020", fred.df.columns[1]])
    # print(fred.df_appendix.head())
    # print(fred.ask_appendix(index=['CUSR0000SAD', 'USCONS', 'VIXCLSx']))
    # print(fred.df_appendix.loc[:,"group"])
    # fred.plot_missing(index=np.arange(0, 4))
    # print(len(fred.df.columns))
    # print(len(fred.df_appendix.index))
    # print(fred.df.head())
    fred.em()
    print(fred.df_imp.isna().sum().sum())
    print(fred.df.isna().sum().sum())
    # print(fred.outliers)
    fred.plot_imputed("UMCSENTx")