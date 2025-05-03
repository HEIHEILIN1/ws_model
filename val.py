import pandas as pd

if __name__=='__main__':
    df=pd.read_csv('data/datasets/cowboy_outfits/resample_train.csv')
    print(df.head())