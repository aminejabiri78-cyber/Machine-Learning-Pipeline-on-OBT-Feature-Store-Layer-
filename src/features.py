import numpy as np

def add_features(df):

    df["prix_m2"] = df["prix"] / df["surface"]
    df["surface_chambres"] = df["surface"] * df["chambres"]
    df["log_prix"] = np.log1p(df["prix"])

    return df