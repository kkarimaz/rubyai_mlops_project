from sklearn.datasets import load_wine

data = load_wine(as_frame=True)
data.frame.to_csv("data/wine.csv", index=False)