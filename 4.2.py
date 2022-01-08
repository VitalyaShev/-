import numpy as np
import pandas as pd
import scipy.stats as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns


def main():
    iris = sns.load_dataset('iris')
    setosa = iris.loc[iris.species == 'setosa']
    plt.figure(figsize=(12, 8))
with sns.axes_style("darkgrid"):
    ax = sns.kdeplot(setosa.sepal_length, setosa.sepal_width, label="setosa", cmap='Blues')


if __name__ == '__main__':
    main()