import seaborn as sns
import matplotlib.pyplot as plt


# General function to get a pair plot of the entire data. Hue can be modified.
def pairplot(df, hue=None, cols=['ALL']):
    if cols == ['ALL']:
        sns.pairplot(data=df, hue=hue)
    else:
        sns.pairplot(data=df, hue=hue, vars=cols)
    plt.show()


# General function to get plots for all columns passed. Press 'q' for next figure.
def distribution(df, **col_names):
    for i in col_names.values():
        sns.distplot(df[i], label=i)
        plt.show()


# General function to plot relationship between 2 random variables. x,y have input types as list/df series.
def plot(x, y, xlabel='x', ylabel='y'):
    plt.plot(x, 'g*', y, 'ro')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend([xlabel, ylabel])
    plt.show()


# General function to plot a scatterplot of the data
def scatter(data, x=None, y=None):
    sns.scatterplot(x=x, y=y, data=data)
    plt.show()


# General function to plot a boxplot of the data
def box(data):
    sns.boxplot(data=data)
    plt.show()
