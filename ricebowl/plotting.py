import seaborn as sns
import matplotlib.pyplot as plt
from ricebowl.processing import data_preproc


def pairplot(df, hue=None, cols=['ALL']):
    """
    General function to get a pair plot of the entire data. Hue can be modified.
    :param df: Input data
    :param hue: Particular column to refer
    :param cols: Columns to refer
    :return: Displays a pairplot
    """
    if cols == ['ALL']:
        sns.pairplot(data=df, hue=hue)
    else:
        sns.pairplot(data=df, hue=hue, vars=cols)
    plt.show()


def distribution(df, **col_names):
    """
    General function to get plots for all columns passed. Press 'q' for next figure.
    :param df: Input data
    :param col_names: Columns to refer and plot
    :return: Displays a distplot
    """
    for i in col_names.values():
        sns.distplot(df[i], label=i)
        plt.show()


def plot(x, y, xlabel='x', ylabel='y'):
    """
    General function to plot relationship between 2 random variables. x,y have input types as list/df series.
    :param x: First random variable (list/series)
    :param y: Second random variable (list/series)
    :param xlabel: Label for x axis
    :param ylabel: Label for y axis
    :return: Displays a chart showing relationship between 2 plots
    """
    plt.plot(x, 'g*', y, 'ro')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend([xlabel, ylabel])
    plt.show()


def scatter(data, x=None, y=None):
    """
    General function to plot a scatterplot of the data
    :param data: Input d
    :param x: x axis
    :param y: y axis
    :return: Produces a scatter plot of the given data
    """
    sns.scatterplot(x=x, y=y, data=data)
    plt.show()


def box(data):
    """
    General function to plot a boxplot of the data
    :param data: Input data
    :return: Produces a box plot
    """
    sns.boxplot(data=data)
    plt.show()


def pie_chart(data, column_name, title='Title', labels=['None'], convert=False):
    """
    General function for plotting a pie chart
    :param data: Input data
    :param column_name: Column to filter on and make a pie chart of
    :param title: Title of the chart
    :param labels: Labels to be used
    :param convert: Convert to label encoded form (True/False)
    :return: Displays a pie chart
    """
    if title == 'Title':
        title = column_name.capitalize()
    if convert == True:
        labels = list(data[column_name].unique())
        data, le = data_preproc.label_encode(data, c1=column_name)

    uniques = list(data[column_name].unique())
    if labels == ['None']:
        labels = labels * len(uniques)

    values = []
    for i in uniques:
        values = values + [data[column_name][data[column_name] == i].count()]
    fig1, ax1 = plt.subplots(figsize=(10, 8))
    ax1.pie(values, labels=labels, autopct='%1.1f%%',
            shadow=True, startangle=90)
    ax1.axis('equal')
    plt.title(title, size=20)
    plt.show()
