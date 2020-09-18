import seaborn as sns
import matplotlib.pyplot as plt
from ricebowl.ricebowl.processing import data_preproc


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


# General function for plotting a pie chart
def pie_chart(data, column_name, title='Title', labels=['None'], convert=False):
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
