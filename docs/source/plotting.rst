Plotting
========
Documentation of ricebowl plotting.
To use this simply do from ricebowl import plotting and then use each function with plotting.<function>

Please note all these are basic plotting graphs and will be plotted. You will have to press 'q' for another graph.


pairplot
^^^^^^^^
General function to get a pair plot of the entire data. Hue can be modified to get the data along a single categorical column. You can see all types of information using this graph.

Parameters- Dataframe, hue(optional ; default=None), columns to plot for(optional ; default=['ALL'])


Output- Graph is plotted

Usage::
    
    pairplot(df, hue='species', cols=['a',b'])



distribution
^^^^^^^^^^^^
General function to get plots for all columns passed. Press 'q' for next figure. You can check the distribution of the data using these graphs

Parameters- Dataframe, kwargs[column names]


Output- Graph is plotted

Usage::
    
    distribution(df, c1='xyz', c2='abc')



plot
^^^^
General function to plot relationship between 2 random variables. x,y have input types as list/df series.

Parameters- x, y, xlabel(optional ; default='x'), ylabel(optional ; default='y')
Please note: x and y can be either list or df series.

Output- Graph is plotted

Usage::
    
    plot(x, y, xlabel='fruits', ylabel='prices')



scatter
^^^^^^^
General function to plot a scatterplot of the data

Parameters- data, x(optional ; default=None), y(optional ; default=None)

Output- Graph is plotted

Usage::
    
    scatter(data, 'length', 'width')


box
^^^
General function to plot a boxplot and check the outliers.

Parameters- data

Output- Graph is plotted

Usage::
    
    box(data)


