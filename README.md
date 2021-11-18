# PoissonRandomForest
In this code, the Random Forest function was implemented on a Poisson bootstrap samples.

This class implements Random Forest Regression and Random Forest Classifire on Poisson bootstrap samples.

The advantage of this method, relative to the standard bootstrap sampling, is to adapt the Random Forest methodology to handle the case when data is received
sequentially. More precisely, when new data arrives, the online samples are updated k times, where k is selected from the Poisson distribution to simulate sample loading of the sample. This means that this new data will be displayed k times in the tree, which simulates the fact that one data can be output k times in the sample.

At the same time, the speed of learning and prediction increases and, accordingly, the execution time of the function decreases, relative to the classic Random Forest, which can be crawling with large amounts of data.
