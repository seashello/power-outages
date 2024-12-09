# Exploring Power Outages

This is a project from DSC 80: Practice and Application of Data Science, exploring data regarding power outages.


### Introduction
This dataset contains information regarding 1500+ power outages across the US, from 2000-2016. It contains other information, such as the state it occurred in, the cause of the power outage, the state's population from that year, etc.. This project focuses on predicting the cause category of an outage. We also explore other aspects about power outages, such as looking at missingness in the dataset. This dataset and question are important to note because power outages affect entire communities, and especially as a Californian, we have experienced many. It is good to know what caused them. There are 1534 rows of data, and 56 columns (but we will be keeping 8).

- `US State`: The state that the power outage occurred in
- `Climate Category`: The climate category during the outage (normal, cold, or warm)
- `Cause Category`: The reason for the power outage
- `Outage Duration`: Duration of the power outage
- `Demand Loss Mw`: The amount of peak demand loss in Megawatt (or the total amount of loss)
- `Customers Affected`: The number of customers affected by power outage
- `Population`: Population at the given US state in a year
- `Popden Rural`: Population density of the urban areas (persons per square mile)

### Data Cleaning and Exploratory Data Analysis
###### Data Cleaning
The original dataset file was formatted oddly, with rows and columns in the Google Sheet that weren't actual data (and were just empty cells outside the actual table). I fixed it to include just the pure data, and also changed the column names for better readability, such as turning "CAUSE.CATEGORY" into "Cause Category". Then, I kept only the columns that we care about.

###### Univariate Analysis
Let's look at the number of power outages per state. It appears that California by far has the most!

###### Bivariate Analysis
California has a lot of power outages. But, is this just because it has the biggest population? Let's put it to scale. Let's find the number of outages per person. This number by itself is meaningless, but it allows us to compare states to each other at the same scale. We find that California doesn't have the biggest number of outages, per person (over the years). It seems like Delaware does!

###### Interesting Aggregates
In this plot, we compare the outage duration between different climate categories, and different cause categories. It seems that generally, values vary a lot and on average, outages caused by fuel supply emergency in a warm climate last the longest.

| US State   | Climate Category   | Cause Category     |   Outage Duration |   Demand Loss Mw |   Customers Affected |   Population |   Popden Rural |
|:-----------|:-------------------|:-------------------|------------------:|-----------------:|---------------------:|-------------:|---------------:|
| Minnesota  | normal             | severe weather     |              3060 |              nan |                70000 |      5348119 |           18.2 |
| Minnesota  | normal             | intentional attack |                 1 |              nan |                  nan |      5457125 |           18.2 |
| Minnesota  | cold               | severe weather     |              3000 |              nan |                70000 |      5310903 |           18.2 |
| Minnesota  | normal             | severe weather     |              2550 |              nan |                68200 |      5380443 |           18.2 |
| Minnesota  | warm               | severe weather     |              1740 |              250 |               250000 |      5489594 |           18.2 |

### Assessment of Missingness
### Hypothesis Testing
### Framing a Prediction Problem
### Baseline Model
### Final Model
### Fairness Analysis