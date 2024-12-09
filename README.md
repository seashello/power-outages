# Exploring Power Outages

This is a project from DSC 80: Practice and Application of Data Science, exploring data regarding power outages.


## Introduction
This dataset contains information regarding 1500+ power outages across the US, from 2000-2016. It contains other information, such as the state it occurred in, the cause of the power outage, the state's population from that year, etc.. This project focuses on predicting the cause category of an outage. We also explore other aspects about power outages, such as looking at missingness in the dataset. This dataset and question are important to note because power outages affect entire communities, and especially as a Californian, we have experienced many. It is good to know what caused them. There are 1534 rows of data, and 56 columns (but we will be keeping 8).

- `US State`: The state that the power outage occurred in
- `Climate Category`: The climate category during the outage (normal, cold, or warm)
- `Cause Category`: The reason for the power outage
- `Outage Duration`: Duration of the power outage
- `Demand Loss Mw`: The amount of peak demand loss in Megawatt (or the total amount of loss)
- `Customers Affected`: The number of customers affected by power outage
- `Population`: Population at the given US state in a year
- `Popden Rural`: Population density of the urban areas (persons per square mile)

## Data Cleaning and Exploratory Data Analysis
### Data Cleaning
The original dataset file was formatted oddly, with rows and columns in the Google Sheet that weren't actual data (and were just empty cells outside the actual table). I fixed it to include just the pure data, and also changed the column names for better readability, such as turning "CAUSE.CATEGORY" into "Cause Category". Then, I kept only the columns that we care about.

`outages.head()`

| US State   | Climate Category   | Cause Category     |   Outage Duration |   Demand Loss Mw |   Customers Affected |   Population |   Popden Rural |
|:-----------|:-------------------|:-------------------|------------------:|-----------------:|---------------------:|-------------:|---------------:|
| Minnesota  | normal             | severe weather     |              3060 |              nan |                70000 |      5348119 |           18.2 |
| Minnesota  | normal             | intentional attack |                 1 |              nan |                  nan |      5457125 |           18.2 |
| Minnesota  | cold               | severe weather     |              3000 |              nan |                70000 |      5310903 |           18.2 |
| Minnesota  | normal             | severe weather     |              2550 |              nan |                68200 |      5380443 |           18.2 |
| Minnesota  | warm               | severe weather     |              1740 |              250 |               250000 |      5489594 |           18.2 |

### Univariate Analysis
Let's look at the number of power outages per state. It appears that California by far has the most!

### Bivariate Analysis
California has a lot of power outages. But, is this just because it has the biggest population? Let's put it to scale. Let's find the number of outages per person. This number by itself is meaningless, but it allows us to compare states to each other at the same scale. We find that California doesn't have the biggest number of outages, per person (over the years). It seems like Delaware does!

### Interesting Aggregates
In this plot, we compare the outage duration between different climate categories, and different cause categories. It seems that generally, values vary a lot and on average, outages caused by fuel supply emergency in a warm climate last the longest, while outages caused by islanding with normal climates last the least amount of time.

| Cause Category                |      cold |   normal |      warm |
|:------------------------------|----------:|---------:|----------:|
| equipment failure             |   308.235 | 3201.43  |   505     |
| fuel supply emergency         | 17433     | 7658.82  | 22799.7   |
| intentional attack            |   497.282 |  426.818 |   312.557 |
| islanding                     |   259.267 |  142.176 |   209.833 |
| public appeal                 |  2125.91  | 1376.53  |   596.231 |
| severe weather                |  3279.95  | 4059.33  |  4416.69  |
| system operability disruption |   601.861 |  941.018 |   478.2   |

## Assessment of Missingness
My data has a lot of missing values. However, I don't think any of the columns from my DataFrame are NMAR. NMAR means that there is missingness based on the values itself. Columns such as "Demand Loss Mw" and "Customers Affected" are likely missing due to other factors, such as information is less available -- for example, different states, or rural populations might not be able to measure how many customers were affected or demand loss from a power outage, due to differences in funding or from being at a rural location with less monitoring. We can test this by running permutation tests.

Does the missingness of "Customers Affected" depend on "Popden Rural"? We conclude yes, with a p-value of 0.0.

## Hypothesis Testing
Null: "Demand Loss Mw" of California or Washington come from the same distribution 
Alternate: Whether a state is California or Washington doesn't have a relationship to "Outage Duration"
Test statistic: absolute difference in group means
P-value threshold: 0.05
We fail to reject that they come from the same population, with a p-value of 0.146

## Framing a Prediction Problem
I will predict the 'Cause Category' of a power outage. This is a classification problem, and we will be performing multiclass classification. I chose to predict the cause category of an outage because it would make sense that different causes to outages have distinct properties, such as how many people it affected, what states they tend to occur in, or climate category. I am using accuracy because it 

## Baseline Model
## Final Model
## Fairness Analysis