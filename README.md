# Exploring Power Outages + Intentional Attacks

Michelle Hong

This is a project from DSC 80: Practice and Application of Data Science, exploring data regarding power outages.


## Introduction
In this project, we will be exploring a power outage dataset and focusing on intentional attacks.

This dataset contains information regarding 1500+ power outages across the US, from 2000-2016. It contains other information, such as the state it occurred in, the cause of the power outage, the state's real GSP that year, etc.. This project focuses on predicting if the cause category of an outage was an intentional attack or not. We also explore other aspects about power outages, such as looking at missingness in the dataset, or plotting different interesting distributions of data. This dataset and question are important to note because power outages affect entire communities, and especially as a Californian, we have experienced many. Additionally, intentional attacks causing power outages can harm millions of people, and different areas may be affected differently. There are 1534 rows of data, and 56 columns (but we will be keeping 8 and creating a new one).

- `US State:` The state that the power outage occurred in
- `Climate Category:` The climate category during the outage (normal, cold, or warm) (str)
- `Cause Category:` The reason for the power outage (str)
- `Outage Duration:` Duration of the power outage in minutes (int)
- `Demand Loss:` The amount of peak demand loss in Megawatts (int)
- `Customers Affected:` The number of customers affected by power outage (int)
- `Population:` Population at the given US state in a year (int)
- `Popden Rural:` Population density of the urban areas (persons per square mile) (float)
- `Attack:` Whether the cause of the outage was an intentional attack (bool)

## Data Cleaning and Exploratory Data Analysis
### Data Cleaning
The original dataset file was formatted oddly, with rows and columns in the Google Sheet that weren't actual data (and were just empty cells outside the actual table). I fixed it to include just the pure data, and also changed the column names for better readability, such as turning "CAUSE.CATEGORY" into "Cause Category". Then, I kept only the columns that we care about. This project focuses on predicting if a power outage was caused by an intentional attack, so I kept the columns that seemed correlated to this. I then added the column `Attack`, whose value is `True` if the `Cause Category` was "intentional attack".

`outages.head()`

|   Obs | US State   | Climate Category   | Cause Category     |   Outage Duration |   Demand Loss |   Customers Affected |   Population |   Popden Rural | Attack   |
|------:|:-----------|:-------------------|:-------------------|------------------:|--------------:|---------------------:|-------------:|---------------:|:---------|
|     1 | Minnesota  | normal             | severe weather     |              3060 |           nan |                70000 |      5348119 |           18.2 | False    |
|     2 | Minnesota  | normal             | intentional attack |                 1 |           nan |                  nan |      5457125 |           18.2 | True     |
|     3 | Minnesota  | cold               | severe weather     |              3000 |           nan |                70000 |      5310903 |           18.2 | False    |
|     4 | Minnesota  | normal             | severe weather     |              2550 |           nan |                68200 |      5380443 |           18.2 | False    |
|     5 | Minnesota  | warm               | severe weather     |              1740 |           250 |               250000 |      5489594 |           18.2 | False    |

### Univariate Analysis
Let's look at the number of power outages per state. It appears that California by far has the most!
<iframe
  src="assets/states_outage_nums.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

### Bivariate Analysis
California has a lot of power outages. But, is this just because it has the biggest population? Let's put it to scale. Let's find the number of outages per person. This allows us to compare states to each other at the same scale. We find that California doesn't have the biggest number of outages, per person (over the years). It seems like Delaware does!
<iframe
  src="assets/states_outage_pop.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

Also, let's look more closely at our `Attack` column. Do intentional attacks appear to happen evenly throughout states? Do specific states tend to have either really high or really low levels of intentional attacks, or is it closer to the average? 

This plot shows the "purity" of the `Attack` column with each state. Each value represents how far the proportion of intentional attacks are from 0.5. This means that larger values either have a high proportion of "intentional attacks", or a high proportion of other. Smaller values represent that the cause of power outages in that state (intentional attacks vs. other) are around equal.

It appears that a big chunk of the states either have super high or super low amounts of intentional attacks. There are a lot of values that are 0.5 (I will call this "completely pure," meaning either all their power outages were intentional attacks, or none of them were.). In fact, 10 states, or 20% of the states, are "completely pure", with a value of 0.5. This means that there may be some correlation between the state, and whether a power outage were an intentional attack. 

<iframe
  src="assets/attack_purity.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

### Interesting Aggregates
In this grouped table, we are comparing different quantitative values, grouped by whether the outage was caused by an intentional attack or not. There appears to be great differences between outages due to intentional attacks, and others (by eyeballing - we can't truly be sure if this difference is really significant, but I think it's a safe assumption).

| Attack   |   Outage Duration |   Demand Loss |   Customers Affected |   Population |
|:---------|------------------:|--------------:|---------------------:|-------------:|
| False    |           3449.96 |     687.716   |            175061    |  1.51104e+07 |
| True     |            429.98 |       9.15135 |              1790.53 |  8.07757e+06 |

In all of these, the average value is greater when it's not an intentional attack. Here is a table representing how much larger the value is for a non-intentional attack vs. an intentional attack.

|                    |   Multiplier |
|:-------------------|-------------:|
| Outage Duration    |      8.02353 |
| Demand Loss        |     75.1491  |
| Customers Affected |     97.7706  |
| Population         |      1.87066 |


## Assessment of Missingness
My data has a lot of missing values. Not missing at random (NMAR) means that there is missingness based on the values itself. Columns such as "Demand Loss Mw" and "Customers Affected" may be NMAR... For example, if the values were too big, or too small, it might've been harder to record, and may not have been reported in the first place. We would need to understand more about the situation to see if it's really NMAR (for example, learn more about how they measure demand loss or count the number of customers affected, and determine if it is harder to measure for certain scenarios)

Despite this, I wanted to run some tests to determine if those columns may be MCAR (Missing Completely at Random). I reasoned that if there is a greater rural po

Does the missingness of "Demand Loss" depend on "US State"? 
Null Hypothesis: the distribution of 'US State' when 'Demand Loss' is missing is the same as the distribution of 'US State' when 'Demand Loss' is not missing.
Alternative Hypothesis: these distributions are not the same

After running a permutation test with 1000 repetitions, we get a p-value of 0.0. We reject the null, and conclude that demand loss is likely MAR (Missing at Random) based on US State. This makes sense because different states may have different damage levels, due to the fact that some are more developed than others.

Empirical Distribution:
<iframe
  src="assets/missingness_tvd.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

Separated by State:
<iframe
  src="assets/missingness_by_state.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

## Hypothesis Testing
Are the demand losses from California and Washington the same? Do they come from the same population, or is one bigger than the other?
Null: "Demand Loss Mw" of California or Washington come from the same distribution 
Alternate: Whether a state is California or Washington doesn't have a relationship to "Outage Duration"
Test statistic: difference in group means. This is a good test statistic to use because it involves direction, and we are trying to see if one value is bigger than the other.
P-value threshold: 0.05
We fail to reject that they come from the same population, with a p-value of 0.146
<iframe
  src="assets/cali_wash.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

## Framing a Prediction Problem
I will predict if a power outage is an intentional attack or not. We will be performing binary classification. I chose to predict if a power outage is an intentional attack or not because there were a lot of outages classified as intentional attacks, and I think it is an interesting topic. I will be using F-score instead of accuracy because only 27% of the cause categories in my dataset are intentional attacks (but still will show the accuracy). Accuracy wouldn't be a good measure due to the disparity of the groups sizes. F-score is useful when data is imbalanced and takes into consideration False Negatives and False Positives (and not just True Positives).
- Even though we will be focusing on F-score, I will still include accuracy. We want it to be over 73% because the dumbest model (which would predict not "intentional attack" 100% of the time) would have an accuracy of 73%, which is meaningless.

## Baseline Model
To predict if a power outage is an intentional attack, we will fit a `DecisionTreeClassifier()`. My model uses the columns:
- `Outage Duration`: quantitative discrete
- `Population`: quantitative discrete
- `Demand Loss`: quantitative continuous
- `Climate Category`: qualitative nominal
- `Popden Rural`: quantitative continuous
These are all numerical values, besides "Climate Category,". I one-hot-encoded this column to make it quantitative so that we can use it in our model.

## Final Model
In addition to the features, I also added 'US State' (qualitative nominal) and `Customers Affected` (and took out population). Based on the analysis from earlier, it seemed like the state you're in may help contribute to predicting if an outage was due to an attack. Additionally, customers affected probably was kind of correlated to population, and the customers affected was a more specific version so I decided to swap them. Additionally, for the final model, I decided to use `RandomForestClasifier()` instead.
- `Outage Duration`: it would make sense if attacks may have similar outage durations -- maybe they are all similarly attacked
- `Customers Affected`: additionally, assuming that planned attacks are simliar to each other, it would probably affect similar amounts of customers (especially because they tend to happen in specific states)
- `Demand Loss`: assuming planned attacks have similar distributions, demand loss might look similar 
- `Climate Category`: more planned attacks may occur in specific climate categories
- `Popden Rural`:  the population density may affect if a planned attack occured - it wouldn't make sense for it to be a planned attack if there's no one there
- `US State`:  US states were found to often have either really high or really low amounts of planned attacks, making this a good indicator
I think changing these features improved my accuracy by being more specific, and adding more useful information.

Using a RandomForestClassifier() over a DecisionTreeClassifier() was beneficial because it is easy for decision trees to overfit, which we don't want! Additionally, random forests are better because they "vote" and are able to be more reliable. The hyperparameters that ended up being best were {'max_depth': 16, 'min_samples_split': 6, 'n_estimators': 105}. I selected these hyperparameters by running them under `GridSearchCV`, to find the best combination. It performed better in accuracy, but more importantly, F-score.

| Measure   |   Baseline |   Changed Features |   Final (Changed Hyperparameters) |
|:----------|-----------:|-------------------:|----------------------------------:|
| Accuracy  |   0.885417 |           0.914062 |                          0.914062 |
| F-Score   |   0.794393 |           0.84507  |                          0.846512 |


## Fairness Analysis
Does this model perform fairly for  higher/lower values of `Popden Rural`? We will use the mean of the entire dataset's `Popden Rural` column as a threshold to split values into high and low Popden Rural groups.
- Group X: Popden Rural > `outages["Popden Rural"].mean()` (39.47349081364819)
- Group Y: Popden Rural <= 39.47349081364819
- Null Hypothesis: Our model is fair. Its accuracy for lower and higher rural population densities are roughly the same, and any differences are due to random chance.
- Alternative Hypothesis: Our model is unfair. The accuracy for rural communities with higher population densities is higher than in rural areas with less dense populations.
- Evaluation Metric: Accuracy
- Test statistic: Differences in accuracy
- Significance level: p = 0.05
- P-value: 0.412
- **Conclusion**: With a p-value that is greater than 0.05, we are unable to reject the null. This means that we don't have significant information to prove that our model performs unequally between higher/lower Popden Rural groups. This meas that it likely achieves Demographic Parity.

<iframe
  src="assets/fairness.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>