# Income-Classification-Prediction-Model

Description of the problem
The problem presented by the dataset is one that everyone at one point will face. That is
earning money. Everyone would rather earn more than less. By building this model one can
predict how likely they are to earn a satisfying amount and if not one could adjust their
future strategy for earning money. Of course one can’t change some things about themselve,
but how much those things impact your future, isn’t that clear. By creating this model we can
help less wealthy people come to necessary conclusions which will help them shape a better
future for themselves.

Data
Data has been provided by Lorenzo De Tomasi on the kaggle website.
https://www.kaggle.com/datasets/lodetomasi1995/income-classification/data
The dataset contains information about the Census Income.
The dataset contains 32,561 rows of entries.
In this dataset 14 explanatory variables can be found:

● age: Age of the person

● workclass: the type of work the person is doing

● fnlwgt: (final weight) the number of people the census believes the entry represents

● education: the level of education the person stopped at

● education-num: value representing education entry in a numerical form

● marital-status: the marital status of the person (if they are married or not)

● occupation: what kind of work the person is doing

● relationship: relatives of the person

● race: the race of the person

● sex: the race of the person

● capital-gain: capital gain of the person

● capital-loss: capital loss of the person

● hours-per-week: hours the person works during the week

● native-country: where the person is from

Based on this data we will be able to build a predictive model allowing us to predict whether
a person has an income lower or higher than 50K. However before building the model we
need to prepare the dataset to get the most accurate results possible.

A way to solve the problem
1. Data set analysis
At the beginning the data was cleaned and prepared for later training.
Firstly the whole dataset has been analyzed, errors and missing values were found,
which is important as thanks to this they were properly dealt with.
A new column named high_income has been created, based on the income column, it
features False, True boolean values as this is the column which we are going to
predict. False means that the person has income <=50K and True means the income
is >50K.

![image](https://github.com/user-attachments/assets/e42c09fa-342d-4619-8a05-6f98e9c28507)

In the column “workclass”, rows with values: “Without-pay” or “Never-worked” were
deemed unnecessary and removed from the dataset. There were only 2-3 rows
containing such values so they wouldn’t impact the predictions in any major way, as
well if someone never got paid it is obvious that their income would be below 50K
and if by some chance they had income more than 50K we could safely say that it’s an
anomaly, which won’t help in our predictions.

![image](https://github.com/user-attachments/assets/36426b89-b41f-4d90-ac38-fbec49fc023d)

Column “fnlwgt” has been dropped. It is a number which is mostly a unique value
and has little to none correlation between it and income.

![image](https://github.com/user-attachments/assets/028034f3-6f86-4095-8b82-edc23e743203)

Column “Education” has been dropped as there exists an Education-num column
which mirrors this column and is much more useful than the “education” feature.

![image](https://github.com/user-attachments/assets/2b021b26-9cb7-49e3-950c-b697eb467ddf)

It was found that columns: workclass, occupation and native-country had “?” values
which would be considered as the missing values. There were 585, 585 and 181
missing values like that respectively.
To fix this problem first “?” values were changed to Nan values so that Dataiku could
recognize them as missing values.

![image](https://github.com/user-attachments/assets/1ef49084-5f1d-43dc-a933-543e64fe4736)

After that the missing values were filled by Filling the empty cells in column with
Mode (most frequent value).

![image](https://github.com/user-attachments/assets/ccec4c9d-03d9-4929-a867-928cfd9ea14f)

3. Trimming outliers
In the search for outliers the scatter plot has been used, which is located in the chart
tab.
● any capital gain greater than 28000 is considered an outlier.

![image](https://github.com/user-attachments/assets/70799a2f-3a54-4fc4-bbbf-434b272ef232)

● any capital loss greater than 2600 is considered an outlier

![image](https://github.com/user-attachments/assets/736efd57-f775-42f7-b1b8-c46ce3d496f5)

● Considering that people usually spend at least 20 hours per week on
work - it seems highly suspicious that someone working below that
amount of time can achieve high income. Of course there may be
wealthy people or maybe children of wealthy families. However, this
kind of data does not seem to be important.
Solution to this was to create a filter recipe with following filter
conditions.

![image](https://github.com/user-attachments/assets/9123f391-3d11-4f0b-8ed2-7ef7a0a95cbe)

![image](https://github.com/user-attachments/assets/f8525a39-7f3e-44bd-8f93-3d94181ebf61)

● Next step is to remove outliers in the form of people older than 85.

![image](https://github.com/user-attachments/assets/ab3e452e-5acc-48f6-8317-64e8755d4fb3)

In order to fix that, a condition in the filter recipe was introduced.

![image](https://github.com/user-attachments/assets/06792dfc-40eb-4b91-bf90-638dea42790d)


5. Feature engineering
a) For further research it is wise to extend the categorisation with
grouping people by their age with:
● young adults being younger than 25,
● adults younger than 60,
● and older people as elders.
![image](https://github.com/user-attachments/assets/9c16d1d7-2375-4a83-b0f3-b70d347028bd)

b) For more clarity and easier understanding of capital loss and capital
gain, two new tables were created: capital-loss_simple and
capital-gain_simple respectively. Capital loss is divided in three
groups: low(0-1200), medium(1200-1800) and high(<1800), as for
capital gain it’s true or false these are based on statistics shows below.
![image](https://github.com/user-attachments/assets/e6137072-46f2-4e1e-a10a-312f67f342a7)

![image](https://github.com/user-attachments/assets/e1dd157a-2342-4e21-9289-24fa6f2a6449)


7. Metadata editing
Dataiku correctly assigned proper meanings and types. No further action was
required.

8. Missing data completion
The missing data has been dealt with.
In workclass, occupation and native-country columns, missing rows have been filled
with the most frequently occurring values.

9. Selection of learning algorithm
The newly created column “high_income” from the dataset has been chosen, as the
attribute on which to create the prediction model.
The algorithms chosen for the initial automatic training were: Random forest,
Gradient Boosted Trees, Logistic Regression, LightGBM, XGBoost and Extra trees.
There was no special feature handling done to the model at that point, except turning
off the “income” column which is the same as the “high-income” column and was left
in the dataset only as reference for the latter column). The training shows that the
algorithm yielding the highest accuracy is LightGBM and XGBoost, both with ROC
AUC value of 0.905 and showing that the most important features are: marital-status,
education-num, age-range, occupation, hours-per-week and relationship in
LightGBM and capital-gain-simple in XGBoost.
![image](https://github.com/user-attachments/assets/6ea9295b-eecb-48e7-8896-9ec769e5cd26)

At this point the model has been evaluated and further improved by adjusting the
feature handling settings.
![image](https://github.com/user-attachments/assets/8c3c0d77-1614-45e4-bc14-699215ee9f15)

![image](https://github.com/user-attachments/assets/2e335d00-a9da-4403-a3e6-2292c6db7eda)

The results we got were fairly satisfactory if not for the precision of the confusion matrix.
That is why we attempted to run further sessions without marital status and/or relationship
features to check whether the importance of being married(as seen on the plot above) is not
causing too many predicted true results that are actually false. However the results were
even worse without taking those features into account and therefore it was decided that the
results cannot be improved with such actions. Similarly using only five most important
features also did not improve the precision score, neither did it help with the overall score.

Session 2
The next session commenced with the following setup.
Marital status was removed due to too big of an influence on the results. As shown on
previous screenshots.
Extra trees and Random Forest algorithms were not used - reason is, these
algorithms were the ‘weakest’, they had the lowest accuracy. While, Gradient Boosted
Trees (which is also a tree-based algorithm) was the best amongst all tree algorithms.
The model was set with algorithms : Gradient Boosted Trees, Logistic Regression,
LightGBM, XGBoost. Features turned off - marital status.
Session 2 results are quite similar to the initial session. First of all XGBoost wins but
nearly ties with LightGBM and with Gradient Boosted Trees. ROC AUC score is still
pretty suspiciously high, varying between 0.898 (Logistic Regression) and 0.902
(XGBoost).
Taking a deep-dive into the results of XGBoost it looks like the algorithm results has
some flaws. Predicted False, actually True (342) vs actually False (3932) seems
rational, but predicted True: actually True (1194) vs actually False (693) result is
poor. It does resemble the confusion matrix from the previous session, but
worse.Maybe it can be improved in the next session.
![image](https://github.com/user-attachments/assets/7a898970-acf6-4c14-8528-9ffb96fe5a14)

Moving to feature importance. Relationship (32%) reigns over all other features. The
second most influential feature is educational-num (15%). Relationship feature is
double the importance than the educational-num.
From the feature dependency plot it seems that when being a wife or husband in
relationship status, immediately sky-rockets the influence on high income.
![image](https://github.com/user-attachments/assets/a374c1bd-67dd-4807-b89a-f2bf8c304901)

On the graph the ‘others’ contains ‘wife’ values which are represented by dots that are
above zero Shapley value.
This session results in the relationship being a replacement for the turned-off marital
status feature, in a way. In the later session it might be a good idea to turn off
relationship and observe the outcomes.
It is safe to make an assumption that married couples are capable of sustaining high
income. While education (surprisingly) takes second place in importance. Obviously
better education may lead to better earnings.





























