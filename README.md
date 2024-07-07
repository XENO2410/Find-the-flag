# Find the Flag!

## Overview
In this project, we use decision trees to predict the continent of flags based on several features. We'll explore which features are the best to use and the best way to create the decision tree.
![image](https://github.com/XENO2410/Find-the-flag/assets/97669140/3a260c3c-b36f-48ea-8a45-a39b8a726cd4)


## Datasets
The original dataset is available at the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Flags).

### Investigate the Data
1. The dataset is loaded as a DataFrame named `df`. Some of the input and output features include:
   - `name`: Name of the country
   - `landmass`: 1=N.America, 2=S.America, 3=Europe, 4=Africa, 5=Asia, 6=Oceania
   - `bars`: Number of vertical bars in the flag
   - `stripes`: Number of horizontal stripes in the flag
   - `colours`: Number of different colours in the flag
   - `red`: 0 if red absent, 1 if red present in the flag
   - `mainhue`: Predominant colour in the flag
   - `circles`: Number of circles in the flag
   - `crosses`: Number of upright crosses
   - `saltires`: Number of diagonal crosses
   - `quarters`: Number of quartered sections
   - `sunstars`: Number of sun or star symbols

   Calculate the count of flags by landmass using `df['landmass'].value_counts()`.

2. Focus on Europe and Oceania. Create a new DataFrame with flags from these continents using `df["landmass"].isin([3,6])`.

3. Print the average values of predictors for these two continents using `.groupby('landmass')[var].mean()`.

4. Inspect the variable types for each of the predictors using `df_36[var].dtypes`.

5. Transform the dataset of predictor variables to dummy variables and save this in a new DataFrame called `data`.

### Split the Data
6. Split the data into a train and test set using `train_test_split(X, y, random_state=1, test_size=.4)`.

### Tune Decision Tree Classifiers by Depth
7. Fit a decision tree classifier for `max_depth` values from 1-20. Save the accuracy score for each depth in the list `acc_depth`.

8. Plot the accuracy of the decision tree models versus the `max_depth`.

9. Find the largest accuracy and the depth this occurs using `np.max(acc_depth)`.

10. Refit the decision tree model using the `max_depth` from above and plot the decision tree.

### Tune Decision Tree Classifiers by Pruning
11. Tune the tree by using the hyperparameter `ccp_alpha`, a pruning parameter. Fit a decision tree classifier for each value in `ccp` and save the accuracy score in the list `acc_pruned`.

12. Plot the accuracy of the decision tree models versus the `ccp_alpha`.

13. Find the largest accuracy and the `ccp_alpha` value this occurs.

14. Fit a decision tree model with the values for `max_depth` and `ccp_alpha` found above and plot the final decision tree.

15. Note that the accuracy of the final model increased, and the structure of the tree is simpler. Many unnecessary branches were removed in the pruning process.

## Extensions
- Try to classify another feature, such as the "Language".
- Find a subset of features that work better.
- Tune more parameters of the model. Consider parameters like `max_leaf_nodes`.

