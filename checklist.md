# Model Review Checklist

## Data

- [ ] If data is pulled using SQL, are queries saved in `.sql` files and accessed as needed?

- [ ] Are queries parameterized?

- [ ] At what point in the data generating process will the model need to score new records? Is the training data constructed such that predictor variables are captured at the appropriate point in the process? Is there the possibility of data leakage / postdictors?

## Data Partition

- [ ] Are In-Time and Out-of-Time Holdout data set aside immediately and used only in final model evaluation?

- [ ] How is data partitioned for training? Common scenarios include k-fold cross-validation and splitting into a training, validation and test sets.

- [ ] If an entity can occur multiple times in the data, are all instances assigned to the same partition each time data is split?
      If an entity occurs in multiple splits, what issues might this cause?
	  
## Exploratory Data Analysis

- [ ] Is there a data dictionary? Are all fields clearly defined?

- [ ] What is the grain of the data? What does each observation represent?

- [ ] Are duplicate records present? If so, are they appropriate?

- [ ] Are there fields with a large number of missing/NULL values? Is there a reason for these values?

- [ ] Are variables plotted across time to identify any time-specific data quality issues?

- [ ] Are features highly correlated? Measures of correlation may include Pearson or Spearman correlation, Cramer's V, etc.

## Feature Engineering

- [ ] How are missing values handled for numeric variables? If imputation methods with learned parameters are used, are transformations repeatable? Are indicators created for these missing values?

- [ ] How are missing/NULL values handled for categorical variables?

- [ ] How are rare levels in categorical variables handled? Are transformations repeatable?

- [ ] How is freeform test handled?

  - [ ] Is string cleaning applied? This may include replacing special characters and/or punctuation, casing, etc.
  
  - [ ] What type of tokenization is applied?
  
  - [ ] Is stemming or lemmitization applied?
  
  - [ ] Does the model use bag of words, ngrams, word embeddings, etc?
  
- [ ] How are dates and timestamps used?

- [ ] How are categorical variables encoded?
      Common encodings include onehot/dummy encoding, feature hashing, ordinal encoding, supervised/effect encoding.
	  
  - [ ] If a supervised/effect encoding method is used, is target leakage a concern?
  
  - [ ] If the encoding has learned parameters, is it repeatable?
  
## Parameter Tuning

- [ ] What parameters does the model have? How are parameters tuned?

- [ ] How are data partitions used to select the final tuning parameters?


## Evaluation

- [ ] What metrics are used to evaluate the final model?