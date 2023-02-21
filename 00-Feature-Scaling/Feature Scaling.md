# Feature Scaling

## What is Feature Scaling?

> Feature Scaling is the process of bringing all of our features to the same or very similar ranges of values or distribution. — **Machine Learning Engineering by Andriy Burkov** [URL](http://mlebook.com/)

## Why do we need Feature Scaling?

- Most of the Machine Learning Algorithms show `significantly better results` when the features are transformed into the same or very similar range, `i.e. a fixed scale`.

- To understand the importance of feature scaling, we are going to use the `diabetes` dataset from the [Source](https://www4.stat.ncsu.edu/~boos/var.select/diabetes.html).

- Here we are considering a problem of estimating a quantitative measure of diabetes disease progression one year after baseline using the ten baseline variables, `age`, `sex`, `body mass index`, `average blood pressure`, and six `blood serum measurements`.

- **Dataset Description**
    - `age`: age in years
    - `sex` : gender
    - `bmi`: body mass index
    - `bp`: average blood pressure
    - `s1`: tc, total serum cholesterol
    - `s2`: ldl, low-density lipoproteins
    - `s3`: hdl, high-density lipoproteins
    - `s4`: tch, total cholesterol / HDL
    - `s5`: ltg, possibly log of serum triglycerides level
    - `s6`: glu, blood sugar level
    - `target`: a quantitative measure of diabetes disease progression one year after baseline

### Import Necessary Packages


```python
# Import Necessary Packages
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
```

### Dataset Description


```python
# Load the Dataset
X, targets = load_diabetes(
    return_X_y=True, # Return Input Features and Target
    as_frame=True, # Return Input Features and Target as Pandas Dataframe
    scaled=False # Return Input Features and Target is NOT Scaled
)
print(f"The input features are of type {type(X)}")
print(f"The target is of type {type(targets)}")
```

    The input features are of type <class 'pandas.core.frame.DataFrame'>
    The target is of type <class 'pandas.core.series.Series'>



```python
# Check a Sample from the Dataset
X.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>sex</th>
      <th>bmi</th>
      <th>bp</th>
      <th>s1</th>
      <th>s2</th>
      <th>s3</th>
      <th>s4</th>
      <th>s5</th>
      <th>s6</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>59.0</td>
      <td>2.0</td>
      <td>32.1</td>
      <td>101.0</td>
      <td>157.0</td>
      <td>93.2</td>
      <td>38.0</td>
      <td>4.00</td>
      <td>4.8598</td>
      <td>87.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>48.0</td>
      <td>1.0</td>
      <td>21.6</td>
      <td>87.0</td>
      <td>183.0</td>
      <td>103.2</td>
      <td>70.0</td>
      <td>3.00</td>
      <td>3.8918</td>
      <td>69.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>72.0</td>
      <td>2.0</td>
      <td>30.5</td>
      <td>93.0</td>
      <td>156.0</td>
      <td>93.6</td>
      <td>41.0</td>
      <td>4.00</td>
      <td>4.6728</td>
      <td>85.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>24.0</td>
      <td>1.0</td>
      <td>25.3</td>
      <td>84.0</td>
      <td>198.0</td>
      <td>131.4</td>
      <td>40.0</td>
      <td>5.00</td>
      <td>4.8903</td>
      <td>89.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>50.0</td>
      <td>1.0</td>
      <td>23.0</td>
      <td>101.0</td>
      <td>192.0</td>
      <td>125.4</td>
      <td>52.0</td>
      <td>4.00</td>
      <td>4.2905</td>
      <td>80.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>23.0</td>
      <td>1.0</td>
      <td>22.6</td>
      <td>89.0</td>
      <td>139.0</td>
      <td>64.8</td>
      <td>61.0</td>
      <td>2.00</td>
      <td>4.1897</td>
      <td>68.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>36.0</td>
      <td>2.0</td>
      <td>22.0</td>
      <td>90.0</td>
      <td>160.0</td>
      <td>99.6</td>
      <td>50.0</td>
      <td>3.00</td>
      <td>3.9512</td>
      <td>82.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>66.0</td>
      <td>2.0</td>
      <td>26.2</td>
      <td>114.0</td>
      <td>255.0</td>
      <td>185.0</td>
      <td>56.0</td>
      <td>4.55</td>
      <td>4.2485</td>
      <td>92.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>60.0</td>
      <td>2.0</td>
      <td>32.1</td>
      <td>83.0</td>
      <td>179.0</td>
      <td>119.4</td>
      <td>42.0</td>
      <td>4.00</td>
      <td>4.4773</td>
      <td>94.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>29.0</td>
      <td>1.0</td>
      <td>30.0</td>
      <td>85.0</td>
      <td>180.0</td>
      <td>93.4</td>
      <td>43.0</td>
      <td>4.00</td>
      <td>5.3845</td>
      <td>88.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Check Descriptive Statistics of the Dataset
X.describe().T
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>age</th>
      <td>442.0</td>
      <td>48.518100</td>
      <td>13.109028</td>
      <td>19.0000</td>
      <td>38.2500</td>
      <td>50.00000</td>
      <td>59.0000</td>
      <td>79.000</td>
    </tr>
    <tr>
      <th>sex</th>
      <td>442.0</td>
      <td>1.468326</td>
      <td>0.499561</td>
      <td>1.0000</td>
      <td>1.0000</td>
      <td>1.00000</td>
      <td>2.0000</td>
      <td>2.000</td>
    </tr>
    <tr>
      <th>bmi</th>
      <td>442.0</td>
      <td>26.375792</td>
      <td>4.418122</td>
      <td>18.0000</td>
      <td>23.2000</td>
      <td>25.70000</td>
      <td>29.2750</td>
      <td>42.200</td>
    </tr>
    <tr>
      <th>bp</th>
      <td>442.0</td>
      <td>94.647014</td>
      <td>13.831283</td>
      <td>62.0000</td>
      <td>84.0000</td>
      <td>93.00000</td>
      <td>105.0000</td>
      <td>133.000</td>
    </tr>
    <tr>
      <th>s1</th>
      <td>442.0</td>
      <td>189.140271</td>
      <td>34.608052</td>
      <td>97.0000</td>
      <td>164.2500</td>
      <td>186.00000</td>
      <td>209.7500</td>
      <td>301.000</td>
    </tr>
    <tr>
      <th>s2</th>
      <td>442.0</td>
      <td>115.439140</td>
      <td>30.413081</td>
      <td>41.6000</td>
      <td>96.0500</td>
      <td>113.00000</td>
      <td>134.5000</td>
      <td>242.400</td>
    </tr>
    <tr>
      <th>s3</th>
      <td>442.0</td>
      <td>49.788462</td>
      <td>12.934202</td>
      <td>22.0000</td>
      <td>40.2500</td>
      <td>48.00000</td>
      <td>57.7500</td>
      <td>99.000</td>
    </tr>
    <tr>
      <th>s4</th>
      <td>442.0</td>
      <td>4.070249</td>
      <td>1.290450</td>
      <td>2.0000</td>
      <td>3.0000</td>
      <td>4.00000</td>
      <td>5.0000</td>
      <td>9.090</td>
    </tr>
    <tr>
      <th>s5</th>
      <td>442.0</td>
      <td>4.641411</td>
      <td>0.522391</td>
      <td>3.2581</td>
      <td>4.2767</td>
      <td>4.62005</td>
      <td>4.9972</td>
      <td>6.107</td>
    </tr>
    <tr>
      <th>s6</th>
      <td>442.0</td>
      <td>91.260181</td>
      <td>11.496335</td>
      <td>58.0000</td>
      <td>83.2500</td>
      <td>91.00000</td>
      <td>98.0000</td>
      <td>124.000</td>
    </tr>
  </tbody>
</table>
</div>




```python
list(X['age'][0:10])
```




    [59.0, 48.0, 72.0, 24.0, 50.0, 23.0, 36.0, 66.0, 60.0, 29.0]



- Few observations from the descriptive statistics
    - The `age` feature is in range [18, 79], indicating patients ranging from 18 to 79 years old. 
    - The `bmi` feature is in range [18, 42], indicating patients with body mass index of 18 to 42.
    - The `s1` feature is in range [97, 301], indicating patients with total serum cholesterol of 97 to 301.
    - ...
- As we can see here, every feature has a different range. 
- When we use these features to build a Machine Learning model, the learning algorithm won’t differentiate that values 18-79 and 97-301 represent two different things `age` and `s1-total serum cholesterol`. It will end up treating them both as numbers. 
- As the numbers for total serum cholesterol i.e. 97-301 are much bigger in value compared to the numbers representing the age, the learning algorithm might end up giving more importance to total serum cholesterol over the age, `regardless of which variable is actually more helpful` in generating predictions. 
- **To avoid such an issue we prefer to transform the features into the same or very similar range, `i.e. a fixed scale`.**

## Different Types of Feature Scaling

- **`Normalization (Min-Max Scaling)`** and **`Standardization (Standard Scaling)`** are the two of the most widely used methods for feature scaling. 
- Normalization transforms each feature to a range of [0 - 1]. On the other hand, standardization scales each input variable by subtracting the mean and dividing by the standard deviation, resulting in a distribution (almost!) with a mean of zero and a standard deviation of one. 

### Normalization

Let's consider a sample from our dataset with $age = \{59.0, 48.0, 72.0, 24.0, 50.0, 23.0, 36.0, 66.0, 60.0, 29.0\}$. We can use following equation to Normalize the data. After Normalization, our sample transformed to $age = \{0.73, 0.51, 1.0, 0.02, 0.55, 0.0, 0.27, 0.88, 0.76, 0.12\}$. 

$$s' = \frac{s - \min(S)}{\max(S) - \min(S)} $$


```python
age_sample = list(X['age'][:10])
normalized_age = [((age - min(age_sample))/(max(age_sample) - min(age_sample))) for age in age_sample]
normalized_age = [round(age, 2) for age in normalized_age]
print(f"First 10 Age Values before Normalization: {age_sample}")
print(f"First 10 Age Values after Normalization: {normalized_age}")
```

    First 10 Age Values before Normalization: [59.0, 48.0, 72.0, 24.0, 50.0, 23.0, 36.0, 66.0, 60.0, 29.0]
    First 10 Age Values after Normalization: [0.73, 0.51, 1.0, 0.02, 0.55, 0.0, 0.27, 0.88, 0.76, 0.12]


- In this example, we can see that the data point with 72 years of age is scaled to 1.0 as 72 is the maximum number out of those 10 samples of age from our dataset. Similarly data point with 23 years of age is scaled to 0.0 as 23 is the minimum number out of those 10 samples of age from our dataset. 
- Here key point to observe is that the scaling operation is executed based on the minimum and maximum of those 10 samples and not all the samples of the dataset.

- Instead of the range $[0, 1]$, if we are interested to transform in some arbitrary range $[a, b]$ we can use following equation to Normalize the data.

$$s' = a + \frac{\big(s - \min(S)\big) \big(b - a\big)}{\max(S) - \min(S)} 

For example, we can transform $age = \{59.0, 48.0, 72.0, 24.0, 50.0, 23.0, 36.0, 66.0, 60.0, 29.0\}$ in the range $[-1, 1]$ to get scaled dataset $age = \{0.47, 0.02, 1.0, -0.96, 0.1, -1.0, -0.47, 0.76, 0.51, -0.76\}$.


```python
a, b = -1, 1
age_sample = list(X['age'][:10])
normalized_age = []
for age in age_sample:
    numerator = (age - min(age_sample))*(b - a)
    denominator = max(age_sample) - min(age_sample)
    normalized_age.append(a + (numerator/denominator))
normalized_age = [round(a, 2) for a in normalized_age]
print(f"First 10 Age Values before Normalization: {age_sample}")
print(f"First 10 Age Values after Normalization: {normalized_age}")
```

    First 10 Age Values before Normalization: [59.0, 48.0, 72.0, 24.0, 50.0, 23.0, 36.0, 66.0, 60.0, 29.0]
    First 10 Age Values after Normalization: [0.47, 0.02, 1.0, -0.96, 0.1, -1.0, -0.47, 0.76, 0.51, -0.76]


- In this example, we can see that the data point with 72 years of age is scaled to 1.0 as 72 is the maximum number out of those 10 samples of age from our dataset. Similarly data point with 23 years of age is scaled to -1.0 as 23 is the minimum number out of those 10 samples of age from our dataset. 
- Here key point to observe is that the scaling operation is executed based on the minimum and maximum of those 10 samples and not all the samples of the dataset.

### Standardization

Let's consider a sample from our dataset with $age = \{59.0, 48.0, 72.0, 24.0, 50.0, 23.0, 36.0, 66.0, 60.0, 29.0\}$. We can use following equation to Standardize the data. After Standardization, our sample transformed to $age = \{0.73, 0.08, 1.5, -1.34, 0.2, -1.4, -0.63, 1.14, 0.79, -1.05\}$. 

$$s' = \frac{s - mean(S)}{std(S)}$$


```python
age_sample = list(X['age'][:10])
standardize_age = [((age - np.average(age_sample))/np.std(age_sample)) for age in age_sample]
standardize_age = [round(age, 2) for age in standardize_age]
print(f"First 10 Age Values before Standardization: {age_sample}")
print(f"First 10 Age Values after Standardization: {standardize_age}")
```

    First 10 Age Values before Standardization: [59.0, 48.0, 72.0, 24.0, 50.0, 23.0, 36.0, 66.0, 60.0, 29.0]
    First 10 Age Values after Standardization: [0.73, 0.08, 1.5, -1.34, 0.2, -1.4, -0.63, 1.14, 0.79, -1.05]


### Robust Scaling

- Standardization scales the data such that the mean of values after scaling becomes zero and the standard deviation of values after scaling becomes one. This way it transforms the data such that it follows the `standard normal distribution`. 
- It uses `mean` and `standard deviation` of original data to perform scaling. Usually Mean and Standard Deviation is very sensitive to **`outliers`**.

> **`outliers`**. are the values on the edge of the distribution that may have a low probability of occurrence, yet are overrepresented for some reason. Outliers can skew a probability distribution and make data scaling using standardization difficult as the calculated mean and standard deviation will be skewed by the presence of the outliers. $-$ Jason Brownlee from Machine Learning Mastery [URL](https://machinelearningmastery.com/robust-scaler-transforms-for-machine-learning/).

- `Meadian` i.e. `50th Percentile` is less sensitive to outliers and similarly `Inter-Quartile Range (IQR)` i.e. `IQR = (75th Percentile - 25th Percentile)` is also less sensitive to outliers. 
- Robust Scaling uses Median and IQR to scale the data.

$$s' = \frac{s - median(S)}{IQR(S)}$$

Let's consider a sample from our dataset with $age = \{59.0, 48.0, 72.0, 24.0, 50.0, 23.0, 36.0, 66.0, 60.0, 29.0, 8.0, 10.0, 5.0\}$. We can use the above equation to scale the data. After scaling, our sample transformed to $age = \{0.64, 0.33, 1.0, -0.33, 0.39, -0.36, 0.0, 0.83, 0.67, -0.19, -0.78, -0.72, -0.86\}$. 

Here the key point to observe is that we have purposefully added three some outlier samples (8.0, 10.0, and 5.0)  in the `age` feature.


```python
age_sample = list(X['age'][:10])
age_sample.extend([8.0, 10.0, 5.0])
IQR = np.subtract(*np.percentile(age_sample, [75, 25]))
robust_scaled_age = [((age - np.median(age_sample))/IQR) for age in age_sample]
robust_scaled_age = [round(age, 2) for age in robust_scaled_age]
print(f"First 10 Age Values before Robust Scaling: {age_sample}")
print(f"First 10 Age Values after Robust Scaling: {robust_scaled_age}")
```

    First 10 Age Values before Robust Scaling: [59.0, 48.0, 72.0, 24.0, 50.0, 23.0, 36.0, 66.0, 60.0, 29.0, 8.0, 10.0, 5.0]
    First 10 Age Values after Robust Scaling: [0.64, 0.33, 1.0, -0.33, 0.39, -0.36, 0.0, 0.83, 0.67, -0.19, -0.78, -0.72, -0.86]


## How to Choose Scaling Type?

Even though there are no fix rules for selecting a particular scaler, broadly the selection depends on `Outliers` and `Understanding of Features`.

The selection of feature scaling depends on couple of factors:

1. **Understanding of Features**
    - There some features where `Min` and `Max` values from the dataset might not correspond to the actual possible `Min` and `Max` values for a feature. From statistical perspective, `Min` and `Max` of sample doesn't always guarantees a good estimation of the `Min` and `Max` of population. In such cases, `Standardization` or `RobustScaling` would be a better choice over `Normalization`. 
    
    - For example, in our dataset, the minimum `age` is 19 years. We are intending to use this dataset to build a model which can predict a quantitative measure of diabetes disease progression one year after baseline. If we use `Normalization` for scaling, we are assuming that we will always receive patients aged 19 years or older. In future, if we receive a patient who is younger than the 19 years, the scaled `age` value for that patient will be a negative number and doesn't align with the original idea of scaling `age` in range [0, 1]. 

    - This can negatively impact the predictions of the model as model has never seen a data sample with negative age value during training process. 
    
    - Similarly, the maximum `age` in our dataset is 79 years. If we receive a patient who is older than the 79 years, the scaled `age` value for that patient will be a greater than 1 which doesn't align with the original idea of scaling `age` in range [0, 1]. 
    
    - On other end, there could be features where it is easy to estimate `Min` and `Max` of population just from the sample. For example any form of customer star ratings is usually represented in the range of [0 - 5] stars. Here there is no scope of receiving a rating less than 0 or more than 5. In this case it is easy to estimate `Min` and `Max` of population and could be based on our understanding of the feature. In such cases, we can use `Normalization` for scaling. Digital Images are another such example, where we can use Normalization to scale the data.

2. **Outliers**
    - Usually descriptive statistics such `Min`, `Max`, `Mean` and `Standard deviation` are very sensitive to outliers and can change significantly by a small presence of outliers in the data. On the other end, descriptive statistics such as `Median` and `Inter-Quartile Range` is less sensitive to outliers.
    - Robust Scaler is one which uses `Median` and `Inter-Quartile Range` to scale the data. Because of this it is less sensitive to outliers.
    - So if the input features has significantly higher number of outliers, it is always better to use Robust Scaler.

## Impact of Scaling

In this section we will compare different types of scaler on our dataset.


```python
def plot_scaling_comparison(data, scaled_data, column, title):
    fig, axs = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=(8, 8),
        gridspec_kw={"height_ratios": (.20, .80)},
        dpi=100,
        constrained_layout=False
    )
    fig.suptitle(title)
    
    bplot = sns.boxplot(data=data, x=column, ax=axs[0][0])
    hplot = sns.histplot(data=data, x=column, ax=axs[1][0], kde=True, bins='sqrt')
    hplot.vlines(x=[np.mean(data[column]), np.median(data[column])], ymin=hplot.get_ylim()[0], ymax=hplot.get_ylim()[1], ls='--', colors=['tab:green', 'tab:red'], lw=2)
    
    bplot = sns.boxplot(data=scaled_data, x=column, ax=axs[0][1])
    hplot = sns.histplot(data=scaled_data, x=column, ax=axs[1][1], kde=True, bins='sqrt')
    hplot.vlines(x=[np.mean(scaled_data[column]), np.median(scaled_data[column])], ymin=hplot.get_ylim()[
                 0], ymax=hplot.get_ylim()[1], ls='--', colors=['tab:green', 'tab:red'], lw=2)
    
    axs[0][0].set(xlabel='')
    axs[0][0].set_facecolor('white')
    axs[1][0].set_facecolor('white')
    axs[0][1].set(xlabel='')
    axs[0][1].set_facecolor('white')
    axs[1][1].set_facecolor('white')
   

```


```python
normalization_scaler = MinMaxScaler()
normalized_X = pd.DataFrame(normalization_scaler.fit_transform(X), columns=X.columns)

standard_scaler = StandardScaler()
standardized_X = pd.DataFrame(standard_scaler.fit_transform(X), columns=X.columns)

robust_scaler = RobustScaler()
robust_scaled_X = pd.DataFrame(
    robust_scaler.fit_transform(X), columns=X.columns)
```


```python
normalized_X.describe().T
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>age</th>
      <td>442.0</td>
      <td>0.491968</td>
      <td>0.218484</td>
      <td>0.0</td>
      <td>0.320833</td>
      <td>0.516667</td>
      <td>0.666667</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>sex</th>
      <td>442.0</td>
      <td>0.468326</td>
      <td>0.499561</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>bmi</th>
      <td>442.0</td>
      <td>0.346107</td>
      <td>0.182567</td>
      <td>0.0</td>
      <td>0.214876</td>
      <td>0.318182</td>
      <td>0.465909</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>bp</th>
      <td>442.0</td>
      <td>0.459817</td>
      <td>0.194807</td>
      <td>0.0</td>
      <td>0.309859</td>
      <td>0.436620</td>
      <td>0.605634</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>s1</th>
      <td>442.0</td>
      <td>0.451668</td>
      <td>0.169647</td>
      <td>0.0</td>
      <td>0.329657</td>
      <td>0.436275</td>
      <td>0.552696</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>s2</th>
      <td>442.0</td>
      <td>0.367725</td>
      <td>0.151460</td>
      <td>0.0</td>
      <td>0.271165</td>
      <td>0.355578</td>
      <td>0.462649</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>s3</th>
      <td>442.0</td>
      <td>0.360889</td>
      <td>0.167977</td>
      <td>0.0</td>
      <td>0.237013</td>
      <td>0.337662</td>
      <td>0.464286</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>s4</th>
      <td>442.0</td>
      <td>0.291996</td>
      <td>0.182010</td>
      <td>0.0</td>
      <td>0.141044</td>
      <td>0.282087</td>
      <td>0.423131</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>s5</th>
      <td>442.0</td>
      <td>0.485560</td>
      <td>0.183366</td>
      <td>0.0</td>
      <td>0.357542</td>
      <td>0.478062</td>
      <td>0.610446</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>s6</th>
      <td>442.0</td>
      <td>0.503942</td>
      <td>0.174187</td>
      <td>0.0</td>
      <td>0.382576</td>
      <td>0.500000</td>
      <td>0.606061</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
standardized_X.describe().T
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>age</th>
      <td>442.0</td>
      <td>8.037814e-18</td>
      <td>1.001133</td>
      <td>-2.254290</td>
      <td>-0.784172</td>
      <td>0.113172</td>
      <td>0.800500</td>
      <td>2.327895</td>
    </tr>
    <tr>
      <th>sex</th>
      <td>442.0</td>
      <td>1.607563e-16</td>
      <td>1.001133</td>
      <td>-0.938537</td>
      <td>-0.938537</td>
      <td>-0.938537</td>
      <td>1.065488</td>
      <td>1.065488</td>
    </tr>
    <tr>
      <th>bmi</th>
      <td>442.0</td>
      <td>1.004727e-16</td>
      <td>1.001133</td>
      <td>-1.897929</td>
      <td>-0.719625</td>
      <td>-0.153132</td>
      <td>0.656952</td>
      <td>3.585718</td>
    </tr>
    <tr>
      <th>bp</th>
      <td>442.0</td>
      <td>1.060991e-15</td>
      <td>1.001133</td>
      <td>-2.363050</td>
      <td>-0.770650</td>
      <td>-0.119214</td>
      <td>0.749368</td>
      <td>2.776058</td>
    </tr>
    <tr>
      <th>s1</th>
      <td>442.0</td>
      <td>-2.893613e-16</td>
      <td>1.001133</td>
      <td>-2.665411</td>
      <td>-0.720020</td>
      <td>-0.090841</td>
      <td>0.596193</td>
      <td>3.235851</td>
    </tr>
    <tr>
      <th>s2</th>
      <td>442.0</td>
      <td>-1.245861e-16</td>
      <td>1.001133</td>
      <td>-2.430626</td>
      <td>-0.638249</td>
      <td>-0.080291</td>
      <td>0.627442</td>
      <td>4.179278</td>
    </tr>
    <tr>
      <th>s3</th>
      <td>442.0</td>
      <td>-1.326239e-16</td>
      <td>1.001133</td>
      <td>-2.150883</td>
      <td>-0.738296</td>
      <td>-0.138431</td>
      <td>0.616239</td>
      <td>3.809072</td>
    </tr>
    <tr>
      <th>s4</th>
      <td>442.0</td>
      <td>-1.446806e-16</td>
      <td>1.001133</td>
      <td>-1.606102</td>
      <td>-0.830301</td>
      <td>-0.054499</td>
      <td>0.721302</td>
      <td>3.894331</td>
    </tr>
    <tr>
      <th>s5</th>
      <td>442.0</td>
      <td>2.250588e-16</td>
      <td>1.001133</td>
      <td>-2.651040</td>
      <td>-0.698949</td>
      <td>-0.040937</td>
      <td>0.681851</td>
      <td>2.808722</td>
    </tr>
    <tr>
      <th>s6</th>
      <td>442.0</td>
      <td>2.371155e-16</td>
      <td>1.001133</td>
      <td>-2.896390</td>
      <td>-0.697549</td>
      <td>-0.022657</td>
      <td>0.586922</td>
      <td>2.851075</td>
    </tr>
  </tbody>
</table>
</div>




```python
robust_scaled_X.describe().T
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>age</th>
      <td>442.0</td>
      <td>-0.071417</td>
      <td>0.631760</td>
      <td>-1.493976</td>
      <td>-0.566265</td>
      <td>0.0</td>
      <td>0.433735</td>
      <td>1.397590</td>
    </tr>
    <tr>
      <th>sex</th>
      <td>442.0</td>
      <td>0.468326</td>
      <td>0.499561</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>bmi</th>
      <td>442.0</td>
      <td>0.111241</td>
      <td>0.727263</td>
      <td>-1.267490</td>
      <td>-0.411523</td>
      <td>0.0</td>
      <td>0.588477</td>
      <td>2.716049</td>
    </tr>
    <tr>
      <th>bp</th>
      <td>442.0</td>
      <td>0.078429</td>
      <td>0.658633</td>
      <td>-1.476190</td>
      <td>-0.428571</td>
      <td>0.0</td>
      <td>0.571429</td>
      <td>1.904762</td>
    </tr>
    <tr>
      <th>s1</th>
      <td>442.0</td>
      <td>0.069017</td>
      <td>0.760617</td>
      <td>-1.956044</td>
      <td>-0.478022</td>
      <td>0.0</td>
      <td>0.521978</td>
      <td>2.527473</td>
    </tr>
    <tr>
      <th>s2</th>
      <td>442.0</td>
      <td>0.063437</td>
      <td>0.790977</td>
      <td>-1.856957</td>
      <td>-0.440832</td>
      <td>0.0</td>
      <td>0.559168</td>
      <td>3.365410</td>
    </tr>
    <tr>
      <th>s3</th>
      <td>442.0</td>
      <td>0.102198</td>
      <td>0.739097</td>
      <td>-1.485714</td>
      <td>-0.442857</td>
      <td>0.0</td>
      <td>0.557143</td>
      <td>2.914286</td>
    </tr>
    <tr>
      <th>s4</th>
      <td>442.0</td>
      <td>0.035124</td>
      <td>0.645225</td>
      <td>-1.000000</td>
      <td>-0.500000</td>
      <td>0.0</td>
      <td>0.500000</td>
      <td>2.545000</td>
    </tr>
    <tr>
      <th>s5</th>
      <td>442.0</td>
      <td>0.029647</td>
      <td>0.725039</td>
      <td>-1.890285</td>
      <td>-0.476544</td>
      <td>0.0</td>
      <td>0.523456</td>
      <td>2.063775</td>
    </tr>
    <tr>
      <th>s6</th>
      <td>442.0</td>
      <td>0.017639</td>
      <td>0.779413</td>
      <td>-2.237288</td>
      <td>-0.525424</td>
      <td>0.0</td>
      <td>0.474576</td>
      <td>2.237288</td>
    </tr>
  </tbody>
</table>
</div>




```python
column='bmi'
```


```python
plot_scaling_comparison(X, normalized_X, column=column, title="Original Data - Normalized Data")

```


    
![png](Feature%20Scaling_files/Feature%20Scaling_46_0.png)
    



```python
plot_scaling_comparison(X, standardized_X, column=column,
                        title="Original Data - Standardized Data")

```


    
![png](Feature%20Scaling_files/Feature%20Scaling_47_0.png)
    



```python
plot_scaling_comparison(X, robust_scaled_X, column=column,
                        title="Original Data - Robust Scaled Data")

```


    
![png](Feature%20Scaling_files/Feature%20Scaling_48_0.png)
    

