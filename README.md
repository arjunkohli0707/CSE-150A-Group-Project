# CSE-150A-Group-Project

## Abstract
Predicting Apple stock market trends using probabilistic models can assist investors in making informed decisions, mitigating financial risks, and optimizing investment strategies. This work proposes an AI agent utilizing the PEAS framework, evaluating performance through accuracy, precision, recall, and F1-score to ensure reliability. The agent operates in financial markets, using sensors (e.g., market indicators such as opening price, high, low, volume, and percentage change) to analyze historical stock data and actuators (e.g., buy, sell, or hold recommendations) to guide investment decisions. This work explores Bayesian networks (modeling relationships between stock indicators via probability distributions), Hidden Markov Models (capturing market state transitions such as bullish and bearish trends), and hybrid approaches integrating probabilistic reasoning with machine learning. The prediction tasks focus on classifying quarterly price changes into categories (e.g., increase or decrease) and further categorizing the magnitude of change as small or large. Challenges include handling market volatility, ensuring model robustness against sudden economic shifts, and addressing ethical concerns related to algorithmic trading and investor reliance on AI predictions. By balancing predictive accuracy with financial ethics, the agent could provide a data-driven approach to stock market trend classification.

# Preprocessing & Exploration

## Number of Observations

X_train shape: (9041, 10)  
X_test shape: (1808, 10)  
y_train shape: (9041, 2)  
y_test shape: (1808, 2)  

## Details About Your Data Distributions

The dataset contains stock price information, including open, high, low, closing prices, volume, and percentage changes.

## Normalizing and Cleaning Data

In the preprocessing stage:  
- The `Date` variable was converted into datetime format and sorted by date.  
- String variables such as `Change %` and metric abbreviations were converted, making the data ready to be trained.

## Scales

Time-based features such as `Year`, `Day`, `Day_of_week`, and `Quarter` were extracted into 5 bins for consistency and scaling.

## Missing Data

- Missing values were found in the `Vol.` column of the original data and corrupted data values.  
- These were identified and replaced with column means.  
- All data validity was confirmed using `df.isnull().sum()` to ensure all results are zeros.

## Column Descriptions

- **Date:** The date of the stock price entry.  
- **Price:** Closing price of the stock.  
- **Open:** Opening price of the stock.  
- **High:** Highest price during the day.  
- **Low:** Lowest price during the day.  
- **Vol.:** Trading volume, converted to numeric format.  
- **Change %:** Percentage change in stock price, converted to numeric format.  
- **Year:** Extracted year from the date.  
- **Day:** Extracted day from the date.  
- **Day_of_week:** Extracted day of the week (1-7).  
- **Quarter:** Extracted quarter of the year (1-4).  
- **Year_bin:** Quantile-based binning of years.  
- **Day_bin:** Quantile-based binning of days.  
- **Price_future_quarter:** Shifted price 63 days ahead to predict quarterly change.  
- **Quarterly_change_pct:** Percentage change in price over the next quarter.  
- **Direction_quarter:** Binary classification (1 = price increase, 0 = price decrease).  
- **Trend_class:** Multi-class classification of trend magnitude (encoded using LabelEncoder).

## Categorizing Numerical Data for Our Model

- The numerical data were categorized into time-based features such as `Year`, `Day`, `Day_of_week`, and `Quarter`.  
- This was done by creating bins for year and day using quantile-based discretization.

## Data Exploration
![Correlation Heatmap](correlation_heatmap.png)
![Box plot](box_plot.png)

From the box plot, we can see the distribution of each feature in comparison to the trend of the stock. From this graph we can see that the Price, Open, High, and Low features share very similar distribution to each of the trend categories. From the Correlation Heatmap, we can analyze the correlation between any 2 of our features to see that Price, Open, HIgh, and Low features share really strong relationship to the year while other features have little to no relationship. 


# Model Evaluation and Result Analysis

Evaluating our model led to the following results:

Direction_quarter Evaluation:
- Accuracy: 0.28
- Precision: 0.80
- Recall: 0.28
- F1 Score: 0.13
- Log Loss: 0.7365

Trend_class Evaluation:
- Accuracy: 0.33
- Precision: 0.44
- Recall: 0.33
- F1 Score: 0.30
- Log Loss: 4.4184


**Is our model good?**

Currently the model isn’t performing well on the data.
	
**If not, what is a sign we can see from the results**

The low accuracy (28% for Direction_quarter, 33% for Trend_class) and poor F1 scores (0.13 and 0.30) indicate weak predictive power. Additionally, the high log loss (4.4184 for Trend_class) suggests poor probability calibration. Despite high precision for Direction_quarter (80%), its recall is very low (28%), meaning it fails to identify many positive cases



