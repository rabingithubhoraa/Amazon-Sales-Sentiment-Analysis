# Amazon Product Analytics: Sentiment Analysis and Predictive Modeling

## Overview

This project analyzes Amazon product and customer-review data using exploratory data analysis, statistical modeling, machine learning, and natural language processing (NLP).

The study examines relationships among product prices, discounts, ratings, product categories, and customer-review sentiment. It also evaluates several regression approaches for predicting product discount percentages.

This repository is based on my Master's Data Science capstone project and has been organized as a reproducible research portfolio demonstrating applied data analysis, machine learning, NLP, and business analytics.

---

## Research Motivation

Online marketplaces generate large amounts of both structured and unstructured data.

Structured information such as product prices, ratings, discounts, and product categories can provide insights into pricing patterns and product characteristics, while customer reviews provide textual information about customer experiences and opinions.

This project combines these two forms of data to explore how statistical analysis, machine learning, and NLP can be used to better understand Amazon product and review data and support data-driven business analysis.

---

## Research Questions

The project focuses on the following questions:

1. What patterns can be identified in Amazon product prices, discounts, ratings, and product categories?

2. What sentiment patterns emerge from customer review titles and review content?

3. What relationships exist among product ratings, actual prices, and discount percentages?

4. How accurately can discount percentage be predicted using product price, rating, and category information?

5. Do ensemble machine-learning approaches improve predictive performance relative to multiple linear regression?

---

## Dataset

The dataset was obtained from Kaggle and contains Amazon product and customer-review information.

**Dataset size:** 1,465 products × 16 original variables

Key variables include:

- Product ID
- Product name
- Product category
- Actual price
- Discounted price
- Discount percentage
- Product rating
- Rating count
- Product description
- Review title
- Review content
- User information
- Product and image links

The combination of structured product attributes and unstructured review text makes the dataset useful for investigating questions at the intersection of data science, NLP, and business analytics.

---

## Methodology

### 1. Data Cleaning and Preparation

The original dataset required several preprocessing steps before analysis.

These included:

- Converting price variables from Indian rupee-formatted strings to numerical values
- Converting discount percentages to numerical decimal values
- Correcting an anomalous value in the rating variable
- Converting rating counts to numerical format
- Examining missing values
- Filling missing rating-count values for modeling
- Separating hierarchical product categories into main and subcategories
- Creating a price-difference feature
- One-hot encoding product categories for predictive modeling

---

### 2. Exploratory Data Analysis

Exploratory analysis was conducted to investigate product, pricing, rating, and category patterns.

The analysis includes:

- Distribution of products across categories
- Product-rating distributions
- Actual versus discounted prices
- Discount-percentage distributions
- Most and least expensive products
- Products with the largest price differences
- Ratings by product category
- Rating-count distributions
- Category-level price comparisons
- Correlation analysis among numerical variables
- Scatter plots examining price, rating, and discount relationships

These analyses provide context for the subsequent statistical and machine-learning models.

---

## Sentiment Analysis

Customer review titles and review content were examined using multiple NLP approaches.

### TextBlob

TextBlob was used to calculate:

- Review-title polarity
- Review-title subjectivity
- Review-content polarity
- Review-content subjectivity

The resulting distributions were visualized to examine overall patterns in customer opinion.

### VADER Sentiment Analysis

VADER sentiment analysis was also applied to customer review content.

The analysis generated:

- Positive sentiment scores
- Neutral sentiment scores
- Negative sentiment scores
- Compound sentiment scores

The average VADER compound score was approximately **0.82**, indicating that review text in the dataset generally exhibits positive sentiment.

Word clouds were also generated to explore commonly occurring terms across different sentiment ranges.

### Experimental BERT Analysis

The notebook additionally explores transformer-based sentiment classification using the pretrained:

`nlptown/bert-base-multilingual-uncased-sentiment`

This component is treated as an exploratory extension. The construction of comparison labels requires further refinement before BERT classification accuracy can be interpreted as a validated model-performance result.

---

## Predictive Modeling

The primary predictive task examines whether product discount percentage can be estimated using:

- Actual price
- Product rating
- Main product category

Several modeling approaches were evaluated.

### Multiple Linear Regression

A multiple linear regression model was first used as an interpretable statistical baseline.

**Test R²:** `0.1564`

The relatively modest R² indicates that price, rating, and product category explain only part of the variation in discount percentage.

---

### Random Forest Regression

A Random Forest model was evaluated to capture nonlinear relationships that may not be represented adequately by linear regression.

**Baseline Random Forest R²:** `0.2274`

The Random Forest improved upon the linear regression baseline, suggesting the presence of nonlinear relationships in the data.

---

### Hyperparameter-Tuned Random Forest

Grid search with cross-validation was used to investigate Random Forest hyperparameters.

**Tuned Random Forest R²:** `0.3063`

The tuned model improved predictive performance compared with both the linear regression and baseline Random Forest models.

---

### Gradient Boosting Regression

Gradient Boosting was also evaluated using hyperparameter search.

**Tuned Gradient Boosting R²:** `0.3223`

Among the overall models evaluated in the notebook, Gradient Boosting produced the strongest test-set R².

| Model | Test R² |
|---|---:|
| Multiple Linear Regression | 0.1564 |
| Random Forest | 0.2274 |
| Tuned Random Forest | 0.3063 |
| Tuned Gradient Boosting | **0.3223** |

The results suggest that ensemble methods capture more of the variation in discount percentage than the linear model, although the remaining unexplained variance indicates that additional predictors would likely be necessary for stronger predictive performance.

---

## Category-Level Modeling

Additional models were developed separately for product categories with sufficient observations.

For category-specific Random Forest models, test performance varied substantially:

| Category | Test R² |
|---|---:|
| Computers & Accessories | **0.4195** |
| Electronics | 0.0400 |
| Office Products | -0.2919 |
| Home & Kitchen | 0.0319 |

These results demonstrate that predictive relationships are not consistent across product categories.

The stronger performance for Computers & Accessories suggests that price and rating information may contain more predictive information for discounting within that category, while weak or negative R² values for other categories indicate that additional variables are needed.

---

## Key Findings

The analysis produced several important observations:

- Amazon product categories differ substantially in price, discount, and rating characteristics.
- Customer-review text generally displays positive sentiment according to the lexicon-based sentiment analysis.
- Product rating generally shows a negative relationship with discount percentage in several analyses.
- Actual price has relatively limited linear predictive influence on discount percentage.
- Random Forest improved predictive performance over multiple linear regression.
- Hyperparameter tuning further improved Random Forest performance.
- Gradient Boosting achieved the strongest overall test R² among the models evaluated.
- Predictive performance varies considerably across product categories.
- Price, rating, and category alone are insufficient to fully explain discounting behavior, suggesting opportunities for richer feature development.

---

## Limitations and Future Work

This analysis also highlights several opportunities for further research.

Future work could include:

- Refining sentiment-label construction for supervised NLP evaluation
- Improving transformer-based sentiment classification
- Incorporating additional product and seller characteristics
- Developing more advanced text features from customer reviews
- Comparing additional ensemble and boosting methods
- Using explainable AI techniques to interpret model predictions
- Investigating category-specific pricing and discount strategies
- Expanding the analysis to larger Amazon datasets
- Conducting more rigorous cross-validation and model-comparison experiments

These extensions could strengthen both predictive performance and the business interpretation of the results.

---

## Repository Structure

```text
Amazon-Sales-Sentiment-Analysis/
│
├── data/
│   └── raw/
│       └── amazon.csv
│
├── figures/
│
├── notebooks/
│   └── Amazon_Sales_Sentiment_Analysis.ipynb
│
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
