# Regression Project: Zillow Dataset

## Description:
You are a junior data scientist on the Zillow data science team and receive the following email in your inbox:

We want to be able to predict the property tax assessed values ('taxvaluedollarcnt') of Single Family Properties that had a transaction during 2017.

We have a model already, but we are hoping your insights can help us improve it. I need recommendations on a way to make a better model. Maybe you will create a new feature out of existing ones that works better, try a non-linear regression algorithm, or try to create a different model for each county. Whatever you find that works (or doesn't work) will be useful. Given you have just joined our team, we are excited to see your outside perspective.

One last thing, Maggie lost the email that told us where these properties were located. Ugh, Maggie :-/. Because property taxes are assessed at the county level, we would like to know what states and counties these are located in.

-- The Zillow Data Science Team

## Audience:
Your customer/end user is the Zillow Data Science Team. In your deliverables, be sure to re-state your goals, as if you were delivering this to Zillow. They have asked for something from you, and you are basically communicating in a more concise way, and very clearly, the goals as you understand them and how you have acted upon them through your research.

## Project Goal: 
- Construct an ML Regression model that predicts propery tax assessed values ('taxvaluedollarcnt') of Single Family Properties using attributes of the properties.

- Find the key drivers of property value for single family properties.

- Make recommendations on what works or doesn't work in predicting these homes' values.

## Deliverables: 
1. Github repo
    - Readme.md
    - Aquire & Perpare Modules
    - Final Report
2. Live presentation:
    - 5 minutes max
    
# How to use this repository: 
1. You can download the repository to your local device and run the all the files to understand the process.
    - I have created the notebook in a way where each step of the planning is laid out in a notebook to help with understanding:
        - Data Acquisition
        - Data Preparation
        - Exploratory analysis
        - Modeling
2. If you would like to replicate the project without downloading this repository:
    - Make sure to look at the files to understand what each function does.
    - You will have to copy the wrangle, explore, and model functions to your own python file and import the functions.
    - You will need access to the code-up sql data library with your own credentials.
    
    
# Data Dictoionary
|**Feature**|**Definition**|
|---|---|
|`transactiondate`| The date of the transaction. Year-Month-Day|
|`propertylandusetypeid`| The property type id. 261 is for single family properties|
|`propertylandusedesc`| The propety type desc. Only Single Family Properties|
|---|---|
|`bathrooms`| The number of bathrooms in the property|
|`bedrooms`| The number of bedrooms in the property|
|`sqft`| The sqaure footage of the property|
|`tax_value`| The tax value of the property. In dollars. |
|`yearbuilt`| When the property was built.|
|`house_age`| How olf the property is. |
|`bb_roomcnt`| The number of bedrooms plus the number of bathrooms|

# Project Planning: 
1. Data Acquisition:
    - Aquire the zillow dataset
2. Data Peperation:
    - Clean the dataframe:
        - Remove outliers
        - Nulls
        - Create a data dictionary
        - Create a graph of locations for Maggie
        - Split data into train, validate, test
3. Explore the data:
    - Look a univate stats: analysis of infividual data
    - Look at Bivariate stats: analysis of interactions between the target and features
    - look a multivare stats: analysis of interactions of 3+ variables
4. Modeling:
    - Feature engineering
    - Create a Ordinary Least Sqaures 
    - Lasso + Lars
    - Polynomial Regression
    - Generalized Linear Model
5. Create Report

# Where is each county located: 
|**Fips**|**County**|**State**|
|---|---|---|
|6037|Los Angeles|California|
|6059|Orange|California|
|6111|Ventura|California|

Graph showing where the majority of houses are compared to each county
![Alt text](image-4.png)



## Initial Questions Before looking into the data:
1. What is defined as a Single Family Property?
    - "a standalone, detached house used as a single dwelling unit"
2. Could how old the houses are play a factor into this?
3. Where are most houses located?
4. Between bedrooms and bathrooms, which has a higher drive for tax value?

## Initial questions after looking into the data: 
Could these features be factors? 
- `yearbuilt` : How old the house is?
- `lotsizesqaurefeet` : How big is the property?
- `numberofstories` : Does number of stories influence purchase? (need to remove outliers to make data normal)

Features that were looked into but don't look reliable(outliers)
- `fullbathcnt` : This is the same as bathroomcnt
- `roomcnt` : How many rooms are there, is this even actuarte? (Not reliable)



# Key findings: 
#### Univarite Summary: 
- The largest count of bathrooms 5.5. Data looks normally distributed.
- The largest count of bedrooms is 6. Data looks normally distributed.
- SQFT has a postive skew. The highest amount of SQFT is 5303!
- Year built: There are some old, OLD, houses! Who gets an **1878's** house?
- House age: Again, old houses. Youngest is 1 year old. The data has a negative skew.
- Bedroom+Bathroom count: The higest amount of rooms was 11.5.
- Target Varible: has a psotive skew. Higest value: **$1,829,696**. That is alot of money! 
#### Bi/Multivariate Summary:
- Based on the Heatmap: The top 3 features with the strongest correlation:
    - Sqft -> increase tax value
    - Bathrooms -> increase tax value
    - bb_count -> increase tax value
    - House_age <- deacrease tax value

## Hypothesis testing: 
- Hypothesis 1: Is there a realtionship between sqft and tax value?
    - $H_0$: There is no monotonic relationship between sqft and tax value.
    - $H_a$: There is a montonic relationship between sqft and tax value.
    - Answer: We reject H₀, there is a monotonic relationship.
- Hypothesis 2: Is there a realtionship between bathrooms and tax value?
    - $H_0$: There is no monotonic relationship between bathrooms and tax value.
    - $H_a$: There is a montonic relationship between bathrooms and tax value.
    - Answer: We reject H₀, there is a monotonic relationship.
- Hypothesis 3: Is there a realtionship between house_age and tax value?
    - $H_0$: There is no monotonic relationship between house age and tax value.
    - $H_a$: There is a montonic relationship between house age and tax value.
    - Answer : We reject H₀, there is a monotonic relationship.
- Hypothesis 4: Is there a realtionship between bb_roomcnt and tax value?
    - $H_0$: There is no monotonic relationship between bb_roomcnt and tax value.
    - $H_a$: There is a montonic relationship between bb_roomcnt and tax value.
    - Answer: We reject H₀, there is a monotonic relationship.
    
## Modeling 
- So based on explore our top 3 features are: 
    - SQFT
    - Bathrooms
    - House age
- Mean is the better baseline to work with
- RFE best features were:
    - Bathrooms
    - Year Built
    - House age
- Polynomial Regression preformed the best with our given data set
- The best features in explore did not end up being the best features in modeling.
- There is still a large mean sqaured error. 



# Conclusion: 
1. Summary:
    - Polynomial Regression was the best preforming model
    - SQFT, bedrooms, and bathrooms increase tax value
    - House age decreased tax value
2. Recomendations: 
    - I would consider not selling older houses. These houses tend to lower the amount of profit that can be made of. 
    - I would recomend advertising larger homes or smaller homes with a large number of bathrooms. It seems that bathrooms are very sought after comidity.
3. Next steps:
    - I would take a look at the data relative the the locations of the house. Specifically each county that they are in.
    - I would also take more time and add new features, such as how many stories each single residental property has. If the propety has a pool, and so on.