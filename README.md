# House-Price-Prediction-Python-
Prediction of House prices.

# Objective: 
### To determine the market price of a King County, USA house.</br>
Data: https://www.kaggle.com/datasets/harlfoxem/housesalesprediction</br>
Data Description:
  - Data from May 2014 and May 2015</br>
  - It contains house sale prices for King County, which includes Seattle


## What I've Learnt

# Module 1: Importing Data Sets
- df.dtypes
- df.describe()
![image](https://github.com/ParthM16/House-Price-Prediction-Python-/assets/136796479/a58715c4-c68f-4d6c-b782-22faa5af22cc)


# Module 2: Data Wrangling
- Changed "Date" column datatype
 ![image](https://github.com/ParthM16/House-Price-Prediction-Python-/assets/136796479/e2c1c3c3-7d60-4208-93df-392e8f858bf9)

- df.drop(['id', 'Unnamed: 0'], axis=1, inplace=True)
  ![image](https://github.com/ParthM16/House-Price-Prediction-Python-/assets/136796479/b095e095-78ed-4845-9000-1e4c15756d93)

- Is there any null value:
	df['bedrooms'].isnull().sum(): 
	df['bathrooms'].isnull().sum()
![image](https://github.com/ParthM16/House-Price-Prediction-Python-/assets/136796479/f85f36a0-84d4-493d-9ccb-e80f07028fcb)

- Replace with Mean</br>
	mean=df['bedrooms'].mean()</br>
	df['bedrooms'].replace(np.nan,mean, inplace=True)</br></br>
  mean=df['bathrooms'].mean()</br>
	df['bathrooms'].replace(np.nan,mean, inplace=True)
![image](https://github.com/ParthM16/House-Price-Prediction-Python-/assets/136796479/461041b1-be74-4b0e-9504-38db03fb3b7d)


# Module 3: Exploratory Data Analysis
- Unique value for a particular column and convert into dataframe:</br>
df['floors'].value_counts().to_frame()</br>
![image](https://github.com/ParthM16/House-Price-Prediction-Python-/assets/136796479/83ad9caa-86da-47fa-a93b-c6277b8fb4f1)


- Find outlier using boxplot from "seaborn" library:</br>
 e.g. - plt.figure(figsize=(10,6))</br>
	sns.boxplot(x='waterfront',y='price', data=df)</br>
	plt.xlabel('Waterfront View (0: No, 1: Yes)')</br>
	plt.ylabel('Price')</br>
	plt.title('Price distribution for houses w.r.t Waterfront')</br>
	plt.show()</br>
![image](https://github.com/ParthM16/House-Price-Prediction-Python-/assets/136796479/82a7c339-5745-4552-9a3f-4cc2b37f2486)


- To see correlation with all column with price column:</br>
	df.corr()['price'].sort_values()</br>
![image](https://github.com/ParthM16/House-Price-Prediction-Python-/assets/136796479/bbcbcde0-87eb-4d6e-a24e-66026750b101)


- Used regplot to see correlation between 'sqft_above' and 'price':</br>
 e.g. - sns.regplot(x='sqft_above', y='price', data=df)</br>
	plt.xlabel('sqft_above')</br>
	plt.ylabel('Price')</br>
	plt.title('sqft_above Vs Price')</br>
	plt.show()</br>
![image](https://github.com/ParthM16/House-Price-Prediction-Python-/assets/136796479/528f5d72-6d43-4fd7-a1bb-db03afd5c1d7)


# Module 4: Model Development
- Multiple Linear Regression, prediction, and R^2</br></br></br>
  - Multiple Linear Regression and R^2</br>
    X1= df[['grade','sqft_living','waterfront','view','bedrooms','sqft_above','bathrooms','yr_renovated','lat','condition']]</br>
    Y1= df[['price']]</br>
    Lr = LinearRegression()</br>
    Lr.fit(X1,Y1)</br>
    r_squared = Lr.score(X1, Y1)</br>
    r_squared</br></br>
    ![image](https://github.com/ParthM16/House-Price-Prediction-Python-/assets/136796479/c6918ca6-4923-4b75-b327-3cfbf6a2eac3)


  - Calculating best Alpha value for regression based on data:</br>
    - from sklearn.linear_model import Ridge</br>
      from sklearn.model_selection import GridSearchCV</br>

      alphas = np.logspace(-4, 4, 9)  # This will create alpha values from 0.0001 to 10000</br>
  
      param_grid = {'alpha': alphas}</br>
  
      ridge = Ridge()</br>
      grid_search = GridSearchCV(ridge, param_grid, cv=5, scoring='r2')</br>
      grid_search.fit(X1, Y1)</br>
  
      best_alpha = grid_search.best_params_['alpha']</br>
      print("Best Alpha:", best_alpha)</br>

      ![image](https://github.com/ParthM16/House-Price-Prediction-Python-/assets/136796479/8b6c9c00-6bd3-4a9d-a08f-11169a931293)

  - Calculating R^2 value for different Regression Models on entire data
      ![image](https://github.com/ParthM16/House-Price-Prediction-Python-/assets/136796479/898eeb7b-110e-43a8-abdf-ea1b4b66dffe)
      ![image](https://github.com/ParthM16/House-Price-Prediction-Python-/assets/136796479/e4bb047d-6b82-43a5-b5a5-215f1077ae21)


- **From above R^2 value for couple of regression model is >97%. So, there is a chance of overfitting. Let's split the data and evaluate the model**</br>


# Module 5: Model Evaluation and Refinement

- divided the dataset into training and testing: 
	![image](https://github.com/ParthM16/House-Price-Prediction-Python-/assets/136796479/e87d60dd-1682-4c81-94b4-1ca0cfd82da7)</br>
- Based on the output, I selected Random Forest and Gradient Boosting but there are chances of overfitting. So, I'm using Cross-Validation to predict and calculate MSE to get the best model for prediction.</br>
- GradientBoostingRegressor: </br>
  ![image](https://github.com/ParthM16/House-Price-Prediction-Python-/assets/136796479/1f9e1ce3-48f2-4fe5-94e1-e559f1177964)</br>
- Specific Record:</br>
![image](https://github.com/ParthM16/House-Price-Prediction-Python-/assets/136796479/45a7f469-2a9f-43dc-a74f-eeb0987f5ab6)

- RandomForestRegressor: </br>
  ![image](https://github.com/ParthM16/House-Price-Prediction-Python-/assets/136796479/b719cc5f-3109-4a34-9698-a7fdc443b70d)
</br>

# Conclusion

- After considering the results for <code>GradientBoostingRegressor</code> and <code>RandomForestRegressor</code>, we are moving forward with GradientBoostingRegressor because its R^2 value for train and test data is much closer than RandomForestRegressor's R^2 value.

- #### I achieved <code>81.90%</code> on train data and <code>79.39%</code> on test data.

- Average MSE (Cross-Validation): 23547104411.38543</br>
  Average R-squared (Cross-Validation): 0.8187851874637466  **(Train data: 81.87%)**</br>
  Mean Squared Error (Test Set): 30966330818.590614</br>
  R-squared (Test Set): <FONT COLOR="#ff0000">0.7951646761177257</FONT> **(Test data: 79.51%)**</br>


