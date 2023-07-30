*******README FILE*******


MADE BY:- 
	Diksha Sharma:- 21bce058
	Nupur Gandhi:- 21bce074


##### Overview


To be able to predict used cars market value can help both buyers and sellers. 
There are lots of individuals who are interested in the used car market at some points in their life because they wanted to sell their car or buy a used car.
In this process, it’s a big corner to pay too much or sell less then it’s market value.

In this Project, we are going to predict the Price of Used Cars using various features like Present_Price, Selling_Price, Kms_Driven, Fuel_Type, Year etc.
We have determined the price using three regression models i.e linear regression, random forest regression and gradient booster regression after cleaning the dataset.
Then on fitting the dataset into these models, we found out that the best model for car price prediction is gradient boosting regression model as it generates the leat error and gives the best graph.
Then we tried to take one datarow and manually fit it into the dataset.



##### Models Used


1) Extra Tree Regresser
2) Linear Regression
3) Random Forest Regression
4) Gradient boosting regression



##### Working of the code


We have written all the code in the .ipynb file.
To open it and run the code, open the .ipynb file on jupyter notebook or google collab.

Before running the code, it is essential to install the various packages used in the prediction model.
1)  Numpy :- pip install numpy
1)  Matplotlib :- pip install matplotlib
1)  Seaborn :- pip install seaborn
1)  Pandas :- pip install pandas
1)  Sklearn :- pip install sklearn

Moreover, the dataset used in the prediction modelhas been uploaded. Before running the ipynb file, make sure to keep both the dataset and code file
in the same folder or write the full pathname of the dataset in the code before running it. Else it will generate an error.

  

##### PROCEDURE 


In this project, we first prepared our dataset.
Then on analyzing the dataset, we found some anomalies and thus unsuitable to directly use the data to fit in the regression models.
So, we first explored the data and cleaned it using concerned functions according to our requirements to make it suitable to fit the regression models.


1) DATA PREPROCESSING

So, our first step was Data Preprocessing i.e analyzing the dataset on how it is present, the variables and it's description.
For that, first we imported the necessary packages and modules required. Then we loaded the dataset.
Then we did the following steps:-  
	a) Exploring the Descriptive Statistics of the variables.
	b) Drop features that are not required to build our model.
	c) Check for any missing value in data set and treat it.


2) DATA EXPLORATION

Our next step was analyzing how each variable in the dataset is present, spotting the anomalies.
For that the following steps was followed:-
	a) Exploring the probability distribution function(PDF):- 
		It will show the variables are distributed.
		The distribution function graph clearly shows us some outliers present in our concerned variables. So our next step is dealing with them.
	b) Dealing with the Outliers: 
		Here, we will remove the outliers from our dataset by dropping those odd values.
	c) Checking the linearity: 
		We will check the linearity of our cleaned data using sactter plot. 
		It shows how the dependent and independent variables are related and thus we can decide which regression model is suitable to fit in our dataset.
		It shows that our dependent variable is not linearly related to the independent variables thus to fit in the linear regression model, we need totransorm the dependent variable.  
	d) Transform independent variable using a log-transformation: 
		We transformed our independent variable using log to make it linearly dependent with the dependent variables.
		Then we dropped the previous independent variable and added the one after transforming to our cleaned data.
	e) Checking the multicollinearity using VIF: 
		We will check our variables for any possibilities of multicollinearity using the feature of VIF and fix them.
		Multicollinearity exists when there is a correlation between multiple independent variables in a multiple regression model. 
		This can adversely affect the regression results.A variance inflation factor (VIF) is a measure of the amount of multicollinearity in regression analysis.


3) FEATURE SELECTION

The goal of feature selection techniques in machine learning is to find the best set of features that allows one to build optimized models of studied phenomena.
In this section, we followed the following steps:-
	a) LabelEncoder() :- 
		We first our dataset into labelencoder function column by column and then checked the features correlated with the target variable.
	b) Heatmap:- 
		Then we generated a heatmap of our cleaned dataset.A heatmap is basically a graphical representation where individual values of a matrix are represented as colors.
		It is very useful in visualizing the concentration of values between two dimensions of a matrix. This helps in finding patterns and gives a perspective of depth. 
	c) Feature Importance:- 
		Feature importance gives a score for each feature of the data, the higher the score more important or relevant is the feature towards the Target variable.
		We used ExtraReggressor model to fit our dataset into it so as to find which feature is more dominant towards our target variable.
		Then we used a horizontal bar graph for its visualisation and sorted the values in descending order.
	d) Categorical features:-
		Our next step was converting the columns with categorial data i.e. non-numbers into numerical form so as to fit into our regression model.
		We used get_dummies function which assigned a number to each different types of values present in the dataset column and 
		then converted that row into multiple columns with values consisting of 0 for that property to not be relevant and 1 for it to be present. 


4) MODEL DEVELOPMENT:-

After the selection of important and relevant features our next step is developing the dataset according to our model needs.
The process is as follows:-
	a) Declaring dependent and independent variables:- 
		We decided upon the variables i.e. price is the dependent variable and all the other columns present in the cleaned dataset are independent variables.
	b) Feature scaling:- 
		In this step we standardised the range of our independent variables. Our dataset contained features that are varying in degrees of magnitude, range and units. 
		Therefore, in order for our regression models to interpret these features on the same scale, we performed feature scaling.
	c)Train and test data:-
		Next, we split our dataset into train and test datasets randomly.
		Training dataset is used to train our model i.e. fit the model. 
		Then we use test dataset to evaluate our dataset whether it can generalize well to the new or unseen dataset i.e. test set. 


5) LINEAR REGRESSION:-

	First we are using linear regression model. 
	Linear regression analysis is basically used to predict the value of a variable based on the value of another variable.

	First, we fit our training dataset into the model and then predicted the values of the dependent variable i.e. the price of the car by using it on test dataset.
	Then we found the a) r2 score
			  b) mean squared error 
			  c) mean absolute error 
	of our predicted model to determine whether this regression is working well on it or not.

6) RANDOM FOREST REGRESSION:-

	Next we are using random forest regression model. 
	Random Forest Regression is a supervised learning algorithm that uses ensemble learning method for regression. 
	It is a commonly-used machine learning algorithm which combines the output of multiple decision trees to reach a single result.

	First, we fit our training dataset into the model and then predicted the values of the dependent variable i.e. the price of the car by using it on test dataset.
	Then we found the a) r2 score
			  b) mean squared error 
			  c) mean absolute error 
	of our predicted model to determine whether this regression is working well on it or not.


7) GRADIENT BOOSTING REGRESSION:-

	And lastly, we are using gradient booster regression model. 
	Gradient boosting Regression calculates the difference between the current prediction and the known correct target value. 
	This difference is called residual. After that Gradient boosting Regression trains a weak model that maps features to that residual.
	Gradient boosting gives a prediction model in the form of an ensemble of weak prediction models, which are typically decision trees.

	First, we fit our training dataset into the model and then predicted the values of the dependent variable i.e. the price of the car by using it on test dataset.
	Then we found the a) r2 score
			  b) mean squared error 
			  c) mean absolute error 
	of our predicted model to determine whether this regression is working well on it or not.

8) BEST MODEL

	After working on all the three models. our next step is finding the model most suitable for our dataset.
	For this we plotted the graph of all the three regression models i.e. linear, random forest, gradient boosting.
	We plotted a scatter plot of actual vs predicted price .
	And in analysing the plots, we found that gradient booster is the most suitable for our dataset as it's scatter points are the most concentrated aling the linear line among the three .
	Thus, it produced the least error and gave us the most accurate predicted price.

9) MANUALLY CHECKING THE PREDICTIONS

	For manually checking the error in our predictions , we followed the following steps.
	First, to find the actual price, we took the exponential of the price column present in our dataset as we had log transformation on it during preparation of data.
	Then, we found the residual i.e. difference between the actual and predicted price and residuak in percentage.
	Then, we plotted it in the form of table to show clear results of our models.

10) CHECKING OUR MODEL

	And lastly, we took one row from our cleaned dataset and predicted it's price by directly fitting the row values.


##### CONCLUSION

Thus, we conclude that on our dataset the best model to predict the prices of the car is gradient boosting regression model.







