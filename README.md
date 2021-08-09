# OpenTable Restaurant Review Predictor: Project Overview

## Summary
* Developed a machine learning model to help restaurant owners understand the features of successful restaurants and predict the review of their own business (~0.24 MAE based on 5 stars ratings).
* Built a multi-page webscraper using Python and bs4 to extract information of over 1000 Bay Area restaurants on popular reservations site OpenTable.
* Introduced median income figures by city to data and engineered review count and price features from OpenTable text.
* Conducted exploratory analysis on data to determine most relevant features.
* Fitted different models including multiple linear, lasso, and random forest regressors and used GridSearchCV to find the most accurate.
* Created an API for client input using Flask.

## Resources Used & Acknowledgements
**Python Version:** 3.8.3  
**Packages:** pandas, numpy, seaborns, scikit-learn, matplotlib, Flask, pickle, BeautifulSoup, json, requests   
**Web Framework Requirements:** listed in *Requirements.txt* -- Run `pip install -r requirements.txt`  
**Scraper Guidance:** https://betterprogramming.pub/how-to-scrape-multiple-pages-of-a-website-using-a-python-web-scraper-4e2c641cff8  
**Flask Model Production:** https://towardsdatascience.com/productionize-a-machine-learning-model-with-flask-and-heroku-8201260503d2  
**Project Guidance & Special Thanks:** Ken Jee https://github.com/PlayingNumbers/ds_salary_proj

## Multi-Page Web Scraping
One of the more challenging steps, building this scraper required disecting OpenTable's html and overall site structure to ensure compatibility.

Scraped Features:
* Name
* City
* Rating
* Review Count
* Promoted
* Price
* Cuisine

I appended median household income by city data as an extra feature to see if the wealth of a restaurant's location contributed to rating.

## Data Cleaning
I conducted much of the cleaning in the scraping process:  
* Converted presence of 'Promoted' to binary.  
* Converted '$' price system into 1-4 integers.
* Used regex and python to remove miscelanneous text from Rating, Review Count, City, and Median Income.
* Dropped restaurants with less than 50 reviews.

## Exploratory Data Analysis
I learned from looking at the distribution of ratings that OpenTable prevents restaurants from recieving less than 3 stars. This differs greatly from other review sites such as Yelp and Google Reviews.

![alt text](https://github.com/jcannice/ds_personal_proj/blob/main/Ratings%20Distribution.png)

This heat map shows that Review Count, Position on list, and Price are correlated with Rating.
![alt text](https://github.com/jcannice/ds_personal_proj/blob/main/Heatmap.png)

Unsurprising San Francisco hosts the greatest amount of restaurants.  
![alt text](https://github.com/jcannice/ds_personal_proj/blob/main/City%20Count.png)

Italian Cuisine and American cuisines are the most represented Bay Area businesses on OpenTable.  
![alt text](https://github.com/jcannice/ds_personal_proj/blob/main/Cuisine%20Count.png)

## Model Fitting & Performance
Since my data included categorical features, I converted them into dummy variables. I then made a 75/25 training/test split.  
I built and compared 3 models using a cross validation by MAE scorer. An MAE measure provided easily interpretable results given the 3-5 rating prediction. 
The models include multiple linear regression, lasso regression, and a random forest.

Lasso regression and random forest performed far better than the standard OLS model. This is due to the dummy variable induced sparsity of the data benefitting from the normalizing effect.

**Multiple Linear Regression:** ~1.9  
**Lasso Regression:** ~0.25  
**Random Forest by GridSearchCV:** ~0.24

## API Production
I wanted to make sure that my model was useful and accessible to others by building a Flask API endpoint which takes in a client's request with a list of values from a restaurant listing and returns a predicted rating.

## Web Application
To further the interpretability of the model, I build a simple web application using streamlit. Users can input their own feature parameters to guage the predicted success of their restaurant. The app is currently in deployment and will be available shortly here!

## Next Steps
**Data Engineering**  
While I scraped as much data as I could from the OpenTable site, I think many other features could be pulled and optimized from other sources. These might include health code scores/violations by restaurant from county data, yelp rating, and population size. Scraping the number of bookings a restaurant has recieved in a given day may also prove to be an interesting and potentially useful feature, although this metric changes throughout the day -- a longer term analysis would be needed to produce average booking numbers for each restaurant. Finally the cuisine type feature included over 100 types which could be reduced to much fewer to remove overlap.

**Model Selection**  
While lasso and random forest regression produced sufficient prediction results, exploring other model types such as support vector machines and other normalizing regressors would be worthwhile.


### Thanks for stopping by :)

