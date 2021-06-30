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

![alt text](
