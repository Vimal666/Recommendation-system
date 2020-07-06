# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 11:03:18 2020

@author: Vimal PM
"""

#importing the libraries
import pandas as pd
import numpy as np
#Loading the dataset using pd.read_csv()
book=pd.read_csv("D://DATA SCIENCE//ASSIGNMENT//ASSIGNMENT10//book.csv",encoding='ISO-8859-1')
book.shape
#(10000, 3)
book.columns
#Index(['ID', 'Title', 'Rating'], dtype='object')
book.Rating #the first and last five observation of rating variable from my dataset
0       5
1       3
2       6
3       8
4       6
       ..
9995    7
9996    9
9997    7
9998    8
9999    6

*******************************************************************************
#Creating a new dataframe  called rating_count and inside this dataset am just adding only the variables called 'ID','Rating'...
rating_count = pd.DataFrame(book, columns=['ID','Rating'])
# Sorting and dropping the duplicates
rating_count.sort_values('Rating', ascending=False).drop_duplicates().head(10)
          ID  Rating
7785    3943      10
2318  278750      10
2325  278772      10
5634    2453      10
2340  278807      10
2341  278818      10
2348  278831      10
2349  278832      10
5608    2442      10
2362  278843      10
#Above shows the best 10 books ID's and it's ratings
#Next I would like to create a dataframe for my  best five  books ID 
most_rated_books = pd.DataFrame([3943, 278750, 278772, 2453, 278807], index=np.arange(5), columns=['ID'])
#Here am merging the above dataset to my original dataset called book. 
#And I'm going to see the best five books based on the rating 
detail = pd.merge(most_rated_books, book, on='ID')
detail
       ID                                              Title  Rating
0    3943                                        The Bunyans      10
1  278750         Sit &amp; Solve - Lateral Thinking Puzzles      10
2  278772                Feeling Good : The New Mood Therapy      10
3    2453  O Little Town Of Glory  (Men Of Glory) (Harleq...      10
4  278807  Women Can't Hear What Men Don't Say: Destroyin...      10
#Above shows the best five books based the ratings