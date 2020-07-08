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

rating = pd.DataFrame(book.groupby('ID')['Rating'].mean())
rating.head()
#the mean value for my first five observations
       Rating
ID           
8    5.571429
9    6.000000
10   6.000000
12  10.000000
14   5.333333

#getting the desription of the rating using describe() 
rating.describe()
            Rating
count  2182.000000
mean      7.440003
std       1.655948
min       1.000000
25%       6.500000
50%       7.666667
75%       8.500000
max      10.000000

from sklearn.feature_extraction.text import TfidfVectorizer #term frequencey- inverse document frequncy is a numerical statistic that is intended to reflect how important a word is to document in a collecion or corpus

# Creating a Tfidf Vectorizer to remove all stop words
tfidf = TfidfVectorizer(stop_words="english")    #
book["Title"].isnull().sum() 
book["Title"] = book["Title"].fillna(" ")

# Preparing the Tfidf matrix by fitting and transforming

tfidf_matrix = tfidf.fit_transform(book.Title)   #Transform a count matrix to a normalized tf or tf-idf representation
tfidf_matrix.shape

from sklearn.metrics.pairwise import linear_kernel

# Computing the cosine similarity on Tfidf matrix
cosine_sim_matrix = linear_kernel(tfidf_matrix,tfidf_matrix)

# creating a mapping of book name to index number 
book_index = pd.Series(book.index,index=book['Title']).drop_duplicates()
book_index["Classical Mythology"]
#0
def get_book_recommendations(Title,topN):
    
   
    #topN = 10
    # Getting the book index using its title 
    book_id = book_index[Title]
    
    # Getting the pair wise similarity score for all the book's with that 
    # book
    cosine_scores = list(enumerate(cosine_sim_matrix[book_id]))
    
    # Sorting the cosine_similarity scores based on scores 
    cosine_scores = sorted(cosine_scores,key=lambda x:x[1],reverse = True)
    
    # Get the scores of top 10 most similar book's 
    cosine_scores_10 = cosine_scores[0:topN+1]
    
    # Getting the book index 
    book_idx  =  [i[0] for i in cosine_scores_10]
    book_scores =  [i[1] for i in cosine_scores_10]
    
    # Similar book and Rating
    similar_books = pd.DataFrame(columns=["Title","Rating"])
    similar_books["Title"] = book.loc[book_idx,"Title"]
    similar_books["Rating"] = book_scores
    similar_books.reset_index(inplace=True)  
    similar_books.drop(["index"],axis=1,inplace=True)
    print (similar_books)
    # Getting the similar top5 books for Classical Mythology 
    get_book_recommendations("Classical Mythology",topN=5)
                                               Title    Rating
0                                Classical Mythology  1.000000
1                    Mythology 101 (Questar Fantasy)  0.364121
2  Celtic Mythology (Library of the World's Myths...  0.325074
3                                       Clara Callan  0.000000
4                               Decision in Normandy  0.000000
5  Flu: The Story of the Great Influenza Pandemic...  0.000000