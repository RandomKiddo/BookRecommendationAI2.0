import os

import pandas as pd
import kagglehub as kh

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from model_loader import *
from recommender import recommend_books

app = FastAPI() 

path = kh.dataset_download("mohamedbakhet/amazon-books-reviews")
df_ratings = pd.read_csv(os.path.join(path, 'Books_rating.csv'))
df_data = pd.read_csv(os.path.join(path, 'books_data.csv'))

df = pd.merge(
    df_ratings.drop(['Price', 'User_id', 'profileName', 'review/helpfulness',
                     'review/score', 'review/time', 'review/summary', 'review/text'], axis=1),
    df_data.drop(['image', 'previewLink', 'publishedDate', 'infoLink', 
                  'publisher', 'ratingsCount'], axis=1),
    on='Title', how='inner'
)
df = df.dropna(how='any', axis=0)
df = df.drop_duplicates(keep='first')

class UserRequest(BaseModel):
    review_test: str
    book_title: str

@app.post('/recommend')
def recommend(user_input: UserRequest):
    review = user_input.review_text
    book_title = user_input.book_title

    sentiment = predict_sentiment(review)

    matches = df[df['Title'].str.lower()==book_title.lower()]
    if matches.empty:
        raise HTTPException(status_code=404, detail='Book title not found.')
    book_id = matches.iloc[0]['Id']

    recommendations = recommend_books(book_id=book_id, sentiment=sentiment)

    return {
        'predicted_sentiment': int(sentiment),
        'recommendations': recommendations
    }
