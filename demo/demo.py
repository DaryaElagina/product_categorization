from joblib import load
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

category_model = load('model_svc.joblib')
category_vectorizer = load('vectorizer.joblib')
data_cats = pd.read_csv('./data/categories.csv')

print('.: Данная модель призвана определить категорию введенного товара :.')
print('Введите название товара: ')
product_name = input()

print('Вы ввели: ', product_name)


df_new = pd.DataFrame([[product_name]], columns=['title'])
data_to_predict = category_vectorizer.transform(df_new['title'].fillna(''))
predictions = category_model.predict(data_to_predict)

predicted_cat = data_cats[data_cats['category_id'] == predictions[0]]

print('Определена категория: ')
print(predicted_cat)
