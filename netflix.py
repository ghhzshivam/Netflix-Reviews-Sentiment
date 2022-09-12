import streamlit as st
import pandas as pd 
import numpy as np
import joblib
import string

vec = open('transform.pkl','rb')
cv = joblib.load(vec)

model = open('naive_clf.pkl','rb')
clf = joblib.load(model)

st.title('Netflix Sentiment Analysis')


# @st.cache
def get_vect(txt):
    x = str(txt).lower()
    txt = [x]
    vect = cv.transform(txt).toarray()
    pred = clf.predict(vect)
    return pred


text = st.text_input('Write movie reviews here')
pred = get_vect(text)
if pred == 0:
    st.text('The sentiment predicted by the model is Negative sentiment')
elif pred ==1:
    st.text('The sentiment predicted by the model is Positive sentiment')


# sentiment = dic[pred[0]]
# st.text(f'The sentiment predicted by the model is {sentiment} sentiment')
