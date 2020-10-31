import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

def apply(question):
    pattern = '[0-9]'
    text = re.sub(pattern, '',question)

    text = text.lower()
    text = re.sub('<[^<]+?>', '', question)
    text = re.sub('nan', '', question)
    text = re.sub("(\\d|\\W)+", " ", question)

    return question

data = pd.read_csv('results.csv')
data['text'] = data['title'].astype(str) + data['body'].astype(str)
print(data['text'])

data['text'] = data['text'].apply(lambda x:apply(x))
print(data['text'])

texts = data['text'].tolist()

cv = CountVectorizer(max_df=0.7, lowercase = True, analyzer = 'word', stop_words='english')
word_count_matrix = cv.fit_transform(texts)
tf_idf = TfidfTransformer()
tf_idf.fit(word_count_matrix)

for i in range(len(data['text'])):

    question = [texts[i]]

    tf_idf_matrix = tf_idf.transform(cv.transform(question)).toarray()
    first_doc_scores = tf_idf_matrix[0]

    df_tf_idf = pd.DataFrame(first_doc_scores, index=cv.get_feature_names(), columns=['tf_idf_scores'])
    df_tf_idf=df_tf_idf.sort_values(by=['tf_idf_scores'])

    first = df_tf_idf.index[-1]
    second = df_tf_idf.index[-2]
    third = df_tf_idf.index[-3]
    print('Question number ', i,':', first,',' ,second,',', third)
