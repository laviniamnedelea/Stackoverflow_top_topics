#Most used programming language

import numpy as np
import pandas as pd
import collections
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import re
data = pd.read_csv('users.csv')

from wordcloud import WordCloud, STOPWORDS

about = pd.Series(data['about_me'].tolist()).astype(str)
about = about.str.strip()

all = ' '.join(about.str.lower())

all = re.sub('<[^<]+?>', '', all)
prog_lang = ['python', 'c++', '.net', 'asp.net', 'bash','c', 'c#', 'r', 'sql', 'assembly' ,'perl', 'ruby', 'perl','html','css','java','javascript','go','swift','matlab','cobol', 'cobra','d','ibm','j','apache', 'jython','shell', 'php' ]
filtered_words = [word for word in all.split() if word in prog_lang]

counted_words = collections.Counter(filtered_words)

words = []
counts = []

cloud = WordCloud(width = 1400, height= 1080).generate(" ".join(filtered_words))

plt.figure(figsize=(20, 15))
plt.imshow(cloud)
plt.axis('off')
plt.show()


for letter, count in counted_words.most_common(10):
    words.append(letter)
    counts.append(count)
print(words)




colors = cm.rainbow(np.linspace(0, 1 , 3))
plt.figure(figsize=(30, 30))

plt.title('Top 10 programming languages their count')
plt.xlabel('Count')
plt.ylabel('Language')
plt.barh(words, counts, color=colors)
plt.show()

