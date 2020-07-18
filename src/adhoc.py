import pandas as pd
import config

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectPercentile, f_classif

vectorizer = TfidfVectorizer()

df_news = pd.read_csv(config.NEW_TRAINING_FILE).fillna("none").reset_index(drop=True)
print(df_news.count())       

X_train = df_news.clean_text.values
# Y_train = df_news.label.values

train_vectors = vectorizer.fit_transform(X_train)
feature_names = vectorizer.get_feature_names()
dense = train_vectors.todense()
denselist = dense.tolist()
df = pd.DataFrame(denselist, columns=feature_names)
print(df.head())

# selector = SelectPercentile(f_classif, percentile=10).fit_transform(X_train,Y_train)
# print(selector.shape)


