import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
import pickle
from xgboost import XGBRegressor
from nltk.corpus import stopwords
import nltk
from sklearn_pandas import DataFrameMapper
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
import logging
import config
import nltk
from nltk.corpus import stopwords
import string
from nltk import word_tokenize

import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = list(stopwords.words('english'))
from nltk.stem import WordNetLemmatizer 
lemmatizer = WordNetLemmatizer()  

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

class FeatureCreation():
    def __init__(self):
        pass

    def processText(self,df):
        by_article_list=[]
        for article in (df["text"]):
            words = word_tokenize(article)
            # words = [word.lower() for word in words if word.isalpha()] #lowercase
            words = [word for word in words if word not in string.punctuation and word not in stop_words] #punctuation, stopwords
            words = [lemmatizer.lemmatize(word) for word in words] #convert word to root form
            by_article_list.append(' '.join(words))
        return by_article_list

        

    def generate_features(self, data):
        """
        This function transforms data into set of features that are domain agnostic

        Feature list ===>
                'ActionCount, VERBRatio', 'NOUNRatio', 'last_activity_len',
                'ProblemCount', 'activity_count', 'ADJRatio', 'acronym_to_activity_ratio',
                'is_len_range_1_500', 'PUNCRatio', 'NUMRatio', 'ADVPNRatio', 'u_sla_status',
                'acronym count', 'num value count', 'kb count', 'ip address count', 'server name count'
        """
        logging.info("Generating Features")

        def get_word_count(column, analyzer):
            """
            Calculate the frequently used words from the entire bag of words using Count Vectorizer
            """
            count_vect = CountVectorizer(analyzer=analyzer)
            bag_of_words = count_vect.fit_transform(data[column])
            sum_of_words = bag_of_words.sum(axis=0)
            words_freq = [(word, sum_of_words[0, i])
                          for word, i in count_vect.vocabulary_.items()]
            df = pd.DataFrame(words_freq, columns=['word', 'count'])
            return df

        def get_acronym_count(text):
            """
            Return the count of all the uppercase words
            Based on the assumption that if the word is in uppercase, it will be an Acronym.
            """

            text = str(text)
            count = 0
            for word in text.split():
                if word.isupper():
                    count += 1
            return count

        def get_url_count(text):
            """
            Return the count of URLs found in the text
            """
            return len(re.findall('https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+', text))

       

        def get_numerical_value_count(text):
            """
            Return the count of occurences of numerical values in the text
            """

            text = str(text)
            return len([s for s in text.split() if s.isnumeric()])

        def get_filepath_count(text):
            """
            Return the count of occurences of filepaths in the text
            """
            text = str(text)
            return len(re.findall('(\/.*?\.[\w:]+)', text))

        def get_date_count(text):
            """
            Return the count of occurences of dates in the text
            """
            return len(re.findall(
                '(?P<month_0>(?:0[1-9]|1[12])|(?:[1-9]|1[12]))/(?P<day_0>(?:0[1-9]|[12][0-9]|3[01])|(?:[1-9]|[12][0-9]|3[01]))/(?P<year_0>(?:[0-9][0-9])|(?:[0-9]{3}[1-9]|[0-9]{2}[1-9][0-9]|[0-9][1-9][0-9]{2}|[1-9][0-9]{3}))|(?P<month_1>(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec))\\-(?P<day_1>(?:0[1-9]|[12][0-9]|3[01])|(?:[1-9]|[12][0-9]|3[01]))\\-(?P<year_1>(?:[0-9]{3}[1-9]|[0-9]{2}[1-9][0-9]|[0-9][1-9][0-9]{2}|[1-9][0-9]{3}))|(?P<month_2>(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)|(?:January|February|March|April|May|June|July|August|September|October|November|December))(?:\\.\\s+|\\s*)(?P<day_2>(?:0[1-9]|[12][0-9]|3[01])|(?:[1-9]|[12][0-9]|3[01]))(?:,\\s+|\\s*)(?P<year_2>(?:[0-9]{3}[1-9]|[0-9]{2}[1-9][0-9]|[0-9][1-9][0-9]{2}|[1-9][0-9]{3}))|(?P<day_3>(?:0[1-9]|[12][0-9]|3[01])|(?:[1-9]|[12][0-9]|3[01]))(?:[\\.,]\\s+|\\s*)(?P<month_3>(?:January|February|March|April|May|June|July|August|September|October|November|December)|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec))(?:[\\.,]\\s+|\\s*)(?P<year_3>(?:[0-9]{3}[1-9]|[0-9]{2}[1-9][0-9]|[0-9][1-9][0-9]{2}|[1-9][0-9]{3}))|(?P<month_4>(?:January|February|March|April|May|June|July|August|September|October|November|December)|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec))\\s+(?P<year_4>(?:[0-9]{3}[1-9]|[0-9]{2}[1-9][0-9]|[0-9][1-9][0-9]{2}|[1-9][0-9]{3}))|(?P<month_5>(?:0[1-9]|1[12])|(?:[1-9]|1[12]))/(?P<year_5>(?:[0-9]{3}[1-9]|[0-9]{2}[1-9][0-9]|[0-9][1-9][0-9]{2}|[1-9][0-9]{3}))|(?P<year_6>(?:[0-9]{3}[1-9]|[0-9]{2}[1-9][0-9]|[0-9][1-9][0-9]{2}|[1-9][0-9]{3}))|(?P<month_6>(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec))\\s+(?P<day_4>(?=(?:2?1st|2?2nd|2?3rd|[12]?[4-9]th|1[123]th|[123]0th|31st))[0-9][0-9]?)..,\\s*(?P<year_7>(?:[0-9]{3}[1-9]|[0-9]{2}[1-9][0-9]|[0-9][1-9][0-9]{2}|[1-9][0-9]{3}))',
                text))

        def get_time_count(text):
            """
            Return the count of occurences of time/timestamp in the text
            """
            return len(re.findall('[0-9]?[0-9]{1}:[0-9]{2}', text))

        def get_ip_address_count(text):
            """
            Return the count of occurences of IP Address in the text
            """

            text = str(text)
            return len(re.findall(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b", text))

        def get_servername_count(text):
            """
            Return the count of occurences of server name/host name in the text
            """
            text = str(text)
            return len(re.findall('(w{3}?\.?)?[\w?-]+\.(com|in|edu|co|uk|biz|edu|me|org|gov|net|nic)', text))

        def get_pos_tagging(s):


            text = s['clean_text']
            VERB_count = 0
            NOUN_count = 0
            PRON_count = 0
            ADJ_count = 0
            ADVPN_count = 0
            ADP_count = 0
            CONJ_count = 0
            DET_count = 0
            NUM_count = 0
            PRT_count = 0
            PUNC_count = 0
            Action_count = 0
            

            pos_taggs = nltk.pos_tag(nltk.word_tokenize(text))

            for index, pos in enumerate(pos_taggs):
                if 'VB' in pos[1]:
                    VERB_count = VERB_count + 1

                elif 'NN' in pos[1]:
                    NOUN_count = NOUN_count + 1

                    if ((index - 1) > 0 and 'VB' in pos_taggs[index - 1][1]) or (
                            (index - 2) > 0 and 'VB' in pos_taggs[index - 2][1]) or (
                            (index + 1) < len(pos_taggs) and 'VB' in pos_taggs[index + 1][1]) or (
                            (index + 2) < len(pos_taggs) and 'VB' in pos_taggs[index + 2][1]):
                        Action_count = Action_count + 1                    

                elif pos[1] == 'WP':
                    PRON_count = PRON_count + 1
                elif pos[1] in ['JJ', 'JJR', 'JJS']:
                    ADJ_count = ADJ_count + 1
                elif pos[1] in ['ADV', 'RB', 'RBR', 'RBS']:
                    ADVPN_count = ADVPN_count + 1
                elif pos[1] == 'ADP':
                    ADP_count = ADP_count + 1
                elif pos[1] == 'CC ':
                    CONJ_count = CONJ_count + 1
                elif pos[1] == 'WDT':
                    DET_count = DET_count + 1
                elif pos[1] == 'CD':
                    NUM_count = NUM_count + 1
                elif pos[1] == 'RP':
                    PRT_count = PRT_count + 1
                elif pos[1] in ['.', ':', '']:
                    PUNC_count = PUNC_count + 1

            if len(pos_taggs) > 0:
                VERBRatio = VERB_count / len(pos_taggs)
                NOUNRatio = NOUN_count / len(pos_taggs)
                PRONRatio = PRON_count / len(pos_taggs)
                ADJRatio = ADJ_count / len(pos_taggs)
                ADVPNRatio = ADVPN_count / len(pos_taggs)
                ADPRatio = ADP_count / len(pos_taggs)
                CONJRatio = CONJ_count / len(pos_taggs)
                DETRatio = DET_count / len(pos_taggs)
                NUMRatio = NUM_count / len(pos_taggs)
                PRTRatio = PRT_count / len(pos_taggs)
                PUNCRatio = PUNC_count / len(pos_taggs)
            else:
                VERBRatio = 0
                NOUNRatio = 0
                PRONRatio = 0
                ADJRatio = 0
                ADVPNRatio = 0
                ADPRatio = 0
                CONJRatio = 0
                DETRatio = 0
                NUMRatio = 0
                PRTRatio = 0
                PUNCRatio = 0
            return VERBRatio, NOUNRatio, PRONRatio, ADJRatio, ADVPNRatio, ADPRatio, CONJRatio, DETRatio, NUMRatio, PRTRatio, PUNCRatio, Action_count

        # convert the column to string type
        data['clean_text'] = data['clean_text'].astype(str)

        # getting count of text lengthand word count 
        data['text_len'] = data['clean_text'].astype(str).apply(len)
        data['text_word_count'] = data['clean_text'].apply(lambda x: len(str(x).split()))
        
        #Unique word text count
        df['text_unique_word_count']=df["clean_text"].apply(lambda x: len(set(str(x).split())))


        # getting count of text lengthand word count 
        data['title_len'] = data['title'].astype(str).apply(len)
        data['title_word_count'] = data['title'].apply(lambda x: len(str(x).split()))

        #Unique word title count
        df['title_unique_word_count']=df["title"].apply(lambda x: len(set(str(x).split())))

 
        data[['VERBRatio', 'NOUNRatio', 'PRONRatio', 'ADJRatio',
              'ADVPNRatio', 'ADPRatio', 'CONJRatio', 'DETRatio', 'NUMRatio', 'PRTRatio', 'PUNCRatio',
              'ActionCount']] = data.apply(get_pos_tagging, axis=1, result_type="expand")

        logging.info("Creating Features: Acroynym")
        data['acronym_to_activity_ratio'] = data['clean_text'].apply(
            get_acronym_count)

        data['acronym count'] = data['clean_text'].apply(get_acronym_count)

        logging.info("Creating Features: Numeric Value Count")
        data['num value count'] = data['clean_text'].apply(
            get_numerical_value_count)

        
        logging.info("Creating Features: Last Activity Count")
        data['is_len_range_1_500'] = data.apply(
            lambda x: 1 if 0 < int(x['text_len']) <= 500 else 0, axis=1)

        logging.info("Creating Features: Last Activity Count")
        data['is_len_range_1_500'] = data.apply(
            lambda x: 1 if 0 < int(x['title_len']) <= 500 else 0, axis=1)

        logging.info("Features are created")
        return data

    def scaler(self, df, prefix, load="No"):
        logging.info("Normalizing the features")

        labels = ['u_sla_status', 'u_service', 'u_ess_service']
        ignore = ['short_description', 'without junk', 'last_activity', 'number']

        fields = []
        for col in df.columns:
            if col not in ignore:
                if col in labels:
                    fields.append((col, LabelEncoder()))
                else:
                    fields.append(([col], StandardScaler()))

        if load == "No":
            mapper = DataFrameMapper(fields, df_out=True)
            mapper.fit(df.copy())
            pickle.dump(mapper, open(prefix + "_" + "mapper.pkl", "wb"))
        else:
            mapper = pickle.load(open(prefix + "_" + "mapper.pkl", "rb"))

        return mapper.transform(df.copy())

    

    def xgbclassifier(self, train_data, train_label, test_data, test_label):
        logging.info("Building XGB Classifier")
        model = xgb.XGBClassifier(objective="binary:logistic",
                                  random_state=42)
        #                               learning_rate=0.3,
        #                               max_leaf_nodes= 8,
        #                               n_estimators=100,
        #                               max_features=0.5,
        #                               max_depth=7)

        model.fit(train_data, train_label, eval_set=[(test_data, test_label)],
                  eval_metric='auc',
                  early_stopping_rounds=100)

        # y_pred = model.predict(test_data)
        # y_proba = model.predict_proba(test_data)

        y_pred = model.predict(test_data)
        # predictions = [round(value) for value in y_pred]
        # evaluate predictions
        accuracy = accuracy_score(test_label, y_pred)
        print("Accuracy: %.2f%%" % (accuracy * 100.0))

        # plt.subplots(figsize=(15, 10))
        # ax = sn.distplot(y_proba[:, 1])
        # # Set title
        # ax.set_title("XGBoost | AUC score:" + str(roc_auc_score(test_label, y_proba[:, 1])))
        # # Show the plot
        # plt.show()

        # Feature Importance
        # fig, ax = plt.subplots(figsize=(8, 12))
        # xgb.plot_importance(model, max_num_features=20, height=0.8, ax=ax)
        # plt.show()

        model.save_model('xgb_classifier_auc_76_model_v2(mapper).bin')
        logging.info('Model  xgb_classifier_auc_76_model_v2(mapper).bin Stoerd')
        return model

    def xgbRegressor(self, X_train, y_train, X_test, y_test):
        logging.info("Building XGB Regressor")
        xgb_regressor_model = XGBRegressor(objective='reg:linear', n_estimators=500)
        xgb_regressor_model.fit(X_train, y_train,
                                eval_set=[(X_test, y_test)], early_stopping_rounds=50)

        y_test_pred = xgb_regressor_model.predict(X_test)

        mse = mean_squared_error(y_test_pred, y_test)
        print(mse)
        logging.info("Mean Square Error", mse)
        xgb_regressor_model.save_model('xgb_regressor_rmse_7_point_8_percent_model_v2(mapper).bin')
        logging.info("Model xgb_regressor_rmse_7_point_8_percent_model_v2(mapper).bin Stored")

    

    def pre_processing(self, dataframe):
        dataframe.drop(columns=[], inplace=True)
        dataframe.fillna('', inplace=True)
        return dataframe
    
    def caculateFeatureImp(self):
        from numpy import loadtxt
        from numpy import sort
        from xgboost import XGBClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score
        from sklearn.feature_selection import SelectFromModel
        
        # load data
        dataset = loadtxt(config.NEW_TRAINING_FILE, delimiter=",", encoding='utf-8')
        # split data into X and y
        X = dataset[:,1:25]
        Y = dataset[:,26]
        # split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=7)
        # fit model on all training data
        model = MyXGBClassifier()
        model.fit(X_train, y_train)
        # make predictions for test data and evaluate
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        print("Accuracy: %.2f%%" % (accuracy * 100.0))
        # Fit model using each importance as a threshold
        thresholds = sort(model.feature_importances_)
        for thresh in thresholds:
            # select features using threshold
            selection = SelectFromModel(model, threshold=thresh, prefit=True)
            select_X_train = selection.transform(X_train)
            # train model
            selection_model = XGBClassifier()
            selection_model.fit(select_X_train, y_train)
            # eval model
            select_X_test = selection.transform(X_test)
            predictions = selection_model.predict(select_X_test)
            accuracy = accuracy_score(y_test, predictions)
            print("Thresh=%.3f, n=%d, Accuracy: %.2f%%" % (thresh, select_X_train.shape[1], accuracy*100.0))
    

    def execute_classifier(self):

        logging.info("EDA")
        
        df_news = pd.read_csv(config.TRAINING_FILE).fillna("none").reset_index(drop=True)
        df_news.dropna(inplace=True)
        # df_train, df_test = model_selection.train_test_split(dfx, test_size=0.1, random_state=42, stratify=dfx.label.values)
        df_news['clean_text'] =self.processText(df_news)
        df_news.dropna(inplace=True)
        # print(df_news.head())
        df_news = self.generate_features(df_news.copy())
        print(df_news.columns)
        df_news = df_news[['id', 'title', 'author', 'text', 'clean_text', 'text_len','text_word_count', 'title_len', 'title_word_count', 'VERBRatio',       'NOUNRatio', 'PRONRatio', 'ADJRatio', 'ADVPNRatio', 'ADPRatio',       'CONJRatio', 'DETRatio', 'NUMRatio', 'PRTRatio', 'PUNCRatio',       'ActionCount', 'acronym_to_activity_ratio', 'acronym count',       'num value count', 'is_len_range_1_500','label']]
        df_news.to_csv(config.NEW_TRAINING_FILE)
        self.caculateFeatureImp()

        # resolutions.drop(columns=['priority',
        #                           'u_incident_type',
        #                           'u_1st_time_fix',
        #                           'u_closure_fix',
        #                           'state', 'u_source',
        #                           'subcategory', 'work_notes',
        #                           'impact', 'u_closure_subcategory',
        #                           'cmdb_ci', 'severity', 'description'], inplace=True)

        
        # resolutions['label'].value_counts()

        # resolutions_one = resolutions[resolutions['label'] == 1].sample(903)

        # resolutions_zero = resolutions[resolutions['label'] == 0].sample(1000)

        # resolutions = pd.concat([resolutions_one, resolutions_zero], axis=0, sort=True)
        # resolutions['label'].value_counts()

        # y = resolutions['label']
        # X = resolutions.drop(columns=['label'])
        # X = X[
        #     ['ADJRatio', 'ADPRatio', 'ADVPNRatio', 'ActionCount', 'CONJRatio', 'DETRatio', 'NOUNRatio', 'NUMRatio',
        #      'PRONRatio',
        #      'PRTRatio', 'PUNCRatio', 'ProblemCount', 'VERBRatio', 'acronym count', 'acronym_to_activity_ratio',
        #      'activity_count', 'ip address count', 'is_len_range_1_500', 'kb count', 'last_activity_len',
        #      'num value count',
        #      'server name count', 'u_ess_service', 'u_service', 'u_sla_status']]

        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # # ## Label encoding and Scaling the data

        # prefix = "cls"
        # X_train_scaled = self.scaler(X_train, prefix)
        # X_test_scaled = self.scaler(X_test, prefix, "Yes")

        # models = []

        # # add the list of algorithms that will be evaluated for spot checking
        # models.append(("LR", LogisticRegression(C=0.1)))
        # models.append(("SVC", SVC()))
        # models.append(("DT", DecisionTreeClassifier()))
        # models.append(("RF", RandomForestClassifier()))
        # models.append(("GB", GradientBoostingClassifier()))
        # models.append(("LGBC", LGBMClassifier()))

        # results = []
        # names = []

        # # set the measure that will be used for evaluation
        # scoring = 'accuracy'

        # # KFold cross validations to evaluate the algorithms
        # for name, model in models:
        #     kfold = KFold(n_splits=7, random_state=7)
        #     cv_results = cross_val_score(model, X_train_scaled, y_train, cv=kfold, scoring=scoring)
        #     names.append(name)
        #     results.append(cv_results.mean())
        #     msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        #     logging.info(msg)
        #     print(msg)
        # results.sort(reverse=True)
        # print(results)
        # logging.info(results)

        # # ## Model - XGBoost
        # self.xgb_model = self.xgbclassifier(X_train_scaled, y_train, X_test_scaled, y_test)

       

if __name__ == '__main__':
    rc = FeatureCreation()
    rc.execute_classifier()
    # rc.execute_regessors()
