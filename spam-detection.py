import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.preprocessing import LabelEncoder



df= pd.read_csv('spam_sms.csv' ,sep=',',encoding='latin-1' )
df.rename(columns={'v1':'label','v2':'message'}, inplace=True)
print(df.head())
# print(df.info())
# print(df.describe())


nltk.download('stopwords')
nltk.download('punkt_tab')

# Set up stopwords and stemmer
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

# Cleaning function
def clean_text(text):
    text = text.lower()                                # Lowercase
    text = re.sub(r'[^a-z\s]', '', text)               # Remove numbers and special chars
    text = ' '.join(word for word in text.split() if word not in stop_words)  # Remove stopwords
    return text

# Tokenize and stem
def tokenize_and_stem(text):
    tokens = word_tokenize(text)
    return ' '.join(stemmer.stem(token) for token in tokens)

# Apply to dataset
df['cleaned_message'] = df['message'].apply(clean_text)
df['processed_message'] = df['cleaned_message'].apply(tokenize_and_stem)

# View result
# print(df[['label', 'message', 'processed_message']].head())


vectorizer = TfidfVectorizer()
x = vectorizer.fit_transform(df['processed_message'])
y = df['label']

# print(f'Shape of Tf IDF matrix:  {x.shape}')

lbl_enc = LabelEncoder()
lbl = lbl_enc.fit_transform(y)

x_train,x_test,y_train,y_test = train_test_split(x,lbl, random_state=42,test_size=0.2)


lr = LogisticRegression(max_iter=1000)
lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)

print(f"\n Logistic Regression Results:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=["ham", "spam"]))




