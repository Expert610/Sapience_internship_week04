import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from keras.models import Sequential
from keras.layers import Embedding, GlobalAveragePooling1D, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns




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




# Parameters
vocab_size = 5000
max_length = 100
trunc_type = 'post'
padding_type = 'post'
oov_tok = "<OOV>"

# Tokenize the processed message
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(df['processed_message'])

sequences = tokenizer.texts_to_sequences(df['processed_message'])
padded = pad_sequences(sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

# Encode labels (ham=0, spam=1)
le = LabelEncoder()
labels = le.fit_transform(df['label'])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(padded, labels, test_size=0.2, random_state=42)


model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=32, input_length=max_length),
    GlobalAveragePooling1D(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), batch_size=32)
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy:.4f}")


# Get predictions (probabilities → binary labels)
y_pred_probs = model.predict(X_test)
y_pred = (y_pred_probs > 0.5).astype("int32").flatten()

# Print classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["ham", "spam"]))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot confusion matrix
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.show()
