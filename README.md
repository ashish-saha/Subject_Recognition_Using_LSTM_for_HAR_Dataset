# Subject_Recognition_Using_LSTM_for_HAR_Dataset


### Creation of new test set by combining all data

```
# training data
X_train = pd.read_csv('UCI HAR Dataset/train/X_train.txt', delim_whitespace=True, header=None)
X_train.columns = features
X_train['Activity'] = pd.read_csv('UCI HAR Dataset/train/y_train.txt', names=['Activity'])
y_train_subject = pd.read_csv('UCI HAR Dataset/train/subject_train.txt', names=['subject'])
X_train['subject'] = y_train_subject

# test data
X_test = pd.read_csv('UCI HAR Dataset/test/X_test.txt', delim_whitespace=True, header=None)
X_test.columns = features
X_test['Activity'] = pd.read_csv('UCI HAR Dataset/test/y_test.txt', names=['Activity'])
y_test_subject = pd.read_csv('UCI HAR Dataset/test/subject_test.txt', names=['Activity'])
X_test['subject'] = y_test_subject

all_X_data = np.concatenate((X_train, X_test))
all_y_data = np.concatenate((y_train_subject, y_test_subject))

X_train, X_test, y_train, y_test = train_test_split(all_X_data, all_y_data, test_size=0.2, random_state=0)


```




### LSTM model used for training 

```
from keras.models import Sequential
from keras.layers import LSTM, Dense, Conv2D, MaxPooling2D, Flatten, Dropout


model = Sequential()
# RNN layer
model.add(LSTM(units = 128, input_shape = (X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(units = 64, activation='relu'))
model.add(Dense(y_train.shape[1], activation = 'softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

```


### Accuracy for predecting 30 subjects accross dataset

| Models    | Training | Validation | Test |
| -------- | ------- | -------- | ------- |
| LogisticRegression  | 85.8%    | | 68.1% |
| SGD | 65.8%     | | 55.9% |
| Deep Learning (CNN)    | 73.7%   | 69.5% | 69.53% |
| Deep Learning (LSTM)    | 64.3%    | 63.3% | 66.58% |
