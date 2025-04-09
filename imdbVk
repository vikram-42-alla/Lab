#Import the necessary Libraries
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
(X_train,y_train),(X_test,y_test)=imdb.load_data(num_words=10000)
maxlen=200
X_train=pad_sequences(X_train,maxlen=maxlen)
X_test=pad_sequences(X_test,maxlen=maxlen)
#Model
model=Sequential()
model.add(Dense(128,activation='relu',input_shape=(maxlen,)))
model.add(Dropout(0.5))
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1,activation='sigmoid'))
#Compile
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=11,batch_size=128)
scores=model.evaluate(X_test,y_test,verbose=0)
print("Accuracy : %.2f%%"%(scores[1]*100))
