from demo_test_new_data_classify import get_lying_data_card,get_lying_data_card_features
import numpy as np
from sklearn.model_selection import StratifiedKFold
from keras.layers import Dense, Dropout,LSTM,TimeDistributed,Activation
from keras.models import Sequential
from ppg.utils import save_neural_network_model
from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV,cross_val_score,train_test_split
from sklearn.svm import SVC
import pickle
import utils_ppg
import sys



def cal_len():
	x, y = get_lying_data_card()
	print(len(x))
	len_dict = {}
	for wave in x:
		if len(wave) in len_dict.keys():
			len_dict[len(wave)] += 1
		else:
			len_dict[len(wave)] = 1
	res_count = sorted(len_dict.items(),key=lambda x:x[1],reverse=True)
	res_len = sorted(len_dict.items(),key=lambda x:x[0],reverse=True)
	print(res_count)
	print(res_len)

def lstm_waves():
	x, y = get_lying_data_card()
	x = utils_ppg.expand_single_wave_to_200(x)
	x = x.astype('float32')
	x = x.reshape((-1,200,1))
	x = x/255
	print('end input shape is ',x.shape,y.shape)
	kfold = StratifiedKFold(n_splits=5,shuffle=True)
	acc = []
	los = []
	models_path = './models/lstm'
	index = 0
	early_stopping = EarlyStopping(monitor='loss',patience=30,min_delta=0.0001,verbose=2)
	for train,test in kfold.split(x,y):
		model = Sequential()
		model.add(LSTM(8, return_sequences=True, input_shape=(200,1)))
		model.add(Dropout(0.3))
		model.add(LSTM(16, return_sequences=True))
		model.add(Dropout(0.3))
		model.add(LSTM(8))
		model.add(Dropout(0.3))
		model.add(Dense(1, activation='sigmoid'))
		model.compile(loss='binary_crossentropy',
					  optimizer=RMSprop(),
					  metrics=['accuracy'])
		model.fit(x[train],y[train],batch_size=16,epochs=100,verbose=1,validation_data=(x[test],y[test]),callbacks=[early_stopping])
		scores = model.evaluate(x[test],y[test],batch_size=16,verbose=1)
		los.append(scores[0])
		acc.append(scores[1])
		index+=1
		model_name = models_path+'/'+str(index)
		save_neural_network_model(model=model, pathname=model_name)
	print('the k-fold loss are', los)
	print('the k-fold acc are', acc)
	print('the average los is:', np.mean(los))
	print('the average acc is :', np.mean(acc))

def lstm_features(num_waves, num_gap):
	# num_waves = 15
	# num_gap = 1
	path = './data/47_features/featuresUnbalance.csv'
	x,y = get_lying_data_card_features(path)
	x = utils_ppg.standard(x)
	# x, y = utils_ppg.feature_selection(x, y)
	y = utils_ppg.to_categorical(y,2)
	feature_len = x.shape[1]
	x_data = []
	y_data = []
	for i in range(0,len(x)-num_waves+1,num_gap):
		x_data.append(x[i:i+num_waves])
		y_data.append(y[i:i+num_waves])
	x_data = np.array(x_data)
	y_data = np.array(y_data)
	print('the input shape of x and y is ',x_data.shape,y_data.shape)
	x_train, x_test, y_train, y_test = train_test_split(x_data,y_data, test_size = 0.3, random_state = 0)
	model = Sequential()
	model.add(LSTM(32, return_sequences=True, input_shape=(num_waves, feature_len)))
	model.add(Dropout(0.5))
	# model.add(LSTM(16,return_sequences=True))
	# model.add(Dropout(0.5))
	# model.add(LSTM(16,return_sequences=True))
	# model.add(Dropout(0.5))
	model.add(TimeDistributed(Dense(2,activation='softmax')))
	model.compile(loss='categorical_crossentropy',
				  optimizer='adam',
				  metrics=['accuracy'])

	model.summary()
	model.fit(x_train, y_train, batch_size=2, epochs=50, verbose=1,validation_data=(x_test,y_test))
	scores = model.evaluate(x_test, y_test, batch_size=2, verbose=1)
	print('the result of lstm_feature is ', scores)
	model_name = './models/lstm_features_win10'
	save_neural_network_model(model=model, pathname=model_name)
	return scores
	# with open('./log.txt','w') as lg:
	# 	lg.write(str(hist.history))
	# plt.plot(hist.history['loss'])
	# plt.show()


def multiply_networks():
	x, y = get_lying_data_card()
	# y = keras.utils.to_categorical(y, num_classes)
	x = utils_ppg.expand_single_wave_to_200(x)
	x = x.astype('float32')
	# scaler = StandardScaler().fit(x)
	# x = scaler.transform(x)
	# min_max_scaler = MinMaxScaler()
	# x = min_max_scaler.fit_transform(x)
	# x = x/255
	# for wave in x:
	# 	plt.plot(wave)
	# 	plt.show()
	# 	plt.pause(0.2)
		# plt.clf()
		# plt.ioff()
	test = []
	for i in x:
		test.extend(i)
	plt.plot(test)
	plt.show()
	exit()
	kfold = StratifiedKFold(n_splits=5,shuffle=True)
	acc = []
	los = []
	models_path = './models/mlp'
	index = 0
	for train,test in kfold.split(x,y):
		model = Sequential()
		model.add(Dense(32, activation='relu', input_dim=200))
		model.add(Dropout(0.2))
		# model.add(Dense(256, activation='relu'))
		# model.add(Dropout(0.5))
		# model.add(Dense(64, activation='relu'))
		# model.add(Dropout(0.2))
		# model.add(Dense(64, activation='relu'))
		# model.add(Dropout(0.2))
		model.add(Dense(1, activation='sigmoid'))
		model.compile(loss='binary_crossentropy',
					  optimizer=RMSprop(),
					  metrics=['accuracy'])
		model.summary()
		model.fit(x[train],y[train],batch_size=32,epochs=50,verbose=1,validation_data=(x[test],y[test]))
		scores = model.evaluate(x[test],y[test],batch_size=32,verbose=1)
		los.append(scores[0])
		acc.append(scores[1])
		index+=1
		model_name = models_path+'/'+str(index)
		save_neural_network_model(model=model, pathname=model_name)
	print('the k-fold loss are', los)
	print('the k-fold acc are', acc)
	print('the average los is:', np.mean(los))
	print('the average acc is :', np.mean(acc))

def svm_method():
	x,y = get_lying_data_card_features()
	x = utils_ppg.standard(x)
	x,y = utils_ppg.feature_selection(x,y)
	# parameters = {
	# 	'C': [1e2,0.1,1,10],
	# 	'gamma': [0.2,0.1,0.05]
	# }
	# clf = GridSearchCV(SVC(kernel='rbf'),param_grid=parameters,cv=5)
	# clf.fit(x,y)
	clf = SVC(kernel='linear')
	scores = cross_val_score(clf, x, y, cv=5)
	print('the end scores is:',scores)
	# print("The best parameters are %s with a score of %0.2f"
	# 	  % (clf.best_params_, clf.best_score_))
	save_path = './models/svm/svm_param.pickle'
	file = open(save_path,'wb')
	pickle.dump(clf,file)
	file.close()



if __name__ == '__main__':
	# classify_lstm()
	# classify_mlp()
	# svm_method()


	# with open('./result.txt', 'w+') as file:
	# 	# sys.stdout = file
	# 	print('start')
	# 	scores = []
	# 	for i in range(60, 300, 10):
	# 		# print(i)
	# 		scores.append(lstm_features(i, 1))
	# 	for j in range(1, len(scores)):
	# 		print('num_waves is %d: result is %s', j, scores[j])
    lstm_waves()
	#score = lstm_features(10, 1)
	#print(score)
