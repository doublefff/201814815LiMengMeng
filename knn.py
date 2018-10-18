import os
import math

file = "data/out/tf_idf"
per = 0.1

def distance_compute(data_train, data_test):
	distance = 0
	for i in range(len(data_train)):
		distance  += (data_train[i] - data_test[i]) * (data_train[i] - data_test[i])
	distance = math.sqrt(distance)
	return distance


def similarity_compute(path, list_test, per):
	dic = dict()
	files = os.listdir(path)
	index = 0
	for file in files:
		file_list = os.listdir(path + "/" + file)
		data_len = int(per*len(file_list))
		for i in range(len(file_list) - data_len):
			file_now = path + "/" + file + "/" + file_list[i + data_len]
			with open(file_now, "r", encoding="utf-8") as data_train:
				list_train = []
				data_train = data_train.readlines()
				for i in data_train:
					if(len(i.split(":")) < 2):continue
					list_train.append(float(i.split(":")[1]))
				distance = distance_compute(list_train, list_test)
				index += 1
				if index %1000==0:
					print(index)
				# print(file_now)
				key_file = file_now
				dic[key_file] = distance
	return dic


def KNN(path, per, k):
	files = os.listdir(path)
	for file in files:
		file_list = os.listdir(path + "/" + file)
		data_len = int(per*len(file_list))
		for i in range(data_len):
			list_test = []
			file_final = path + "/" + file + "/" + file_list[i]
			with open(file_final, "r", encoding="utf-8") as data_test:
				data_test = data_test.readlines()
				for i in data_test:
					if (len(i.split(":")) < 2): continue
					list_test.append(float(i.split(":")[1]))
			dic =  similarity_compute(path, list_test, per)
			dic = sorted(dic.items(), key=lambda x:x[1])
			print(file_final)
			for i in range(k):
				print(dic[i])

KNN(file, 0.1, 4)


