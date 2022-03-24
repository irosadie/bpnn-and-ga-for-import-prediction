
import numpy as np
from sklearn.model_selection import train_test_split
from pandas import read_excel
import csv
import json
import pickle


class Data:
    # done
    def dataReading(self, year=False):
        file_name = 'data/origin/import.xlsx'
        df = read_excel(file_name, sheet_name=0)
        data = []
        for key, item in enumerate(df.values.tolist()):
            if(key >= 3):
                data.append(item) if year else data.append(item[1:len(item)])
        return data

    def dataWriting(self, path, data, multiple=False):
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(
                data) if multiple == True else writer.writerow(data)

    def dataJsonWriting(self, path, data):
        with open(path, "w") as f:
            json.dump(data, f)

    def dataPickleWriting(self, path, data):
        f = open(path, 'wb')
        pickle.dump(data, f)
        f.close()

    def dataCsvReading(self, path):
        file = open(path)
        csvreader = csv.reader(file)
        rows = []
        for row in csvreader:
            rows.append(row)
        file.close()
        return rows

    def dataJsonReading(self, path):
        f = open(path)
        data = json.load(f)
        f.close()
        return data

    def dataPickleReading(self, path):
        f = open(path, 'rb')
        data = pickle.load(f)
        f.close()
        return data

    def dataTimeSeries(self, data):
        ts_data = []
        ts = []
        for i, k in enumerate(data):
            for n in k:
                ts_data.append(n)

        length = len(data[0])
        for y in range(0, (len(ts_data)-length)):
            ts.append(ts_data[0+y:(length+1)+y])

        return ts

    def dataNormalization(self, data):
        ts = np.array(data)
        is_min = ts.min()
        is_max = ts.max()
        norm_data = [(0.8*(x-is_min))/(is_max-is_min)+0.1 for x in ts]
        norm_data = np.array(norm_data)
        return {'min': float(is_min), 'max': float(is_max), 'data_ori': data, 'data_norm': norm_data.tolist()}

    def singleDenormalization(self, value: float, min_max: list):
        return (((value-0.1)*(min_max[1]-min_max[0]))/0.8)+min_max[0]

    def dataSplit(self, data, code, shuffle=0):
        data = np.array(data)
        params_ = data[:, 0:(data.shape[1]-1)]
        class_ = data[:, (data.shape[1]-1)]
        X_train, X_test, y_train, y_test = train_test_split(
            params_, class_, test_size=code, shuffle=(True if shuffle == 1 else False))
        return {'train': X_train.tolist(), 'test': X_test.tolist(), 'train_target': y_train.tolist(), 'test_target': y_test.tolist()}
