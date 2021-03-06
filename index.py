'''
Please Read README.md before running this program
Credit by Imron Rosadi | https://imronrosadi.com or https://github.com/irosadie
'''

from flask import request
from flask import jsonify
import logging
import time
import random
import string
import json
import numpy as np
from core.data import Data
from core.bpnn import Bpnn
from core.ga import GeneticAlgorithm
from config.app import app

logging.basicConfig(level=logging.DEBUG)


@app.route('/')
def index():
    return "Hello world!"


@app.route('/ori', methods=['GET'])
def data():
    msg = "OK"
    status = 1
    result = None
    try:
        dt = Data()
        data = dt.dataReading(year=True)
        result = [
            {'year': item[0], 'data_in_month': item[1:len(item)]} for item in data]
        status = 1
    except Exception as e:
        msg = str(e)
        status = 0
    finally:
        return jsonify({'status': status, 'msg': msg, 'data': result})


@app.route('/time-series', methods=['GET'])
def timeSeries():
    msg = "OK"
    status = 1
    result = None
    try:
        dt = Data()
        data_ori = dt.dataReading()
        data = dt.dataTimeSeries(data_ori)
        result = [
            {'data_in_month': item[:len(item)-1], 'target': item[-1]} for item in data]
        status = 1
    except Exception as e:
        msg = str(e)
        status = 0
    finally:
        return jsonify({'status': status, 'msg': msg, 'data': result})


@app.route('/normalization', methods=['GET'])
def normalization():
    msg = "OK"
    status = 1
    result = None
    try:
        dt = Data()
        data_ori = dt.dataReading()
        data_ts = dt.dataTimeSeries(data_ori)
        result = dt.dataNormalization(data_ts)
        status = 1
    except Exception as e:
        msg = str(e)
        status = 0
    finally:
        return jsonify({'status': status, 'msg': msg, 'data': result})


@app.route('/split', methods=['GET'])
def split():
    msg = "OK"
    status = 1
    result = None
    try:
        code = request.args.get('code', default=0.2, type=float)
        shuffle = request.args.get('shuffle', default=0, type=int)
        dt = Data()
        data_ori = dt.dataReading()
        data_ts = dt.dataTimeSeries(data_ori)
        data_norm = dt.dataNormalization(data_ts)
        split_result = dt.dataSplit(data_norm['data_norm'], code, shuffle)
        is_sufix = f'[{int(100-code*100)}-{int(code*100)}]'
        # save train to csv
        train_path = f'data/train/data{is_sufix}.csv'
        train_target_path = f'data/train/target{is_sufix}.csv'
        dt.dataWriting(train_path, split_result['train'], True)
        dt.dataWriting(train_target_path, split_result['train_target'], False)
        # save test to csv
        test_path = f'data/test/data{is_sufix}.csv'
        test_target_path = f'data/test/target{is_sufix}.csv'
        dt.dataWriting(test_path, split_result['test'], True)
        dt.dataWriting(test_target_path, split_result['test_target'], False)
        is_data_percentage = f'{int(100-code*100)}:{int(code*100)}'
        is_data_usage = f'{len(split_result["train"])}:{len(split_result["test"])}'
        result = {'info': {'data_length': len(
            data_norm['data_norm']), 'data_percentage': is_data_percentage, 'data_usage': is_data_usage, 'shuffle': shuffle, 'code': code}, 'data_split': split_result}
        status = 1
    except Exception as e:
        msg = str(e)
        status = 0
    finally:
        return jsonify({'status': status, 'msg': msg, 'data': result})


@ app.route('/chromosome', methods=['GET', 'POST'])
def generateChromosome():
    msg = "OK"
    status = 1
    result = None
    try:
        dt = Data()
        if(request.method == 'POST'):
            random = request.form.get('random', default=0, type=int)
            code = request.form.get('code', default=0.2, type=float)
            is_sufix = f'[{int(100-code*100)}-{int(code*100)}]'
            if(random):
                bpnn = Bpnn()
                X_train_path = f'data/train/data{is_sufix}.csv'
                X_train = dt.dataCsvReading(X_train_path)
                chromosome_result = bpnn.generateChromosome(X_train)
            else:
                if(code != 0.2):
                    raise Exception(
                        f"Sorry, code:{code} have no default chromosome")
                static_chromosome_path = 'data/chromosome/default/default[80-20].json'
                chromosome_result = dt.dataJsonReading(static_chromosome_path)
            is_data_percentage = f'{int(100-code*100)}:{int(code*100)}'
            is_info = {'chromosome_length': len(
                chromosome_result), 'data_percentage': is_data_percentage, 'code': code, 'random': random}
            usage_chromosome_path = f'data/chromosome/usage/usage{is_sufix}.json'
            dt.dataJsonWriting(usage_chromosome_path, chromosome_result)
        else:
            code = request.args.get('code', default=0.2, type=float)
            is_sufix = f'[{int(100-code*100)}-{int(code*100)}]'
            usage_chromosome_path = f'data/chromosome/usage/usage{is_sufix}.json'
            chromosome_result = dt.dataJsonReading(usage_chromosome_path)
            is_data_percentage = f'{int(100-code*100)}:{int(code*100)}'
            is_info = {'chromosome_length': len(
                chromosome_result), 'data_percentage': is_data_percentage, 'code': code}
        result = {'info': is_info, 'chromosome': chromosome_result}
        status = 1
    except Exception as e:
        msg = str(e)
        status = 0
    finally:
        return jsonify({'status': status, 'msg': msg, 'data': result})


@ app.route('/train', methods=['POST', 'GET'])
def trainData():
    msg = "OK"
    status = 1
    result = None
    try:
        dt = Data()
        if(request.method == 'POST'):
            code = request.form.get('code', default=0.2, type=float)
            alfa = request.form.get('alfa', default=0.5, type=float)
            alfa_range = json.loads(request.form.get(
                'alfa_range', default="[-0.25, 1.25]", type=str))
            prob_co = request.form.get('prob_co', default=0.9, type=float)
            prob_m = request.form.get('prob_m', default=0.1, type=float)
            r = request.form.get('r', default=0.075, type=float)
            name = request.form.get(
                'name', default=f'training-'+(''.join(random.choices(string.ascii_uppercase + string.digits, k=5))), type=str)
            is_sufix = f'[{int(100-code*100)}-{int(code*100)}]'
            bpnn = Bpnn()
            start_time = time.time()
            chromosome_path = f'data/chromosome/usage/usage{is_sufix}.json'
            chromosome = dt.dataJsonReading(chromosome_path)
            X_train_path = f'data/train/data{is_sufix}.csv'
            X_train = dt.dataCsvReading(X_train_path)
            y_train_path = f'data/train/target{is_sufix}.csv'
            y_train = dt.dataCsvReading(y_train_path)
            bpnn1_result = bpnn.bpnn(chromosome, X_train, y_train[0], alfa)
            ga = GeneticAlgorithm(
                fitness=bpnn1_result['fitness'], chromosome=chromosome, alfa_range=list(alfa_range), prob_co=prob_co, prob_m=prob_m, r=r)
            aglen_result = ga.geneticAlgorithm()
            # put chromosome mutation into bpnn2
            bpnn2_result = bpnn.bpnn(
                aglen_result['mutation'], X_train, y_train[0], alfa)
            index = bpnn2_result['fitness'].index(max(bpnn2_result['fitness']))
            best_chromosome = bpnn2_result['update_chromosom'][index]
            execution_time = time.time() - start_time
            model_name = name.replace(' ', '-')
            is_data_percentage = f'{int(100-code*100)}:{int(code*100)}'
            result = {'info': {'train_name': name, 'model_name': f'{model_name}.pickle', 'data_percentage': is_data_percentage, 'code': code, 'params': {'alfa': alfa, 'alfa_range': alfa_range, 'prob_co': prob_co, 'prob_m': prob_m, 'r': r}, 'execution_time': {'time': execution_time, 'unit': 'seconds'}}, 'chromosomes': bpnn2_result, 'best': {
                'index': index, 'fitness': max(bpnn2_result['fitness']), 'chromosome': best_chromosome}}
            model_path = f'data/model/{model_name}.pickle'
            dt.dataPickleWriting(model_path, result)
        else:
            model_path = f'data/model/{request.args["model"]}'
            result = dt.dataPickleReading(model_path)
        status = 1
    except Exception as e:
        msg = str(e)
        status = 0
    finally:
        return jsonify({'status': status, 'msg': msg, 'data': result})


@ app.route('/test', methods=['POST'])
def testData():
    msg = "OK"
    status = 1
    result = None
    test_result = None
    try:
        dt = Data()
        model_name = request.form["model"]
        data = request.form.get('data', default="", type=str)
        y_test = request.form.get('y_test', default=None, type=float)

        bpnn = Bpnn()
        dt = Data()
        data_ori = dt.dataReading()
        data_ts = dt.dataTimeSeries(data_ori)
        data_norm = dt.dataNormalization(data_ts)
        model_path = f'data/model/{model_name}'
        model_result = dt.dataPickleReading(model_path)
        code = model_result["info"]["code"]
        is_sufix = f'[{int(100-code*100)}-{int(code*100)}]'

        if(data):
            data = json.loads(data)

            if(not 'test' in data or not 'target' in data):
                raise Exception(
                    "please using format {'test':[[0.0,...11.0],[0.0, ...11.0]], 'target':[1.0, 0]} - using 0 if data test have no target")

            for k, i in enumerate(data['test']):
                if(len(i) != 12):
                    raise Exception(
                        f"index {k} expected have 12 length data, {len(i)} given")

            if(len(data['target']) != len(data['test'])):
                raise Exception(
                    "data target must have the same length as test data")

            X_test = data['test']
            y_test = data['target']
        else:
            X_test_path = f'data/test/data{is_sufix}.csv'
            X_test = dt.dataCsvReading(X_test_path)

            y_test_path = f'data/test/target{is_sufix}.csv'
            y_test = dt.dataCsvReading(y_test_path)[0]

        start_time = time.time()
        test_result = bpnn.testing(
            best_chromosome=model_result['best']['chromosome'], X_test=X_test, y_test=y_test)

        test_result_info = []
        is_min = data_norm['min']
        is_max = data_norm['max']

        test_result = [test_result_info.append(
            {**i, "predict_denorm": dt.singleDenormalization(i['predict'], [is_min, is_max]), "target_denorm": None if(i['target']) == 0 else dt.singleDenormalization(i['target'], [is_min, is_max]), "mse": None if (i['target'] == 0) else i['mse']}) for i in test_result]

        execution_time = time.time() - start_time
        result = {'info': {'time': round(time.time()*1000, 0), 'model_name': f'{model_name}', 'data_percentage': model_result['info']['data_percentage'], 'code': code, 'params': model_result['info']['params'], 'execution_time': {
            'time': execution_time, 'unit': 'seconds'}}, 'predict': test_result_info}
        status = 1
    except Exception as e:
        msg = str(e)
        status = 0
    finally:
        return jsonify({'status': status, 'msg': msg, 'data': result})


if __name__ == "__main__":
    app.run(debug=True)
