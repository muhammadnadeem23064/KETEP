import os
import pathlib
import random
import sqlite3 as sql
import json
import json
import json2html
import numpy as np
import pandas as pd
import pathlib
import lightgbm as lgbm
from json2html import *
import shutil
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor
import base64
import click
import flask as flask
from resources.libs import model_factory, model_names
from flask import current_app, g, url_for, session
from flask.cli import with_appcontext
from flask import Flask, request, jsonify, render_template, flash
from flask import Markup
import pickle

from werkzeug.utils import redirect

app = Flask(__name__, template_folder='resources/templates', static_folder='resources/static')
app.config['UPLOAD_FOLDER'] = "db/database/CSV"
app.secret_key = "123"
Database = "KETEP_Database.db"

# con=sql.connect("KETEP_Database.db")
# con.execute("create table if not exists model_names(m_id integer primary key autoincrement, mdl_names TEXT)")
# con.execute("create table if not exists add_mdl_data(id integer primary key autoincrement, snerio_nm TEXT, trand_model TEXT, exceldata TEXT)")
# con.execute("create table if not exists result_data(r_id integer primary key autoincrement, jason_path TEXT)")
# con.close()


model = ""
root_db_dir = pathlib.Path('db/database')


def round_floats(o):
    if isinstance(o, float):
        return round(o, 1)
    if isinstance(o, dict):
        return {k: round_floats(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [round_floats(x) for x in o]
    return o


@app.route('/delete_record/<string:id>')
def delete_entry(id):
    sqliteConnection = sql.connect(Database)
    cursor = sqliteConnection.cursor()
    cursor1 = sqliteConnection.cursor()
    print("Connected to SQLite")
    cursor.execute('delete from add_mdl_data where id = ?', (id,))
    cursor1.execute('delete from result_data where r_id = ?', (id,))
    sqliteConnection.commit()
    return redirect(url_for("home"))


@app.route('/', methods=['GET', 'POST'])
def home():
    # for i in range(len(model_names)):
    #     sqliteConnection = sql.connect(Database)
    #     cursor = sqliteConnection.cursor()
    #     cursor.execute("Insert into model_names(mdl_names) values(?)", (model_names[i],))
    #     sqliteConnection.commit()
    #     sqliteConnection.close()
    con = sql.connect(Database)
    con.row_factory = sql.Row
    cur = con.cursor()
    cur.execute("select * from add_mdl_data")
    rows = cur.fetchall()
    con.close()
    return render_template("index.html", rows=rows)


@app.route('/up?<string:record_id>', methods=['GET', 'POST'])
def update_record(record_id):
    con = sql.connect(Database)
    con.row_factory = sql.Row
    cur = con.cursor()
    cur.execute('select * from add_mdl_data where id = ?', (record_id,))
    data = cur.fetchone()

    con.close()
    if request.method == 'POST':
        uploadExcel = request.files['uploadexcel']
        if uploadExcel.filename != '':
            random_number = random.randint(1, 1000)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], str(random_number) + uploadExcel.filename)
            uploadExcel.save(filepath)
            trand_model = request.form['trainmodel']
            connection = sql.connect(Database, timeout=10)
            cursor = connection.cursor()
            cursor.execute("UPDATE add_mdl_data SET trand_model=?, exceldata=?  where id=?",
                           (trand_model, uploadExcel.filename, record_id))
            connection.commit()
            connection.close()
            dest = root_db_dir / str(random_number)
            dest.mkdir(exist_ok=True, parents=True)
            result_data = model.predict_from_file(filepath, dest)
            print(jsonify(result_data))
            import json
            filename = str(random_number) + 'result.json'
            dest = root_db_dir / str(random_number)
            with open(dest / 'result.json', 'w') as f:
                json.dump(result_data, f, indent=2)
            sqliteConnection = sql.connect(Database)
            cursor = sqliteConnection.cursor()
            # file_path = pathlib.Path('db/database/generated')
            cursor.execute("Update result_data SET jason_path=? where r_id=?", (str(dest), record_id))
            sqliteConnection.commit()
            sqliteConnection.close()
            return redirect(url_for("home"))
    else:
        return render_template("update.html", data=data, model_names=model_names)


@app.route('/?<string:mrecord_id>', methods=['GET', 'POST'])
def forecast(mrecord_id):
    con = sql.connect(Database)
    con.row_factory = sql.Row
    cur = con.cursor()
    session["result_id"] = mrecord_id
    cur.execute("select * from result_data where r_id=?", (mrecord_id,))
    data = cur.fetchall()
    # print(data)

    for val in data:
        # path = os.path.join("db/database/CSV", val[1])
        # jason_path=os.path.join(val[5] + '/result.json')
        json_path = pathlib.Path(val[1]) / 'result.json'
        # print(val[3])
        # data = pd.read_csv(path)

        with open(json_path, 'r') as myfile:
            jsdata = myfile.read()

            heat_obj = json.loads(jsdata)
            for heat in heat_obj:
                heat["Date"] = heat["DateTime"]
                heat["Temperature"] = heat["temp"]
                heat["Windspeed"] = heat["windspeed"]
                heat["Humidity"] = heat["humidity"]
                heat["Holiday"] = heat["holiday"]
                heat["Prediction"] = heat["pred"]

                del heat["DateTime"]
                del heat["temp"]
                del heat["windspeed"]
                del heat["humidity"]
                del heat["holiday"]
                del heat["pred"]
                del heat["month"]
                del heat["hour"]
                del heat["shap"]

            jsdata = json.dumps(round_floats(heat_obj))

            # jsdata = jsdata.replace("DateTime", "Date")
            # jsdata = jsdata.replace("temp", "Temperature")
            # jsdata =jsdata.replace("windspeed", "Wind speed")
            # jsdata =jsdata.replace("humidity", "Humidity")
            # jsdata = jsdata.replace("holiday", "Is holiday")
            print(jsdata)
    con.close()
    html_table = json2html.convert(jsdata, table_attributes="class=\"table table-sm\"")

    html_table = html_table.replace('<th>Prediction</th>', '<th class="table-primary">Prediction</th>')
    return render_template("forecast.html", data1=html_table)


@app.route('/newScenario.html', methods=['GET', 'POST'])
def edit():
    if request.method == 'POST':
        selected_items = request.form.get('trainmodel')
        model = model_factory[selected_items]
        uploadExcel = request.files['uploadexcel']
        if uploadExcel.filename != '':
            random_number = random.randint(1, 1000)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], str(random_number) + uploadExcel.filename)
            uploadExcel.save(filepath)
            snerio_name = request.form['scnerio_name']
            trand_model = request.form['trainmodel']
            connection = sql.connect(Database, timeout=10)
            cursor = connection.cursor()
            cursor.execute("INSERT INTO add_mdl_data(snerio_nm, trand_model, exceldata) VALUES(?,?,?)",
                           (snerio_name, trand_model, str(random_number) + uploadExcel.filename))
            connection.commit()
            connection.close()
            dest = root_db_dir / str(random_number)
            dest.mkdir(exist_ok=True, parents=True)
            result_data = model.predict_from_file(filepath, dest)
            print(jsonify(result_data))

            import json
            # prediction_result_path = dest / 'result.json'
            with open(dest / 'result.json', 'w') as f:
                json.dump(result_data, f, indent=2)
            sqliteConnection = sql.connect(Database)
            cursor = sqliteConnection.cursor()
            cursor.execute('insert into result_data(jason_path) values(?)', (str(dest),))
            # cursor.execute('insert into result_data(jason_path) values(?)', (dest,))
            sqliteConnection.commit()
            sqliteConnection.close()
        return redirect(url_for("home"))
    else:
        con = sql.connect(Database)
        con.row_factory = sql.Row
        cur = con.cursor()
        cur.execute("select * from model_names")
        rows = cur.fetchall()
        con.close()
        return render_template("newScenario.html", rows=rows, model_names=model_names)


@app.route('/analysis.html', methods=['GET', 'POST'])
def analysis():
    id = session["result_id"]
    print(id)
    folder_path = ""
    con = sql.connect(Database)
    con.row_factory = sql.Row
    cur = con.cursor()
    cur.execute("select * from add_mdl_data,result_data where id=?", (id,))
    data = cur.fetchall()
    print(data)
    for val in data:
        path = os.path.join("db/database/CSV", val[3])
        data = pd.read_csv(path)
        data.drop('hour', axis=1, inplace=True)
        data.drop('month', axis=1, inplace=True)
        jason_path = str(pathlib.Path(val[5]))
        model_name = val[2]
        if model_name == "deayang_ai_lightbgm":
            folder_path = "generated_images/deayang_ai_lightbgm/"
        elif model_name == "deayang_ai_xgboost":
            folder_path = "generated_images/deayang_ai_xgboost/"
        elif model_name == "deayang_ai_random_forest":
            folder_path = "generated_images/deayang_ai_randomforest/"
        elif model_name == "deayang_ai_mlp_regressor":
            folder_path = "generated_images/deayang_ai_mlpregressor/"
        base64_imageslist = []
        for images in os.listdir(jason_path):
            if images.endswith(".png"):
                with open(jason_path + '/' + images, 'rb') as f:
                    base64_img = base64.b64encode(f.read()).decode()
                    # ENG                    html_string = Markup(
                    # ENG                        "<a class='example-image-link' href='data:image/png;base64, " + base64_img + "'  data-lightbox='example-set' data-title='The bottom of a waterfall plot starts as the expected value of the model output, and then each row shows how the positive (red) or negative (blue) contribution of each feature moves the value from the expected model output over the background dataset to the model output for this prediction.'><img class='example-image' src='data:image/png;base64, " + base64_img + "' height='250' width='350'></a>")
                    html_string = Markup(
                        "<a class='example-image-link' href='data:image/png;base64, " + base64_img + "'  data-lightbox='example-set' data-title='폭포수 차트는 모델 기대 값 f(x)에서 시작하여 각 변수별 기여도를 양(빨간색) 또는 음(파란색)을 계산하여 최종적으로 예측된 모델 출력 값 E[f(X)] 를 보여준다.'><img class='example-image' src='data:image/png;base64, " + base64_img + "' height='250' width='350'></a>")
                    base64_imageslist.append(html_string)

    con.close()
    data['img'] = base64_imageslist
    # data['img'] = base64_imageslist
    return render_template("analysis.html",
                           data=data.to_dict(orient='records'),
                           jason_path=jason_path, base64_imageslist=base64_imageslist, folder_path=folder_path,
                           model_name=model_name)


# @app.route('/newScenario.html')
# def newScenario():
#     return render_template("newScenario.html")


@app.route('/predict')
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)
    print(jsonify(output))

    return render_template('index.html', prediction_text='Employee Salary should be $ {}'.format(output))


@app.route('/models_list', methods=['GET', "POST"])
def trained_models():
    if request.method == "GET":
        con = sql.connect(Database)
        con.row_factory = sql.Row
        cur = con.cursor()
        cur.execute("select * from models_list")
        rows = cur.fetchall()
        con.close()
        return render_template("models_list.html", rows=rows)


"""
    model = model_factory["deayang_ai_xgboost"]
    model.train("resources/libs/models/deayang_ai_lightbgm/clean_dataset.csv")
    # model = model_factory['deayang_ai_xgboost']
    # model.train('resources/libs/models/deayang_ai_lightbgm/clean_dataset.csv')
    # model.save_model('resources/libs/models/deayang_ai_xgboost/pretrained.pkl')
    #
    # dest = pathlib.Path('db/database/generated')
    # dest.mkdir(exist_ok=True, parents=True)
    # model.predict_from_file('resources/libs/models/deayang_ai_xgboost/samples.csv', dest)

"""


@app.route('/models_list_save', methods=['GET', "POST"])
def list_models():
    if request.method == 'POST':
        uploadExcel = request.files['uploadexcel']
        selected_items = request.form.get('trainmodel')
        #model = model_factory[selected_items]
        location = uploadExcel.filename
        absolute_path = os.path.abspath(location)
        model_name = request.form['model_name']
        if selected_items == 'deayang_ai_lightbgm':
            dataset4 = pd.read_csv(absolute_path, parse_dates=True)
            train_set4 = dataset4.set_index('DateTime', drop=True)

            # train_set에서 y와 X 분리
            y4 = train_set4['Amount(total)']
            X4 = train_set4.drop(['Amount(total)'], axis=1)
            columns4 = X4.columns
            print(columns4)
            values_str = ','.join(columns4)
            print(values_str)
            X4 = X4.values
            y4 = y4.values

            params = {
                'learning_rate': 0.01,
                'objective': 'regression',
                'metric': 'mae',
                'seed': 42
            }
            tscv = TimeSeriesSplit()
            best_val_score = -1
            for train_index, val_index in tscv.split(X4):
                X_train, X_val = X4[train_index], X4[val_index]
                y_train, y_val = y4[train_index], y4[val_index]
                X_train = pd.DataFrame(X_train, columns=columns4)
                X_val = pd.DataFrame(X_val, columns=columns4)

                # training model
                train_ds1 = lgbm.Dataset(X_train, y_train)
                model01 = lgbm.train(params, train_ds1, 200)

                y_pred = model01.predict(X_train)
                training_r2_score = r2_score(y_train, y_pred)
                print("train r2 score : ", training_r2_score)
                y_pred = model01.predict(X_val)
                test_r2_score = r2_score(y_val, y_pred)
                print("validation r2 score : ", test_r2_score)

                # select best splits
                if test_r2_score > best_val_score:
                    best_val_score = round(test_r2_score,2)
            print('best val score:', best_val_score)
            connection = sql.connect(Database, timeout=10)
            cursor = connection.cursor()
            cursor.execute("INSERT INTO models_list(m_name, train_columns, perf_on_val_data) VALUES(?,?,?)",
                           (model_name, values_str, best_val_score))
            connection.commit()
            connection.close()
        elif selected_items == 'deayang_ai_xgboost':
            dataset4 = pd.read_csv(absolute_path, parse_dates=True)
            train_set4 = dataset4.set_index('DateTime', drop=True)

            # train_set에서 y와 X 분리
            y4 = train_set4['Amount(total)']
            X4 = train_set4.drop(['Amount(total)'], axis=1)
            columns4 = X4.columns
            print(columns4)
            values_str = ','.join(columns4)
            print(values_str)
            X4 = X4.values
            y4 = y4.values

            tscv = TimeSeriesSplit()
            best_val_score = -1
            for train_index, val_index in tscv.split(X4):
                X_train, X_val = X4[train_index], X4[val_index]
                y_train, y_val = y4[train_index], y4[val_index]
                X_train = pd.DataFrame(X_train, columns=columns4)
                X_val = pd.DataFrame(X_val, columns=columns4)

                # training model

                # specify parameters via map

                model01 = XGBRegressor(seed=40)

                model01.fit(X_train, y_train)

                y_pred = model01.predict(X_train)
                training_r2_score = r2_score(y_train, y_pred)
                print("train r2 score : ", training_r2_score)
                y_pred = model01.predict(X_val)
                test_r2_score = r2_score(y_val, y_pred)
                print("validation r2 score : ", test_r2_score)

                # select best splits
                if test_r2_score > best_val_score:
                    best_val_score = round(test_r2_score,2)
            print('best val score:', best_val_score)
            connection = sql.connect(Database, timeout=10)
            cursor = connection.cursor()
            cursor.execute("INSERT INTO models_list(m_name, train_columns, perf_on_val_data) VALUES(?,?,?)",
                           (model_name, values_str, best_val_score))
            connection.commit()
            connection.close()
        elif selected_items == 'deayang_ai_random_forest':
            dataset4 = pd.read_csv(absolute_path, parse_dates=True)
            train_set4 = dataset4.set_index('DateTime', drop=True)

            # train_set에서 y와 X 분리
            y4 = train_set4['Amount(total)']
            X4 = train_set4.drop(['Amount(total)'], axis=1)
            columns4 = X4.columns
            print(columns4)
            values_str = ','.join(columns4)
            print(values_str)
            X4 = X4.values
            y4 = y4.values
            best_val_score = -1
            tscv = TimeSeriesSplit()
            for train_index, val_index in tscv.split(X4):
                X_train, X_val = X4[train_index], X4[val_index]
                y_train, y_val = y4[train_index], y4[val_index]
                X_train = pd.DataFrame(X_train, columns=columns4)
                X_val = pd.DataFrame(X_val, columns=columns4)

                # training model

                # specify parameters via map

                model01 = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=35)

                model01.fit(X_train, y_train)
                y_pred = model01.predict(X_train)
                training_r2_score = r2_score(y_train, y_pred)
                print("train r2 score : ", training_r2_score)
                y_pred = model01.predict(X_val)
                test_r2_score = r2_score(y_val, y_pred)
                print("validation r2 score : ", test_r2_score)
                # select best splits
                if test_r2_score > best_val_score:
                    best_val_score = round(test_r2_score,2)
            print('best val score:', best_val_score)
            connection = sql.connect(Database, timeout=10)
            cursor = connection.cursor()
            cursor.execute("INSERT INTO models_list(m_name, train_columns, perf_on_val_data) VALUES(?,?,?)",
                           (model_name, values_str, best_val_score))
            connection.commit()
            connection.close()
        elif selected_items=='deayang_ai_mlp_regressor':
            dataset4 = pd.read_csv(absolute_path, parse_dates=True)
            train_set4 = dataset4.set_index('DateTime', drop=True)

            # train_set에서 y와 X 분리
            y4 = train_set4['Amount(total)']
            X4 = train_set4.drop(['Amount(total)'], axis=1)
            columns4 = X4.columns
            values_str = ','.join(columns4)
            X4 = X4.values
            y4 = y4.values
            tscv = TimeSeriesSplit()
            best_val_score = -1
            for train_index, val_index in tscv.split(X4):
                X_train, X_val = X4[train_index], X4[val_index]
                y_train, y_val = y4[train_index], y4[val_index]
                X_train = pd.DataFrame(X_train, columns=columns4)
                X_val = pd.DataFrame(X_val, columns=columns4)

                # training model

                # specify parameters via map

                model01 = MLPRegressor(hidden_layer_sizes=(10, 100), activation='relu', solver='adam', random_state=1,
                                       max_iter=5000,
                                       learning_rate='constant', early_stopping=False)

                model01.fit(X_train, y_train)

                y_pred = model01.predict(X_train)
                training_r2_score = r2_score(y_train, y_pred)
                print("train r2 score : ", training_r2_score)
                y_pred = model01.predict(X_val)
                test_r2_score = r2_score(y_val, y_pred)
                print("validation r2 score : ", test_r2_score)

                # select best splits
                if test_r2_score > best_val_score:
                    best_val_score = test_r2_score
            print('best val score:', best_val_score)
            connection = sql.connect(Database, timeout=10)
            cursor = connection.cursor()
            print(model_name)
            cursor.execute("INSERT INTO models_list(m_name, train_columns, perf_on_val_data) VALUES(?,?,?)",
                           (model_name, values_str, best_val_score))
            connection.commit()
            connection.close()
    """
    random_forest = DeyangAICenter_randomforest()
    model = model_factory['deayang_ai_random_forest']
    #model.train('resources/libs/models/deayang_ai_lightbgm/clean_dataset.csv')
    print(random_forest.get_best_val_score())
    values_str = ','.join(model.features_names)
    connection = sql.connect(Database, timeout=10)
    cursor = connection.cursor()
    cursor.execute("INSERT INTO models_list(m_name, train_columns, perf_on_val_data) VALUES(?,?,?)",
                   (model.model_name, values_str,"60%"))
    connection.commit()
    connection.close()
    """
    return render_template("TrainingModel.html")


"""
    if request.method == 'GET':
        connection = sql.connect(Database, timeout=10)
        cursor = connection.cursor()
        cursor.execute("INSERT INTO models_list(m_name, train_columns, perf_on_val_data) VALUES(?,?,?)",
                       (model.model_name, values_str,))
        connection.commit()
        connection.close()
"""
# model.save_model('resources/libs/models/deayang_ai_xgboost/pretrained.pkl')
#
# dest = pathlib.Path('db/database/generated')
# dest.mkdir(exist_ok=True, parents=True)
# model.predict_from_file('resources/libs/models/deayang_ai_xgboost/samples.csv', dest)


if __name__ == "__main__":
    app.run(host='127.0.0.1', threaded=True)
