import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle
import os

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

app = Flask(__name__)
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
# model = pickle.load(open('model.pkl', 'rb'))
xtrain = pd.read_csv('xtrain.csv')

def crea_dict(variable, nombre):
    lista = [{variable : nombre}]
    lista_unique = xtrain[variable].unique()
    lista_unique = sorted(lista_unique, key=lambda x: (str(type(x)), x))
    for opcion in lista_unique:
        lista.append({variable: opcion})

    return lista


@app.route('/')
def home():
    return render_template(
        'indexstyle.html',
        quarter = crea_dict('quarter', 'trimestre'),
        weekday = crea_dict('weekday', 'dia de la semana'),
        hour = crea_dict('hour', 'hora'),
        crash_type = crea_dict('crash_type', 'tipo de accidente'),
        crash_place = crea_dict('crash_place', 'lugar del accidente'),
        crash_weather = crea_dict('crash_weather', 'clima'),
        surface_state = crea_dict('surface_state', 'superficie'),
        road_slope = crea_dict('road_slope', 'inclinacion'),
        traffic_state = crea_dict('traffic_state', 'estado del trafico'),
        vehicle_type = crea_dict('vehicle_type', 'tipo de vehiculo'),
        passenger_sex = crea_dict('passenger_sex', 'sexo'),
        passenger_safety = crea_dict('passenger_safety', 'medida de seguridad'),
        passenger_type = crea_dict('passenger_type', 'tipo de pasajero')
        )

@app.route('/predict',methods=['POST'])
def predict():

    input_data = list(request.form.values())

    array_values = np.array(input_data)

    input_df = pd.DataFrame(array_values.reshape(1,-1), columns=['year', 'quarter', 'weekday',
                                            'hour', 'vehicles_involved', 'crash_type',
                                            'crash_place', 'crash_weather', 'surface_state',
                                            'road_slope', 'traffic_state', 'vehicle_type',
                                            'vehicle_age', 'passenger_sex', 'passenger_age'
                                            ,'passenger_safety', 'passenger_type'])

    pred = model.predict_proba(input_df)[:,1]
    
    output = np.round(*pred, 3) * 100
    #output = round(pred, 3)123

    return render_template('indexstyle.html', prediction_text='La predicción de mortalidad es del',
                                    prediction_prob='{}%'.format(output),
                                    quarter = crea_dict('quarter', 'trimestre'),
                                    weekday = crea_dict('weekday', 'dia de la semana'),
                                    hour = crea_dict('hour', 'hora'),
                                    crash_type = crea_dict('crash_type', 'tipo de accidente'),
                                    crash_place = crea_dict('crash_place', 'lugar del accidente'),
                                    crash_weather = crea_dict('crash_weather', 'clima'),
                                    surface_state = crea_dict('surface_state', 'superficie'),
                                    road_slope = crea_dict('road_slope', 'inclinacion'),
                                    traffic_state = crea_dict('traffic_state', 'estado del trafico'),
                                    vehicle_type = crea_dict('vehicle_type', 'tipo de vehiculo'),
                                    passenger_sex = crea_dict('passenger_sex', 'sexo'),
                                    passenger_safety = crea_dict('passenger_safety', 'medida de seguridad'),
                                    passenger_type = crea_dict('passenger_type', 'tipo de pasajero')
                                    )

# @app.route('/results',methods=['POST'])
# def results():

#     data = request.get_json(force=True)
#     prediction = model.predict([np.array(list(data.values()))])

#     output = prediction[0]
#     return jsonify(output)                

if __name__ == "__main__":
    app.run(debug=True)