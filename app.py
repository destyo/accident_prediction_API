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

def var_temporales(var,nombre,rango): 
    list = [{var: nombre}]
    for opcion in rango:
        list.append({var: opcion})
    return list 

@app.route('/')
def home():
    return render_template(
        'indexstyle.html',
        quarter = var_temporales('quarter', 'Month', range(1,13)),
        weekday = var_temporales('weekday', 'Day of week', range(1,8)),
        hour = var_temporales('hour', 'Hour', range(0,24)),
        crash_type = crea_dict('crash_type','Collision configuration'),
        crash_place = crea_dict('crash_place','Roadway configuration'),
        crash_weather = crea_dict('crash_weather','Weather condition'),
        surface_state = crea_dict('surface_state','Road surface'),
        road_slope = crea_dict('road_slope','Road alignment'),
        traffic_state = crea_dict('traffic_state','Traffic control'),
        vehicle_type = crea_dict('vehicle_type', 'Vehicle type'),
        passenger_sex = crea_dict('passenger_sex', 'Person sex' ),
        passenger_safety = crea_dict('passenger_safety', 'Safety device used'),
        passenger_type = crea_dict('passenger_type', 'Road user class' )
        )

@app.route('/predict',methods=['POST'])
def predict():

    input_data = list(request.form.values())
    
    if int(input_data[0]) & int(input_data[4]) & int(input_data[12]) & int(input_data[14]) == True:
        pass
    else:
        print(ValueError)
        

    array_values = np.array(input_data)

    input_df = pd.DataFrame(array_values.reshape(1,-1), columns=['year', 'quarter', 'weekday',
                                            'hour', 'vehicles_involved', 'crash_type',
                                            'crash_place', 'crash_weather', 'surface_state',
                                            'road_slope', 'traffic_state', 'vehicle_type',
                                            'vehicle_age', 'passenger_sex', 'passenger_age'
                                            ,'passenger_safety', 'passenger_type'])

    input_df['quarter'] = (pd.to_numeric(input_df['quarter'])-1)//3 + 1
    input_df['weekday'] = input_df['weekday'].replace({6:3, 7:3, 1:2, 4:2, 5:2, 2:1, 3:1})
    input_df['hour'] = input_df['hour'].replace({0:1, 1:1, 2:1, 3:1, 4:1, 5:1, 6:2, 7:2, 8:2, 9:3, 10:3, 11:3, 11:4, 12:4, 13:4, 14:4, 15:4, 16:4, 17:4, 18:4, 19:5, 20:5, 21:5, 22:6, 23:6})





    pred = model.predict_proba(input_df)[:,1]
    
    output = np.round(*pred, 3) * 100
    #output = round(pred, 3)123

    return render_template('indexstyle.html', prediction_text='La predicción de mortalidad es del',
                                    prediction_prob='{}%'.format(output),
                                    quarter = var_temporales('quarter', 'Month', range(1,13)),
                                    weekday = var_temporales('weekday', 'Day of week', range(1,8)),
                                    hour = var_temporales('hour', 'Hour', range(0,24)),
                                    crash_type = crea_dict('crash_type','Collision configuration'),
                                    crash_place = crea_dict('crash_place','Roadway configuration'),
                                    crash_weather = crea_dict('crash_weather','Weather condition'),
                                    surface_state = crea_dict('surface_state','Road surface'),
                                    road_slope = crea_dict('road_slope','Road alignment'),
                                    traffic_state = crea_dict('traffic_state','Traffic control'),
                                    vehicle_type = crea_dict('vehicle_type', 'Vehicle type'),
                                    passenger_sex = crea_dict('passenger_sex', 'Person sex' ),
                                    passenger_safety = crea_dict('passenger_safety', 'Safety device used'),
                                    passenger_type = crea_dict('passenger_type', 'Road user class' )
                                    )

# @app.route('/results',methods=['POST'])
# def results():

#     data = request.get_json(force=True)
#     prediction = model.predict([np.array(list(data.values()))])

#     output = prediction[0]
#     return jsonify(output)                

if __name__ == "__main__":
    app.run(debug=True)