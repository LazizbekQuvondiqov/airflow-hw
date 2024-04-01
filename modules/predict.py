# <YOUR_IMPORTS>

import dill
import pandas as pd
import os
import glob
import json

def predict():
    ttt = "C:/Users/User/Desktop/33-modul/airflow_hw/airflow_hw"
    with open(os.path.join(ttt, 'data/models/cars_pipe_202404010243.pkl'), 'rb') as file:
        model = dill.load(file)

    json_dir = os.path.join(ttt, 'data/test')
    json_pattern = os.path.join(json_dir, '*.json')


    for jsons in glob.glob(json_pattern):
        with open(jsons) as f:
            js_read = json.load(f)
        data = pd.DataFrame([js_read])

        pred = model.predict(data)
        t = pd.DataFrame(pred, columns=['predic'])
        common_df = pd.concat([data, t], axis=1)
        print(common_df[['id','predic']])
        common_df.to_csv(f'{ttt}/data/predictions/prediction.csv',index=False)


    pass


if __name__ == '__main__':
    predict()
