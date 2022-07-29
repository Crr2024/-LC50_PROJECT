from flask import Flask,render_template,request
import numpy as np
import pandas as pd 
import pickle
from csv import reader
from pyparsing import null_debug_action
from werkzeug.utils import secure_filename
app = Flask(__name__,template_folder="template")

def ValuePredictor(to_predict_list):
    loaded_model = pickle.load(open("/home/rohit/Desktop/ALL_Projects/LC50 project/LC50_model.pkl", "rb"))
    result = loaded_model.predict(to_predict_list)
    return result

@app.route('/', methods = ['GET','POST'])
def result():   
    return render_template("lc50_html.html")

@app.route('/result', methods = ['POST'])
def result_csv():
    if  request.files and request.files['File']:
        CSV = request.files['File']
        CSV.save(secure_filename(CSV.filename))
        result_csv = []
        with open(CSV.filename, 'r') as read_obj:
            csv_reader = reader(read_obj)
            df = pd.read_csv(CSV.filename)
            columns = len(df.columns)
            rows = len(df)
            row_to_list_= list(map(list, csv_reader))
            if columns != 6:
                print('Expected 6 input values but {} were given',columns)
            for i in range(1,rows):
                to_predict_list_csv = row_to_list_[i]
                to_predict_list_csv = [list(np.array(to_predict_list_csv).reshape(1,-1))[0][:-1]]
                result_csv.append(ValuePredictor(to_predict_list_csv).tolist())
        print({"result":list(result_csv)})            
        return render_template("result.html",result_csv=result_csv)
    else:
        CIC0 = float(request.form.get("CIC0"))
        SM1_Dz = float(request.form.get("SM1_Dz(Z)"))
        GATS1i = float(request.form.get("GATS1i"))
        NdsCH = float(request.form.get("NdsCH"))
        NdssC = float(request.form.get("NdssC"))
        MLOGP = float(request.form.get("MLOGP"))
        to_predict_list = np.array([CIC0,SM1_Dz,GATS1i,NdsCH,NdssC,MLOGP]).reshape(1, -1)
        result_ = ValuePredictor(to_predict_list)    
        print({"result":list(result_)})            
        return render_template("result_values.html",result_=result_)

        
if __name__ == "__main__":
    app.run(debug = True, port = 3543)    