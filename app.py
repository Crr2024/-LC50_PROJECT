from flask import Flask,render_template,request
import numpy as np
import pickle

app = Flask(__name__,template_folder="template")

def ValuePredictor(to_predict_list):
    loaded_model = pickle.load(open("home/rohit/Desktop/ALL_Projects/LC50 project/LC50_model.pkl", "rb"))
    result = loaded_model.predict(to_predict_list)
    return result
     
@app.route('/', methods = ['GET','POST'])
def result():   
    return render_template("lc50_html.html")
    
@app.route('/result', methods = ['POST'])
def result_():        
     CIC0 = float(request.form.get("CIC0"))
     SM1_Dz = float(request.form.get("SM1_Dz(Z)"))
     GATS1i = float(request.form.get("GATS1i"))
     NdsCH = float(request.form.get("NdsCH"))
     NdssC = float(request.form.get("NdssC"))
     MLOGP = float(request.form.get("MLOGP"))
     to_predict_list = np.array([CIC0,SM1_Dz,GATS1i,NdsCH,NdssC,MLOGP]).reshape(1, -1)
     result_ = ValuePredictor(to_predict_list)    
     print({"result":list(result_)})            
     return render_template("result.html",result_=result_)

if __name__ == "__main__":
    app.run(debug = True, port = 3543)
