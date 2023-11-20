import gradio as gr
from PIL import Image
import requests
import hopsworks
import joblib
import pandas as pd

api = 'HeCatNGJxisb99Vf.ircWdTrkgbbZpBMU7iPN2zqDIwoTuaSX88LPeISIMJHuzP3icXixNd6JFcWUqakL'
project = hopsworks.login(api_key_value = api)


mr = project.get_model_registry()
model = mr.get_model("wine_model_2", version=1)
model_dir = model.download()
model = joblib.load(model_dir + "/wine_model_2.pkl")
print("Model downloaded")

def wine_predict(fixed_acidity, volatile_acidity, citric_acid, residual_sugar, 
         chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density, 
         ph, sulphates, alcohol):
    print("Calling function")
#     df = pd.DataFrame([[sepal_length],[sepal_width],[petal_length],[petal_width]], 
    df = pd.DataFrame([[fixed_acidity, volatile_acidity, citric_acid, residual_sugar, 
         chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density, 
         ph, sulphates, alcohol]], 
        columns=['fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar', 
         'chlorides', 'free_sulfur_dioxide', 'total_sulfur_dioxide', 'density', 
         'ph', 'sulphates', 'alcohol'])
    print("Predicting")
    print(df)
    # 'res' is a list of predictions returned as the label.
    res = model.predict(df)
    # We add '[0]' to the result of the transformed 'res', because 'res' is a list, and we only want 
    # the first element.
#     print("Res: {0}").format(res)
    # print(res)
    return res
    
        
demo = gr.Interface(
    fn=wine_predict,
    title="Wine Quality Predictive Analytics",
    description="Enter the wine characteristics to predict its quality.",
    allow_flagging="never",
    inputs=[
        gr.inputs.Number(default=7.4, label="Fixed Acidity"),
        gr.inputs.Number(default=0.7, label="Volatile Acidity"),
        gr.inputs.Number(default=0.0, label="Citric Acid"),
        gr.inputs.Number(default=1.9, label="Residual Sugar"),
        gr.inputs.Number(default=0.076, label="Chlorides"),
        gr.inputs.Number(default=11, label="Free Sulfur Dioxide"),
        gr.inputs.Number(default=34, label="Total Sulfur Dioxide"),
        gr.inputs.Number(default=0.9978, label="Density"),
        gr.inputs.Number(default=3.51, label="pH"),
        gr.inputs.Number(default=0.56, label="Sulphates"),
        gr.inputs.Number(default=9.4, label="Alcohol")
    ],
    outputs=gr.outputs.Textbox(label="Predicted Quality"))

demo.launch(debug=True)

