import gradio as gr
from PIL import Image
import requests
import hopsworks
import joblib
import pandas as pd
import random
from great_expectations.dataset import PandasDataset
from great_expectations.core import ExpectationSuite, ExpectationConfiguration
api = 'HeCatNGJxisb99Vf.ircWdTrkgbbZpBMU7iPN2zqDIwoTuaSX88LPeISIMJHuzP3icXixNd6JFcWUqakL'
project = hopsworks.login(api_key_value = api)
fs = project.get_feature_store()


mr = project.get_model_registry()
model = mr.get_model("wine_model_2", version=1)
model_dir = model.download()
model = joblib.load(model_dir + "/wine_model_2.pkl")
print("Model downloaded")
def generate_samples(num):

    def expect(suite, column, min_val, max_val):
        suite.add_expectation(
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_be_between",
            kwargs={
                "column":column, 
                "min_value":min_val,
                "max_value":max_val,
            }
        )
    )
    suite = ExpectationSuite(expectation_suite_name="wine_dimensions")
    expect(suite, "fixed_acidity", 3.5, 16.0)
    expect(suite, "volatile_acidity", 0.06,1.60)
    expect(suite, "citric_acid", 0.0,7.5)
    expect(suite, "residual_sugar", 0.3,66.0)
    expect(suite, "chlorides", 0.00,0.65)
    expect(suite, "free_sulfur_dioxide", 0.8,290.0)
    expect(suite, "total_sulfur_dioxide", 5.5,450.0)
    expect(suite, "density", 0.95,1.03)
    expect(suite, "ph", 0.3,4.5)
    expect(suite, "sulphates", 0.2, 2.5)
    expect(suite, "alcohol", 7.8, 15.5)
    expect(suite, "quality", 0,2)


    def generate_synthetic_wine(seed):
        random.seed(seed)
        return {
            "fixed_acidity": 5.0,  
            "volatile_acidity": 0.51,
            "citric_acid": 0.3,
            "residual_sugar":1.0,
            "chlorides":random.uniform(0.55, 0.56),
            "free_sulfur_dioxide":30.0,
            "total_sulfur_dioxide":32.0,
            "density":0.98,
            "ph":3.6,
            "sulphates":random.uniform(1.3, 1.5),
            "alcohol":11.5,
            "quality":1
        }

    wine_fg = fs.get_or_create_feature_group(
        name="wine_2",
        version=10,
        primary_key=["fixed_acidity","volatile_acidity","citric_acid", "residual_sugar"	,"chlorides",	
                    "free_sulfur_dioxide",	"total_sulfur_dioxide",	"density","pH","sulphates","alcohol","quality"], 
        description="For new wine data")

    num_samples = int(num)  # Number of synthetic samples to generate

    synthetic_wines_new1 = [generate_synthetic_wine(seed=i) for i in range(num_samples)]
    # synthetic_wines_new = generate_synthetic_wine(seed=2)
    # Convert the list of dictionaries to a DataFrame
    synthetic_wines_df = pd.DataFrame(synthetic_wines_new1)
    synthetic_wines_ge_df = PandasDataset(synthetic_wines_df)
    # Validate the entire DataFrame
    results = synthetic_wines_ge_df.validate(expectation_suite=suite, result_format="SUMMARY")

    # Check if the new data meets the expectations
    if results["success"]:
        wine_fg.insert(synthetic_wines_df, overwrite=False, operation="append")
        print("All synthetic wine data inserted successfully.")
    else:
        print("Data validation failed:", results)
    return synthetic_wines_df

def wine_predict(fixed_acidity, volatile_acidity, citric_acid, residual_sugar, 
         chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density, 
         ph, sulphates, alcohol, num_samples):
    print("Calling function")
    if num_samples == -1:
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
    else:
        # Generate and predict num_samples
        synthetic_wines_df = generate_samples(num_samples)  # This will also insert the data into the feature store
        # wine_2_fg = fs.get_feature_group(name="wine_2", version=10)
        # query = wine_2_fg.select_all()
        # feature_view = fs.get_or_create_feature_view(name="wine_2",
        #                             version=10,
        #                             description="Read from wine dataset",
        #                             labels=["quality"],
        #                             query=query)
        # X_train, X_test, y_train, y_test = feature_view.train_test_split(0.2)
        # batch_data = feature_view.get_batch_data()
        synthetic_wines_df = synthetic_wines_df.drop('quality', axis=1)
        y_pred = model.predict(synthetic_wines_df)
        print("Wine Quality Predictions: ", y_pred)
        print("sample number: ", len(y_pred))
        print("Wine Quality average:: ", y_pred.mean())
        res = y_pred.mean()


    return res
    
        
demo = gr.Interface(
    fn=wine_predict,
    title="Wine Quality Predictive Analytics",
    description="Enter the wine characteristics or the number of samples to predict its quality, enter -1 to predict input sample",
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
        gr.inputs.Number(default=9.4, label="Alcohol"),
        gr.inputs.Number(default=-1, label="num_samples")
    ],
    outputs=gr.outputs.Textbox(label="Predicted Quality"))

demo.launch(debug=True)