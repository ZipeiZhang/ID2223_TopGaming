import gradio as gr
from PIL import Image
import requests
import hopsworks
import joblib
import pandas as pd
import random
import numpy as np
from io import BytesIO
from great_expectations.dataset import PandasDataset
from great_expectations.core import ExpectationSuite, ExpectationConfiguration
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
import seaborn as sns
import matplotlib.pyplot as plt
api = '151p8WWCoctBzBeg.wRj1VwLA6wwjCS2aG7A51NsbhEbqVZ35wLl5g03b85EeetLKtpsO9bDOjy8DR2O3'
project = hopsworks.login(api_key_value = api)
fs = project.get_feature_store()


mr = project.get_model_registry()
model = mr.get_model("wine_model_new", version=1)
model_dir = model.download()
model = joblib.load(model_dir + "/wine_model_new.pkl")
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
            "fixed_acidity": random.uniform(3.6, 15.0),  
            "volatile_acidity": random.uniform(0.061, 1.5),
            "citric_acid": random.uniform(0.1, 7.0),
            "residual_sugar":random.uniform(0.35, 60),
            "chlorides":random.uniform(0.1, 0.60),
            "free_sulfur_dioxide":random.uniform(0.85, 280),
            "total_sulfur_dioxide":random.uniform(5.6, 400),
            "density":random.uniform(0.98, 1),
            "ph":random.uniform(0.35, 4.2),
            "sulphates":random.uniform(0.3,2.3),
            "alcohol":random.uniform(7.9, 12.5),
            "quality": 1,
        }

    wine_fg = fs.get_or_create_feature_group(
        name="wine_2",
        version=2,
        primary_key=["fixed_acidity","volatile_acidity","citric_acid", "residual_sugar"	,"chlorides",	
                    "free_sulfur_dioxide",	"total_sulfur_dioxide",	"density","pH","sulphates","alcohol","quality"], 
        description="For new wine data")
    # query = wine_fg.select_all()
    # feature_view_new = fs.create_feature_view(
    #     name='wine_2',
    #     query=query
    # )

    
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
        return synthetic_wines_df
    else:
        print("Data validation failed:", results)
        return None
   

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d',cmap="crest")
    plt.xlabel('Predicted')
    plt.ylabel('True')
    # Convert the matplotlib plot to a PIL Image
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    img = Image.open(buf)
    return img

def wine_predict(fixed_acidity, volatile_acidity, citric_acid, residual_sugar, 
         chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density, 
         ph, sulphates, alcohol, num_samples, num_samples_matrix):
    print("Calling function")
    feature_view = fs.get_feature_view(name="wine_2", version=1)

    X_train, X_test, y_train, y_test = feature_view.train_test_split(test_size = 0.2)
    num_samples_matrix = int(num_samples_matrix)
    X_test = X_test[:num_samples_matrix]
    y_test = y_test[:num_samples_matrix]
    y_pred = model.predict(X_test)
    confusion_matrix_plot = plot_confusion_matrix(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted')
    rec = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    pred_results = pd.DataFrame([['Random Forest', acc, prec, rec, f1]],
                columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])

    if num_samples == -1:
        df = pd.DataFrame([[fixed_acidity, volatile_acidity, citric_acid, residual_sugar, 
         chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density, 
         ph, sulphates, alcohol]], 
        columns=['fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar', 
         'chlorides', 'free_sulfur_dioxide', 'total_sulfur_dioxide', 'density', 
         'ph', 'sulphates', 'alcohol'])
        print("Predicting")
        print(df)
        feature_view = fs.get_feature_view(name="wine_2", version=1)
        batch_data = feature_view.get_batch_data()
        synthetic_wines_df = batch_data[:50]


        res = model.predict(df)

    else:
        # Generate and predict num_samples
        synthetic_wines_df = generate_samples(num_samples)  # This will also insert the data into the feature store
        synthetic_wines_df = synthetic_wines_df.drop('quality', axis=1)
        y_pred_new = model.predict(synthetic_wines_df)
        synthetic_wines_df = synthetic_wines_df[:50]
        res = y_pred_new.mean()

    if isinstance(synthetic_wines_df, pd.DataFrame):
        return res, synthetic_wines_df, confusion_matrix_plot,pred_results
    else:
        return res, pd.DataFrame(synthetic_wines_df), confusion_matrix_plot,pred_results

    
def update_dashboard(fixed_acidity, volatile_acidity, citric_acid, residual_sugar, 
         chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density, 
         ph, sulphates, alcohol, num_samples, num_samples_matrix):
    

    predction ,generated_samples,confusion_matrix_plot,pred_results= wine_predict(fixed_acidity, 
                                                                        volatile_acidity, citric_acid, residual_sugar, 
                                                                        chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density, 
                                                                        ph, sulphates, alcohol, num_samples, num_samples_matrix)
    

    return predction, confusion_matrix_plot, pred_results,generated_samples
demo = gr.Interface(
    fn=update_dashboard,
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
        gr.inputs.Number(default=-1, label="Samples number to generate"),
        gr.inputs.Number(default=50, label="Samples number for confusion matrix")
    ],
    outputs=[
            gr.outputs.Textbox(label="Predicted Quality"),
            gr.outputs.Image(label="Confusion Matrix",type="pil"),
            gr.outputs.Dataframe(label="Most Recent Prediction results",type='pandas'),
            gr.outputs.Dataframe(label="Most Recent Generated Wine Data",type='pandas')
            ],
)
demo.launch(debug=True)