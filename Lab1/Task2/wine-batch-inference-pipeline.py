import os
import modal
    
LOCAL=False

if LOCAL == False:
   stub = modal.Stub()
   hopsworks_image = modal.Image.debian_slim().pip_install(["hopsworks","joblib","seaborn","scikit-learn==1.1.1","dataframe-image"])
   @stub.function(image=hopsworks_image, schedule=modal.Period(days=1), secret=modal.Secret.from_name("id2223"))
   def f():
       g()

def g():
    import pandas as pd
    import hopsworks
    import joblib
    import datetime
    from PIL import Image
    from datetime import datetime
    import dataframe_image as dfi
    from sklearn.metrics import confusion_matrix
    from matplotlib import pyplot
    import seaborn as sns
    import requests
    import hopsworks
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import classification_report
    from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
    import seaborn as sns
    from matplotlib import pyplot
    from hsml.schema import Schema
    from hsml.model_schema import ModelSchema
    import joblib
    api = '151p8WWCoctBzBeg.wRj1VwLA6wwjCS2aG7A51NsbhEbqVZ35wLl5g03b85EeetLKtpsO9bDOjy8DR2O3'
    project = hopsworks.login(api_key_value = api)
    fs = project.get_feature_store()
    
    mr = project.get_model_registry()
    model = mr.get_model("wine_model_new", version=1)
    model_dir = model.download()
    print("model_dir:",model_dir)
    model = joblib.load(model_dir + "/wine_model_new.pkl")
    
    feature_view = fs.get_feature_view(name="wine_2", version=1)
    batch_data = feature_view.get_batch_data()
    print("shape of batch_data:",batch_data.shape)
    y_pred = model.predict(batch_data)
    count_0 = (y_pred == 0).sum()
    count_1 = (y_pred == 1).sum()
    count_2 = (y_pred == 2).sum()
    print(count_0)
    print(count_1)
    print(count_2)
    print("Wine Quality average:: ", y_pred.mean())
 


if __name__ == "__main__":
    if LOCAL == True :
        g()
    else:
        with stub.run():
            f.remote()

