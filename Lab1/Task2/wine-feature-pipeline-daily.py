import os
import modal
import pandas as pd
import random

LOCAL=False

if LOCAL == False:
   stub = modal.Stub("wine_daily")
   image = modal.Image.debian_slim().pip_install(["hopsworks"]) 

   @stub.function(image=image, schedule=modal.Period(days=1), secret=modal.Secret.from_name("id2223"))
   def f():
       g()

def generate_wine():
    """
    Returns a single wine record as a single row in a DataFrame
    """
    df = pd.DataFrame({
        "fixed_acidity": [random.uniform(4, 15)],
        "volatile_acidity": [random.uniform(0.1, 1.0)],
        "citric_acid": [random.uniform(0, 1)],
        "residual_sugar": [random.uniform(0.1, 15)],
        "chlorides": [random.uniform(0.01, 0.2)],
        "free_sulfur_dioxide": [random.uniform(5, 60)],
        "total_sulfur_dioxide": [random.uniform(20, 200)],
        "density": [random.uniform(0.990, 1.003)],
        "pH": [random.uniform(2.8, 3.8)],
        "sulphates": [random.uniform(0.3, 2.0)],
        "alcohol": [random.uniform(8, 15)],
        # Assuming binary classification of quality, e.g., 'bad' (0) or 'good' (1)
        "quality": [random.choice([0, 1, 2])]
    })
    return df



def g():
    import hopsworks
    import pandas as pd
    api = '151p8WWCoctBzBeg.wRj1VwLA6wwjCS2aG7A51NsbhEbqVZ35wLl5g03b85EeetLKtpsO9bDOjy8DR2O3'
    project = hopsworks.login(api_key_value = api)
    fs = project.get_feature_store()

    wine_df = generate_wine()

    wine_fg = fs.get_feature_group(name="wine_2",version=1)
    wine_fg.insert(wine_df)

if __name__ == "__main__":
    if LOCAL == True :
        g()
    else:
        modal.runner.deploy_stub(stub)
        f.remote()
