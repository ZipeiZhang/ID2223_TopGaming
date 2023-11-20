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

    

# def generate_flower(name, sepal_len_max, sepal_len_min, sepal_width_max, sepal_width_min, 
#                     petal_len_max, petal_len_min, petal_width_max, petal_width_min):
#     """
#     Returns a single iris flower as a single row in a DataFrame
#     """
#     import pandas as pd
#     import random

#     df = pd.DataFrame({ "sepal_length": [random.uniform(sepal_len_max, sepal_len_min)],
#                        "sepal_width": [random.uniform(sepal_width_max, sepal_width_min)],
#                        "petal_length": [random.uniform(petal_len_max, petal_len_min)],
#                        "petal_width": [random.uniform(petal_width_max, petal_width_min)]
#                       })
#     df['variety'] = name

#     return df


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
        "quality": [random.choice([0, 1, 2])],
        "type_white": [random.choice([0,1])]
    })
    return df


# def get_random_iris_flower():
#     """
#     Returns a DataFrame containing one random iris flower
#     """
#     import pandas as pd
#     import random

#     virginica_df = generate_flower("Virginica", 8, 5.5, 3.8, 2.2, 7, 4.5, 2.5, 1.4)
#     versicolor_df = generate_flower("Versicolor", 7.5, 4.5, 3.5, 2.1, 3.1, 5.5, 1.8, 1.0)
#     setosa_df =  generate_flower("Setosa", 6, 4.5, 4.5, 2.3, 1.2, 2, 0.7, 0.3)

#     # randomly pick one of these 3 and write it to the featurestore
#     pick_random = random.uniform(0,3)
#     if pick_random >= 2:
#         iris_df = virginica_df
#         print("Virginica added")
#     elif pick_random >= 1:
#         iris_df = versicolor_df
#         print("Versicolor added")
#     else:
#         iris_df = setosa_df
#         print("Setosa added")

#     return iris_df


def g():
    import hopsworks
    import pandas as pd
    api = 'HeCatNGJxisb99Vf.ircWdTrkgbbZpBMU7iPN2zqDIwoTuaSX88LPeISIMJHuzP3icXixNd6JFcWUqakL'
    project = hopsworks.login(api_key_value = api)
    fs = project.get_feature_store()

    wine_df = generate_wine()

    wine_fg = fs.get_feature_group(name="wine",version=1)
    wine_fg.insert(wine_df)

if __name__ == "__main__":
    if LOCAL == True :
        g()
    else:
        modal.runner.deploy_stub(stub)
        f.remote()
