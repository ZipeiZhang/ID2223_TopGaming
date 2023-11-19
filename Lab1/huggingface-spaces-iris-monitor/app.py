
import gradio as gr
import hopsworks

def create_ui():
    project = hopsworks.login()
    fs = project.get_feature_store()

    dataset_api = project.get_dataset_api()

    # Download images
    dataset_api.download("Resources/images/latest_iris.png")
    dataset_api.download("Resources/images/actual_iris.png")
    dataset_api.download("Resources/images/df_recent.png")
    dataset_api.download("Resources/images/confusion_matrix.png")

    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column():
                gr.Label("Today's Predicted Image")
                gr.Image("latest_iris.png", elem_id="predicted-img")
            with gr.Column():
                gr.Label("Today's Actual Image")
                gr.Image("actual_iris.png", elem_id="actual-img")
        with gr.Row():
            with gr.Column():
                gr.Label("Recent Prediction History")
                gr.Image("df_recent.png", elem_id="recent-predictions")
            with gr.Column():
                gr.Label("Confusion Matrix with Historical Prediction Performance")
                gr.Image("confusion_matrix.png", elem_id="confusion-matrix")

    return demo

if __name__ == "__main__":
    demo = create_ui()
    demo.launch()

# import gradio as gr
# from PIL import Image
# import hopsworks

# project = hopsworks.login()
# fs = project.get_feature_store()

# dataset_api = project.get_dataset_api()

# dataset_api.download("Resources/images/latest_iris.png")
# dataset_api.download("Resources/images/actual_iris.png")
# dataset_api.download("Resources/images/df_recent.png")
# dataset_api.download("Resources/images/confusion_matrix.png")

# with gr.Blocks() as demo:
#     with gr.Row():
#       with gr.Column():
#           gr.Label("Today's Predicted Image")
#           input_img = gr.Image("latest_iris.png", elem_id="predicted-img")
#       with gr.Column():          
#           gr.Label("Today's Actual Image")
#           input_img = gr.Image("actual_iris.png", elem_id="actual-img")        
#     with gr.Row():
#       with gr.Column():
#           gr.Label("Recent Prediction History")
#           input_img = gr.Image("df_recent.png", elem_id="recent-predictions")
#       with gr.Column():          
#           gr.Label("Confusion Maxtrix with Historical Prediction Performance")
#           input_img = gr.Image("confusion_matrix.png", elem_id="confusion-matrix")        

# demo.launch()
