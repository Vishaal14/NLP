import streamlit as st
from PIL import Image
import pickle
import joblib
from ultralytics import YOLO

try:
    label_names_dict = joblib.load('label_names_dict.pkl')
except FileNotFoundError:
    st.error("label_names_dict.pkl file not found. Please make sure the file exists.")
    st.stop()
except Exception as e:
    st.error(f"Error loading label names dictionary: {e}")
    st.stop()

# Load the YOLO model
model = YOLO("yolov8_weights.pt")

def predict(image, model):
    results = model(image)
    predictions = []
    for result in results:
        probs = result.probs
        max_prob, max_index = probs.data.max(dim=0)
        class_index = max_index.item()
        if class_index in label_names_dict:
            class_name = label_names_dict[class_index]
            probability = max_prob.item()
            predictions.append((class_name, probability))
        else:
            print(f"Class index {class_index} not found in label_names_dict.")
    return predictions

def main():
    st.title("YOLOv8 Image Classification")

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image.", use_column_width=True)

        # Perform inference
        if st.button("Classify"):
            predictions = predict(image, model)
            for class_name, probability in predictions:
                st.write(f"Prediction: {class_name}, Probability: {probability}")
                
            # Redirect to another page while preserving class_name
            params = {"class_name": class_name}
            st.experimental_set_query_params(**params)
            st.experimental_rerun()

# Main page
if __name__ == "__main__":
    main()

# Other page
class_name = st.experimental_get_query_params().get("class_name", [""])[0]
if class_name:
    st.write(f"You are redirected to another page. Class Name: {class_name}")

    # Display options
    option = st.radio("Select an option:", ["Information", "Season"])
    st.write(f"Selected Option: {option} ")
    if option=="Information":
        st.write("HMM " ,class_name)
    if option=="Season":
        st.write("Oh  ",class_name)
