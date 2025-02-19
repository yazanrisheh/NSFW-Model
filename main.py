import streamlit as st
import os
import torch
from PIL import Image
from transformers import AutoModelForImageClassification, ViTImageProcessor
import torch.nn.functional as F

# MUST be the very first Streamlit command
st.set_page_config(
    page_title="NSFW Image Detection",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Caching model and processor for performance
@st.cache_resource
def load_model_and_processor():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForImageClassification.from_pretrained("Falconsai/nsfw_image_detection").to(device)
    processor = ViTImageProcessor.from_pretrained("Falconsai/nsfw_image_detection")
    return model, processor, device

model, processor, device = load_model_and_processor()

def run_inference(image: Image.Image):
    """
    Preprocess the image, run the model, and calculate probabilities.
    """
    # Ensure image is in RGB mode
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits
    probabilities = F.softmax(logits, dim=-1)
    predicted_idx = logits.argmax(-1).item()
    predicted_label = model.config.id2label[predicted_idx]

    return logits, probabilities, predicted_label


def main():
    # Custom CSS for a modern, appealing design
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap');
        html, body, [class*="css"]  {
            font-family: 'Roboto', sans-serif;
        }
        body {
            background: #f5f7fa;
            color: #333;
        }
        /* Header styling */
        .header {
            text-align: center;
            padding: 40px 0;
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            margin-bottom: 30px;
            border-radius: 8px;
        }
        /* Sidebar styling */
        [data-testid="stSidebar"] {
            background-color: #f8f9fa;
        }
        /* Image card styling */
        .image-card {
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            border-radius: 8px;
            overflow: hidden;
        }
        /* Footer styling */
        .footer {
            text-align: center;
            font-size: 0.9rem;
            color: #aaa;
            padding: 20px 0;
            margin-top: 40px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    
    st.markdown("<div class='header'><h1>NSFW Image Detection</h1></div>", unsafe_allow_html=True)
    
   
    st.sidebar.image("falcon_logo.jpeg", use_container_width=False, width=200, caption= "[https://huggingface.co/Falconsai/nsfw_image_detection](https://huggingface.co/Falconsai/nsfw_image_detection)")
    
    source_option = st.sidebar.radio("Choose how to provide an image", ("Upload Image", "Built-in Pictures"))
    
    image = None
    if source_option == "Upload Image":
        uploaded_file = st.sidebar.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "webp"])
        if uploaded_file is not None:
            try:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_container_width=True)
            except Exception as e:
                st.error(f"Error processing uploaded file: {e}")
    else:  # Built-in Pictures option
        picture_folder = "pictures"
        if os.path.exists(picture_folder):
            
            picture_files = [f for f in os.listdir(picture_folder) 
                             if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
            if picture_files:
                selected_file = st.sidebar.selectbox("Select a picture", picture_files)
                image_path = os.path.join(picture_folder, selected_file)
                try:
                    image = Image.open(image_path)
                    st.image(image, caption=f"Built-in Image: {selected_file}", use_container_width=True)
                except Exception as e:
                    st.error(f"Error loading image: {e}")
            else:
                st.sidebar.warning("No images found in the 'pictures' folder.")
        else:
            st.sidebar.warning("The 'pictures' folder does not exist.")
    
    # Run inference if an image is available
    if image is not None:
        if st.button("Run NSFW Detection"):
            with st.spinner("Running inference..."):
                logits, probabilities, predicted_label = run_inference(image)
            st.success("Inference completed!")
            
            st.markdown("### Model Output")
            st.markdown(f"**Predicted Label:** {predicted_label}")
            
            # Display probabilities in a neat table format
            prob_dict = {
                model.config.id2label[i]: f"{prob*100:.2f}%" 
                for i, prob in enumerate(probabilities[0].tolist())
            }
            st.table(prob_dict)
    else:
        st.info("Please select or upload an image to run the detection.")
    st.markdown("<div class='footer'>NSFW Image Detection App © 2025</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
