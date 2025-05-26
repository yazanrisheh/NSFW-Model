# NSFW Image Detection App

A production-ready NSFW Image Detection application built with [Streamlit](https://streamlit.io/), [Falconsai NSFW Model](https://huggingface.co/Falconsai/nsfw_image_detection), and [PyTorch](https://pytorch.org/). This application allows users to either upload their own images or select from a built-in collection to detect NSFW content.

## Features

- **Real-time Inference:** Perform NSFW detection on images using a pre-trained model.
- **User-Friendly Interface:** A beautiful, modern, and production-ready UI built with Streamlit.
- **Flexible Image Source:** Users can upload their own images or use built-in images from the provided `pictures` folder.
- **Containerized Deployment:** Easily deployable using Docker.

## Prerequisites

- **Python 3.11 or higher**
- **Docker (Optional):** For containerized deployment.
- A stable internet connection for downloading the model assets on first run.

## Installation

### Locally

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/yazanrisheh/NSFW-Model.git
   cd https://github.com/yazanrisheh/NSFW-Model.git

2. **Set up Virtual Environment using UV (python version is optional)**
   ```bash
   uv venv venv --python 3.11.5
   ```

   ```bash
   # On Linux/macOS:
    source venv/bin/activate
    # On Windows:
    venv\Scripts\activate
    ```

    ```bash
     uv pip install -r requirements.txt
     ```

3. **Running the Application**
   ```bash
   streamlit run main.py
   ```