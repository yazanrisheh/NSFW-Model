# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the UV package manager using pip without caching
RUN pip install --no-cache-dir uv

# Create a virtual environment using UV and install dependencies without caching
RUN uv venv VE_model --python 3.11 && \
    . VE_model/bin/activate && \
    uv pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code to the container
COPY . .

# Streamlit listens on port 8502
EXPOSE 8502

# Activate the virtual environment and run the Streamlit app on port 8502
CMD ["/bin/bash", "-c", ". VE_model/bin/activate && streamlit run main.py --server.fileWatcherType none --server.address 0.0.0.0 --server.port 8502"]
