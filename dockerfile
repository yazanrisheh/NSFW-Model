# Use an official Python runtime as a parent image
FROM python:3.11-slim



# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
# If you don't have one, you can create a file with the necessary dependencies.
COPY requirements.txt .

# Upgrade pip and install any needed packages specified in requirements.txt
RUN uv pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy the rest of your application code to the container
COPY . .

# Expose the port that Streamlit listens on (default is 8501)
EXPOSE 8501

# Command to run the Streamlit app.
# The "--server.fileWatcherType none" disables the file watcher to avoid torch warnings.
CMD ["streamlit", "run", "test.py", "--server.fileWatcherType", "none", "--server.address", "0.0.0.0"]
