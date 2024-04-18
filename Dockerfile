# This is the base image we build our image on
FROM tnodecode/mmcv

# Get root priviledges
USER root

# Install necessary CV libraries
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 -y

# Copy the reuirements txt file
COPY ./requirements ./requirements

# Install the necessary python libraries
RUN pip install -r requirements/build.txt
RUN pip install -r requirements/optional.txt
RUN pip install -r requirements/runtime.txt
RUN pip install fairscale timm

# Copy the code of the root directory of this repository into the /app directory of the docker image
COPY ./ ./

# Command to run the app
CMD uvicorn api:app --host 0.0.0.0 --port 80