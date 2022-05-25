FROM pytorch/pytorch:1.8.1-cuda11.1-cudnn8-runtime
RUN apt-get update && apt-get upgrade -y
RUN apt-get install git libglfw3-dev libgles2-mesa-dev build-essential -y
WORKDIR /workdir
COPY requirements.txt .
RUN pip3 install -r requirements.txt
ENV MUJOCO_GL="egl"
COPY . .