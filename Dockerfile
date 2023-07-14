# pull python base image
FROM python:3.10

# copy application files
ADD /bikeshare_model /bikeshare_model/

# specify working directory
WORKDIR /bikeshare_model

# update pip
RUN pip install --upgrade pip

# install dependencies
RUN pip install -r requirements.txt

# expose port for application
EXPOSE 8001

# start fastapi application
CMD ["python", "app/main.py"]
