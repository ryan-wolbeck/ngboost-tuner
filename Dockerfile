FROM python:3.7-buster

RUN pip install --upgrade pip

RUN apt-get update && apt-get install -y libsasl2-modules libsasl2-dev 

WORKDIR "/usr/src/app"

COPY requirements.txt .
RUN pip install -r ./requirements.txt

COPY setup.py .
COPY README.md .
RUN pip install .[all]
COPY . .

CMD ["tail", "-f", "/dev/null"]