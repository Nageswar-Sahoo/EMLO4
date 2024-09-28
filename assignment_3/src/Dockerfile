FROM python:3.9.20

WORKDIR /workspace

COPY requirements.txt requirements.txt

RUN pip3 install -r requirements.txt

COPY * /workspace/

CMD ["python3", "train.py"] 