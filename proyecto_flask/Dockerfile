# Usar una imagen base de nvidia/cuda para soporte de GPU
FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    build-essential \
    libssl-dev \
    libffi-dev \
    libpq-dev

# Copiar el código de la aplicación
COPY . /app
WORKDIR /app

# Instalar dependencias de Python
RUN pip3 install flask
RUN pip3 install flask_sqlalchemy
RUN pip3 install flask_jwt_extended
RUN pip3 install numpy
RUN pip3 install pillow
RUN pip3 install pycuda
RUN pip3 install psycopg2-binary
RUN pip3 install flask-cors

# Exponer el puerto 5000
EXPOSE 5000

# Comando para ejecutar la aplicación
CMD python3 ./app.py
