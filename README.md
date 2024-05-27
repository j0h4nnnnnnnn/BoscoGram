![Ups](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSSVcwVkLQdA8dF10hNxsJVA_cLgdyPQJH7POguN9hFuA&s)

# BoscoGram

## Descripción

Este proyecto es una aplicación web desarrollada con Flask que permite a los usuarios aplicar diversos filtros a imágenes utilizando la potencia de procesamiento de la GPU a través de PyCUDA. La aplicación incluye funcionalidades de registro, inicio de sesión, subida de imágenes y galerías de imágenes públicas y privadas.

## Funcionalidades

- Registro y autenticación de usuarios con.
- Subida y procesamiento de imágenes con filtros como emboss, logo, desenfoque, entre otros.
- Galerías de imágenes públicas y privadas.
- API para el consumo con clientes externos.

## Requisitos

- Docker
- Python
- NVIDIA GPU con soporte CUDA (revisar la versión instalada)
- Controladores CUDA instalados en el host
- Visual Studio Code
- Android Studio

## Instalación host

1. Clona el repositorio:
   ```bash
   git clone https://github.com/j0h4nnnnnnnn/BoscoGram.git
   cd BoscoGram

2. Construye y levanta los contenedores con Docker Compose:
   ```bash
   docker-compose build
   docker-compose up

3. Desactiva el Firewall de tu equipo.

4. Accede a la aplicación en
   ```bash
   http://localhost:5000

## Instalación apk móvil
  
1. dirigete a la carpeta
   ```bash
   BoscoGram/apk/BoscoGram.apk
  
2. Comparte la aplicación a tu equipo móvil.

3. Concede permiso para instalar "origenes desconocidos"
   
4. Instala y abre la aplicación
   
5. accede a la terminal del equipo host y obten la dirección IP en Windows:
  ```bash
   ipconfig
  ```

6. Coloca la IP en la aplicación.

## Funcionalidades

- **Post**: Al iniciar la aplicación en el móvil o abrir la dirección [http://(IP_HOST):5000](http://(IP_HOST):5000) desde otro equipo, se presentarán los post que han sido publicados.

- **Registrarse**: Ve a la sección de registrarse para crear una cuenta con su respectivo usuario y contraseña.

- **Iniciar Sesión**: Inicia sesión con las credenciales creadas.

- **Cargar Imagen y Aplicar Filtros**: 
  - En la aplicación móvil, selecciona el botón `+` para agregar una foto.
  - En la página web, al iniciar sesión serás redirigido por defecto a la sección donde puedes cargar y aplicar los diferentes filtros.
  - También se da la opción de manipular el tamaño de la máscara.
  - Luego, selecciona "aplicar filtro" para ver los resultados y "publicar" en caso de desearlo.

- **Galería Personal**: Se observa las fotos del mismo usuario, donde se pueden eliminar las fotos que desees.


## Resultados

- La aplicación permite a los usuarios subir imágenes y aplicar filtros de manera eficiente utilizando la GPU.
- Se logró una mejora significativa en el tiempo de procesamiento de imágenes gracias al uso de PyCUDA.
- La aplicación es accesible a través de una interfaz web y también expone una API para integración con otros clientes.

## Recomendaciones
El archivo 'docker-compose.yml' y el Dockerfile están configurados para usar una base de datos PostgreSQL y una imagen de CUDA. Asegúrate de ajustar las variables de entorno según tus necesidades,
la sección más importante del 'Dockerfile' es:
   ```bash
   FROM nvidia/cuda:12.2.0-devel-ubuntu22.04
  ```
Puesto que esta debe coincidir con la versión que se tiene instalada en el equipo host.
Además, es importante verificar que tanto el host como el dispositivo móvil o cualquier otro equipo desde el que se consuma la aplicación estén en la misma red para asegurar la conectividad.


## Conclusiones

Este proyecto demuestra cómo se puede utilizar la potencia de la GPU para acelerar el procesamiento de imágenes en una aplicación web. La combinación de Flask, PyCUDA además, del uso de Docker facilita enormemente el despliegue y la portabilidad de la aplicación, asegurando una configuración consistente en diferentes entornos. 


## Autores

- Paulina Azuero
- John Sanmartín


