import os
from flask import Flask, render_template, request, jsonify, redirect, url_for, session
from flask_sqlalchemy import SQLAlchemy
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity, set_access_cookies, unset_jwt_cookies
from flask_cors import CORS
import time
import numpy as np
from PIL import Image as PILImage
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'super-secret')  # Cambia esto en producción
#CORS(app, supports_credentials=True)


# Configuración de la base de datos PostgreSQL
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('SQLALCHEMY_DATABASE_URI', 'postgresql://postgres:postgres@db:5432/flask_db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['JWT_SECRET_KEY'] = os.getenv('JWT_SECRET_KEY', 'super-secret')  # Cambia esto en producción
app.config['JWT_TOKEN_LOCATION'] = ['cookies', 'headers']
app.config['JWT_ACCESS_COOKIE_PATH'] = '/'
app.config['JWT_COOKIE_CSRF_PROTECT'] = False

db = SQLAlchemy(app)
jwt = JWTManager(app)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)
    images = db.relationship('Image', backref='user', lazy=True)

class Image(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(120), nullable=False)
    filter_type = db.Column(db.String(50), nullable=False)
    processing_time = db.Column(db.Float, nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, 'static')

def initialize_cuda_context():
    try:
        cuda.init()
        print("Inicio bien")
    except cuda.LogicError as e:
        print("Error initializing CUDA context:", e)

def convert_to_array(image):
    ancho, alto = image.size
    datos_imagen = np.array(image, dtype=np.uint8)
    return datos_imagen, ancho, alto

def create_heart_mask(height, width ):
    x = np.linspace(-1.5, 1.5, width) * 5
    y = np.linspace(1.5, -1.5, height) * 10  # Invertir el rango de y para voltear el corazón
    x, y = np.meshgrid(x, y)
    
    # Usar una forma común de la ecuación de corazón en coordenadas cartesianas
    mask = ((x**2 + y**2 - 1)**3 - x**2 * y**3) <= 0
    
    return mask.astype(np.uint8) * 255  # Convertir a valores de 0 y 255

def apply_effects(input_image_path, heart_image_path, output_image_path, block_size, num_hearts):
    # Cargar la imagen principal y la imagen de corazones
    main_img = Image.open(input_image_path)
    heart_img = Image.open(heart_image_path)
    main_np = np.array(main_img).astype(np.uint8)
    heart_np = np.array(heart_img).astype(np.uint8)
    heart_height, heart_width, _ = heart_np.shape

    # Dimensiones de la imagen principal
    height, width, _ = main_np.shape
height, width = 1500, 1500  # Dimensiones de la máscara
#scale = 10  # Ajusta este valor para cambiar el tamaño del corazón (0.5 hará que el corazón sea más pequeño)
mask = create_heart_mask(height, width)

kernel_code_template = """
#define N {}
__global__ void embossFilterKernel(unsigned char* input_img, unsigned char* output_img, int width, int height) {{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {{
        float kernel[N][N];
        int kernel_radius = (N - 1) / 2;

        // Inicializar el kernel con ceros
        for (int i = 0; i < N; i++) {{
            for (int j = 0; j < N; j++) {{
                kernel[i][j] = 0;
            }}
        }}

        // Configurar valores del kernel
        for (int i = 0; i < N; i++) {{
            for (int j = 0; j < N; j++) {{
                if (i == kernel_radius || j == kernel_radius) {{
                    kernel[i][j] = 0;  // Centrales a cero
                }} else {{
                    int dist = kernel_radius - abs(i - kernel_radius);
                    kernel[i][j] = dist;

                    // Ajustar signos para las diagonales
                    if ((i < kernel_radius && j < kernel_radius) || (i > kernel_radius && j > kernel_radius)) {{
                        kernel[i][j] = -dist;
                    }}
                }}
            }}
        }}

        // Aplicar el kernel de emboss a la imagen
        for (int c = 0; c < 3; ++c) {{
            float sum = 0.0f;
            for (int k = -kernel_radius; k <= kernel_radius; k++) {{
                for (int l = -kernel_radius; l <= kernel_radius; l++) {{
                    int neighbor_x = max(0, min(x + k, width - 1));
                    int neighbor_y = max(0, min(y + l, height - 1));
                    int idx = (neighbor_y * width + neighbor_x) * 3;
                    float weight = kernel[k + kernel_radius][l + kernel_radius];
                    sum += weight * (float)input_img[idx + c];
                }}
            }}
            float original_value = (float)input_img[(y * width + x) * 3 + c];
            float mixed_value = 0.5f * original_value + 0.5f * (sum + 128.0f);
            output_img[(y * width + x) * 3 + c] = (unsigned char)fminf(fmaxf(mixed_value, 0.0f), 255.0f);
        }}
    }}
}}





__global__ void applyLogoKernel(unsigned char* input_img, unsigned char* output_img, unsigned char* logo_img, int width, int height, int logo_width, int logo_height) {{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.x + threadIdx.y;

    // Calcular la posición del logo en la parte superior izquierda
    if (x < logo_width && y < logo_height) {{
        int logo_x = x;
        int logo_y = y;

        // El logo solo tiene un canal en escala de grises
        float logo_value = (float)logo_img[logo_y * logo_width + logo_x];
        float factor = 0.2;  // Factor de mezcla para el logo, ajusta este valor según sea necesario

        for (int c = 0; c < 3; ++c) {{
            float input_value = (float)input_img[(y * width + x) * 3 + c];
            float mixed_value = (1.0f - factor) * input_value + factor * logo_value;
            output_img[(y * width + x) * 3 + c] = (unsigned char)fminf(fmaxf(mixed_value, 0.0f), 255.0f);
        }}
    }} else {{
        for (int c = 0; c < 3; ++c) {{
            output_img[(y * width + x) * 3 + c] = input_img[(y * width + x) * 3 + c];
        }}
    }}
}}


__global__ void horizontalBlurKernel(unsigned char* input_img, unsigned char* output_img, int width, int height) {{
    float filter[N];
    float value = 1.0f / N;
    for (int i = 0; i < N; i++) {{
        filter[i] = value;
    }}

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {{
        int pixelPos = y * width * 3 + x * 3;
        float sumR = 0.0;
        float sumG = 0.0;
        float sumB = 0.0;

        // Aplicar el filtro de desenfoque de movimiento
        for (int k = -N / 2; k <= N / 2; k++) {{
            int neighbor_x = min(max(x + k, 0), width - 1);
            int neighborPos = y * width * 3 + neighbor_x * 3;

            sumR += input_img[neighborPos] * filter[k + N / 2];
            sumG += input_img[neighborPos + 1] * filter[k + N / 2];
            sumB += input_img[neighborPos + 2] * filter[k + N / 2];
        }}

        output_img[pixelPos] = min(max(int(sumR), 0), 255);
        output_img[pixelPos + 1] = min(max(int(sumG), 0), 255);
        output_img[pixelPos + 2] = min(max(int(sumB), 0), 255);
    }}
}}



__global__ void sharpenKernel(unsigned char* input_img, unsigned char* output_img, int width, int height) {{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {{
        int kernel_radius = (N-1) / 2;
        float kernel[N][N];

        // Inicializar el kernel con -1
        for (int i = 0; i < N; i++) {{
            for (int j = 0; j < N; j++) {{
                kernel[i][j] = -1.0;
            }}
        }}

        // Configurar el valor central y sus vecinos superior e inferior directos
        kernel[kernel_radius][kernel_radius] =(-N*N)+1;  // Centro
        if (kernel_radius > 0) {{  // Solo si hay espacio arriba y abajo
            kernel[kernel_radius-1][kernel_radius] = 1.0;  // Arriba del centro
            kernel[kernel_radius+1][kernel_radius] = 1.0;  // Abajo del centro
        }}

        float sumR = 0.0, sumG = 0.0, sumB = 0.0;
        for (int k = -kernel_radius; k <= kernel_radius; k++) {{
            for (int l = -kernel_radius; l <= kernel_radius; l++) {{
                int neighbor_x = max(0, min(x + k, width - 1));
                int neighbor_y = max(0, min(y + l, height - 1));
                int idx = (neighbor_y * width + neighbor_x) * 3;
                float weight = kernel[kernel_radius + k][kernel_radius + l];
                sumR += weight * input_img[idx];
                sumG += weight * input_img[idx + 1];
                sumB += weight * input_img[idx + 2];
            }}
        }}

        int idx = (y * width + x) * 3;
        output_img[idx] = min(max(int(sumR), 0), 255);
        output_img[idx + 1] = min(max(int(sumG), 0), 255);
        output_img[idx + 2] = min(max(int(sumB), 0), 255);
    }}
}}


__global__ void contrastEnhancementKernel(unsigned char* input_img, unsigned char* output_img, int width, int height) {{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {{
        int kernel_radius = (N-1) / 2;
        float contrast_kernel[N][N]; // Asegúrate de que 21 es el máximo valor de N esperado.

        for (int i = 0; i < N; i++) {{
            for (int j = 0; j < N; j++) {{
                contrast_kernel[i][j] = -1;
            }}
        }}

        contrast_kernel[kernel_radius][kernel_radius] = (N * N) ;

        for (int c = 0; c < 3; ++c) {{
            float sum = 0.0f;
            for (int k = -kernel_radius; k <= kernel_radius; k++) {{
                for (int l = -kernel_radius; l <= kernel_radius; l++) {{
                    int neighbor_x = max(0, min(x + k, width - 1));
                    int neighbor_y = max(0, min(y + l, height - 1));
                    float weight = contrast_kernel[kernel_radius + k][kernel_radius + l];
                    sum += weight * input_img[(neighbor_y * width + neighbor_x) * 3 + c];
                }}
            }}
            output_img[(y * width + x) * 3 + c] = min(max(int(sum), 0), 255);
        }}
    }}
}}

__global__ void logFilterKernel(unsigned char* input_img, unsigned char* output_img, int width, int height) {{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int kernel_radius = (N-1) / 2;
    const float sigma = (N-1) / 2 ;  // Valor fijo para sigma

    if (x >= kernel_radius && x < width - kernel_radius && y >= kernel_radius && y < height - kernel_radius) {{
        // Definir y calcular el kernel LoG
        float kernel[N][N];
        float sum_kernel = 0;
        for (int i = -kernel_radius; i <= kernel_radius; i++) {{
            for (int j = -kernel_radius; j <= kernel_radius; j++) {{
                float x = i, y = j;
                float g = expf(-(x*x + y*y) / (2 * sigma * sigma));
                float laplacian = ((x*x + y*y - 2 * sigma * sigma) / (sigma * sigma * sigma * sigma)) * g;
                kernel[i + kernel_radius][j + kernel_radius] = laplacian;
                sum_kernel += laplacian;
            }}
        }}

        // Normalizar el kernel para asegurar que la suma sea 0
        float mean = sum_kernel / (N * N);
        for (int i = 0; i < N; i++) {{
            for (int j = 0; j < N; j++) {{
                kernel[i][j] -= mean;
            }}
        }}

        // Aplicar el kernel LoG a cada canal de color de la imagen
        for (int c = 0; c < 3; ++c) {{  // Loop sobre los canales de color
            float sum = 0;
            for (int i = -kernel_radius; i <= kernel_radius; i++) {{
                for (int j = -kernel_radius; j <= kernel_radius; j++) {{
                    int idx = ((y + j) * width + (x + i)) * 3 + c;
                    sum += input_img[idx] * kernel[kernel_radius + i][kernel_radius + j];
                }}
            }}
            int idx = (y * width + x) * 3 + c;
            output_img[idx] = min(max(int(sum + 128.0), 0), 255);  // Ajuste para mantener el rango de color
        }}
    }}
}}

"""

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        data = request.form
        username = data.get('username')
        password = data.get('password')
        
        if User.query.filter_by(username=username).first():
            return render_template('register.html', message='User already exists')
        
        new_user = User(username=username, password=password)
        db.session.add(new_user)
        db.session.commit()
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        data = request.form
        username = data.get('username')
        password = data.get('password')
        
        user = User.query.filter_by(username=username).first()
        if not user or user.password != password:
            return render_template('login.html', message='Invalid credentials')
        
        access_token = create_access_token(identity=user.id)
        response = redirect(url_for('upload_image'))
        set_access_cookies(response, access_token)
        session['user_id'] = user.id
        return response
    return render_template('login.html')

@app.route('/logout')
def logout():
    response = redirect(url_for('index'))
    unset_jwt_cookies(response)
    session.pop('user_id', None)
    return response

@app.route('/upload', methods=['GET', 'POST'])
@jwt_required()
def upload_image():
    user_id = get_jwt_identity()
    user = User.query.get(user_id)

    if request.method == 'POST' and 'apply_filter' in request.form:
        if 'file' not in request.files:
            return render_template('upload.html', error='No file part', username=user.username)

        file = request.files['file']
        if file.filename == '':
            return render_template('upload.html', error='No selected file', username=user.username)

        # Verificar el tipo de archivo
        if not allowed_file(file.filename):
            return render_template('upload.html', error='Tipo de archivo no permitido', username=user.username)

        filter_type = request.form['filterType']
        N = int(request.form['sobelSize'])

        logo_path = os.path.join(STATIC_DIR, 'logo.png')
        logo_img = PILImage.open(logo_path).convert('L')
        logo_ancho, logo_alto = logo_img.size

        if filter_type == 'logo':
            img = PILImage.open(file).resize((logo_ancho, logo_alto))
        else:
            img = PILImage.open(file)

        datos_imagen, ancho, alto = convert_to_array(img)
        logo_array, logo_ancho, logo_alto = convert_to_array(logo_img)

        try:
            cuda.init()
            device = cuda.Device(0)
            ctx = device.make_context()

            datos_imagen_gpu = cuda.mem_alloc(datos_imagen.nbytes)
            resultado_gpu = cuda.mem_alloc(datos_imagen.nbytes)
            logo_array_gpu = cuda.mem_alloc(logo_array.nbytes)
            heart_mask_gpu = None

            cuda.memcpy_htod(datos_imagen_gpu, datos_imagen)
            cuda.memcpy_htod(logo_array_gpu, logo_array)

            kernel_code = kernel_code_template.format(N)
            mod = SourceModule(kernel_code)

            bloques = (ancho // 16 + 1, alto // 16 + 1, 1)
            hilos = (16, 16, 1)

            if filter_type == 'emboss':
                print("Filtro Emboss-GPU")
                embossFilterKernel = mod.get_function("embossFilterKernel")
                start_time = time.time()
                embossFilterKernel(datos_imagen_gpu, resultado_gpu, np.int32(ancho), np.int32(alto), block=hilos, grid=bloques)
                end_time = time.time()
            elif filter_type == 'warm':
                print("Filtro Warm-GPU")
                warmFilterKernel = mod.get_function("warmFilterKernel")
                start_time = time.time()
                warmFilterKernel(datos_imagen_gpu, resultado_gpu, np.int32(ancho), np.int32(alto), block=hilos, grid=bloques)
                end_time = time.time()
            elif filter_type == 'logo':
                print("Filtro Logo-GPU")
                applyLogoKernel = mod.get_function("applyLogoKernel")
                start_time = time.time()
                applyLogoKernel(datos_imagen_gpu, resultado_gpu, logo_array_gpu, np.int32(ancho), np.int32(alto), np.int32(logo_ancho), np.int32(logo_alto), block=hilos, grid=bloques)
                end_time = time.time()
            elif filter_type == 'desenfoque':
                print("Filtro Desenfoque-GPU")
                horizontalBlurKernel = mod.get_function("horizontalBlurKernel")
                start_time = time.time()
                horizontalBlurKernel(datos_imagen_gpu, resultado_gpu, np.int32(ancho), np.int32(alto), block=hilos, grid=bloques)
                end_time = time.time()
            elif filter_type == 'corazon':
                print("Filtro Corazon-GPU")
                heart_mask = create_heart_mask(alto, ancho, 0.5)  # Crear la máscara del corazón
                heart_mask_gpu = cuda.mem_alloc(heart_mask.nbytes)
                cuda.memcpy_htod(heart_mask_gpu, heart_mask)

                applyHeartMaskKernel = mod.get_function("applyHeartMask")
                start_time = time.time()
                applyHeartMaskKernel(datos_imagen_gpu, resultado_gpu, heart_mask_gpu, np.int32(ancho), np.int32(alto), block=hilos, grid=bloques)
                end_time = time.time()

            elapsed_time = end_time - start_time
            elapsed_time = round(elapsed_time, 5)
            print(f"Tiempo transcurrido: {elapsed_time:.5f} segundos")

            resultado_cpu = np.empty_like(datos_imagen)
            cuda.memcpy_dtoh(resultado_cpu, resultado_gpu)

            processed_image_path = os.path.join(STATIC_DIR, f'processed_image_{int(time.time())}.png')  # Cambiado a .png para mantener la transparencia si es necesario
            processed_image = PILImage.fromarray(resultado_cpu)
            processed_image.save(processed_image_path)

            ctx.pop()
        except Exception as e:
            print("Error:", e)
        finally:
            if 'datos_imagen_gpu' in locals():
                datos_imagen_gpu.free()
            if 'resultado_gpu' in locals():
                resultado_gpu.free()
            if 'logo_array_gpu' in locals():
                logo_array_gpu.free()
            if 'heart_mask_gpu' in locals() and heart_mask_gpu is not None:
                heart_mask_gpu.free()

        return render_template('upload.html', image=os.path.basename(processed_image_path), elapsed_time=elapsed_time, filterType=filter_type, numThreads=hilos, numBloques=bloques, mascara=N, username=user.username)

    return render_template('upload.html', username=user.username)

@app.route('/publish', methods=['POST'])
@jwt_required()
def publish_image():
    user_id = get_jwt_identity()
    user = User.query.get(user_id)
    image_url = request.form['image_url']

    if image_url:
        new_image = Image(filename=image_url, filter_type=request.form['filterType'], processing_time=float(request.form['elapsed_time']), user=user)
        db.session.add(new_image)
        db.session.commit()

    return redirect(url_for('gallery'))

@app.route('/gallery', methods=['GET'])
@jwt_required()
def gallery():
    user_id = get_jwt_identity()
    user = User.query.get(user_id)
    images = Image.query.filter_by(user_id=user_id).all()
    return render_template('gallery.html', images=images, username=user.username)

@app.route('/public_gallery', methods=['GET'])
def public_gallery():
    images = Image.query.all()
    return render_template('public_gallery.html', images=images)

@app.route('/delete_image', methods=['POST'])
@jwt_required()
def delete_image():
    user_id = get_jwt_identity()
    image_id = request.form['image_id']
    image = Image.query.filter_by(id=image_id, user_id=user_id).first()
    
    if image:
        try:
            os.remove(os.path.join(STATIC_DIR, image.filename))
        except Exception as e:
            print(f"Error eliminando el archivo: {e}")
        
        db.session.delete(image)
        db.session.commit()
    
    return redirect(url_for('gallery'))

@app.route('/', methods=['GET'])
def index():
    images = Image.query.all()
    return render_template('index.html', images=images)

@app.route('/load_images', methods=['GET'])
def load_images():
    page = request.args.get('page', 1, type=int)
    per_page = 10
    images = Image.query.paginate(page, per_page, False).items

    image_data = []
    for image in images:
        image_data.append({
            'url': url_for('static', filename=image.filename),
            'username': image.user.username
        })

    return jsonify(images=image_data)

def allowed_file(filename):
    allowed_extensions = {'png', 'jpg', 'jpeg', 'gif'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

@app.route('/api/register', methods=['POST'])
def api_register():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    
    if User.query.filter_by(username=username).first():
        return jsonify({"message": "User already exists"}), 400
    
    new_user = User(username=username, password=password)
    db.session.add(new_user)
    db.session.commit()
    return jsonify({"message": "User created successfully"}), 201

@app.route('/api/login', methods=['POST'])
def api_login():
    if request.method == 'POST':
        if request.is_json:
            data = request.get_json()
            username = data.get('username')
            password = data.get('password')
        else:
            data = request.form
            username = data.get('username')
            password = data.get('password')

        user = User.query.filter_by(username=username).first()
        if not user or user.password != password:
            if request.is_json:
                return jsonify({'message': 'Invalid credentials'}), 401
            else:
                return render_template('login.html', message='Invalid credentials')

        access_token = create_access_token(identity=user.id)
        if request.is_json:
            response = jsonify({'message': 'Login successful', 'access_token': access_token})
        else:
            response = redirect(url_for('upload_image'))
        
        set_access_cookies(response, access_token)
        session['user_id'] = user.id
        return response

    return render_template('login.html')

@app.route('/api/upload', methods=['POST'])
@jwt_required()
def api_upload_image():
    user_id = get_jwt_identity()
    user = User.query.get(user_id)

    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Tipo de archivo no permitido'}), 400

    data = request.form if not request.is_json else request.get_json()
    filter_type = data['filterType']
    kernel_size = int(request.form.get('kernelSize', 5))  # Default to 5 if not provided


    logo_path = os.path.join(STATIC_DIR, 'logo.png')
    logo_img = PILImage.open(logo_path).convert('L')
    logo_ancho, logo_alto = logo_img.size

    if filter_type == 'logo':
        img = PILImage.open(file).resize((logo_ancho, logo_alto))
    else:
        img = PILImage.open(file)

    datos_imagen, ancho, alto = convert_to_array(img)
    logo_array, logo_ancho, logo_alto = convert_to_array(logo_img)

    try:
        cuda.init()
        device = cuda.Device(0)
        ctx = device.make_context()

        datos_imagen_gpu = cuda.mem_alloc(datos_imagen.nbytes)
        resultado_gpu = cuda.mem_alloc(datos_imagen.nbytes)
        logo_array_gpu = cuda.mem_alloc(logo_array.nbytes)
        heart_mask_gpu = None

        cuda.memcpy_htod(datos_imagen_gpu, datos_imagen)
        cuda.memcpy_htod(logo_array_gpu, logo_array)

        kernel_code = kernel_code_template.format(kernel_size)  # Cambiar el valor de N según sea necesario
        mod = SourceModule(kernel_code)

        bloques = (ancho // 16 + 1, alto // 16 + 1, 1)
        hilos = (16, 16, 1)

        if filter_type == 'emboss':
            embossFilterKernel = mod.get_function("embossFilterKernel")
            start_time = time.time()
            embossFilterKernel(datos_imagen_gpu, resultado_gpu, np.int32(ancho), np.int32(alto), block=hilos, grid=bloques)
            end_time = time.time()
        elif filter_type == 'logo':
            applyLogoKernel = mod.get_function("applyLogoKernel")
            start_time = time.time()
            applyLogoKernel(datos_imagen_gpu, resultado_gpu, logo_array_gpu, np.int32(ancho), np.int32(alto), np.int32(logo_ancho), np.int32(logo_alto), block=hilos, grid=bloques)
            end_time = time.time()
        elif filter_type == 'desenfoque':
            horizontalBlurKernel = mod.get_function("horizontalBlurKernel")
            start_time = time.time()
            horizontalBlurKernel(datos_imagen_gpu, resultado_gpu, np.int32(ancho), np.int32(alto), block=hilos, grid=bloques)
            end_time = time.time()
        elif filter_type == 'soooft':
            sharpenKernel = mod.get_function("sharpenKernel")
            start_time = time.time()
            sharpenKernel(datos_imagen_gpu, resultado_gpu, np.int32(ancho), np.int32(alto), block=hilos, grid=bloques)
            end_time = time.time()
        elif filter_type == 'soft':
            print("Aplicando filtro de contraste")
            contrastKernel = mod.get_function("contrastEnhancementKernel")
            start_time = time.time()
            contrastKernel(datos_imagen_gpu, resultado_gpu, np.int32(ancho), np.int32(alto), block=hilos, grid=bloques)
            end_time = time.time()
        elif filter_type == 'moon':
            print("Aplicando filtro de LOG")
            logFilterKernel = mod.get_function("logFilterKernel")
            start_time = time.time()
            logFilterKernel(datos_imagen_gpu, resultado_gpu, np.int32(ancho), np.int32(alto), block=hilos, grid=bloques)
            end_time = time.time()
        

        elapsed_time = end_time - start_time
        elapsed_time = round(elapsed_time, 5)

        resultado_cpu = np.empty_like(datos_imagen)
        cuda.memcpy_dtoh(resultado_cpu, resultado_gpu)

        processed_image_name = f'processed_image_{int(time.time())}.png'
        processed_image_path = os.path.join(STATIC_DIR, processed_image_name)
        processed_image = PILImage.fromarray(resultado_cpu)
        processed_image.save(processed_image_path)

        ctx.pop()
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        if 'datos_imagen_gpu' in locals():
            datos_imagen_gpu.free()
        if 'resultado_gpu' in locals():
            resultado_gpu.free()
        if 'logo_array_gpu' in locals():
            logo_array_gpu.free()
        if 'heart_mask_gpu' in locals() and heart_mask_gpu is not None:
            heart_mask_gpu.free()

    relative_image_url = processed_image_name
    print(f"Processed image saved at: {relative_image_url}")
    return jsonify({'image_url': relative_image_url, 'elapsed_time': elapsed_time})

@app.route('/api/publish', methods=['POST'])
@jwt_required()
def api_publish_image():
    user_id = get_jwt_identity()
    data = request.get_json()
    image_url = data.get('image_url')
    filter_type = data.get('filterType')
    elapsed_time = data.get('elapsed_time')

    if not image_url or not filter_type or not elapsed_time:
        return jsonify({'message': 'Faltan datos requeridos'}), 400

    try:
        new_image = Image(filename=image_url, filter_type=filter_type, processing_time=float(elapsed_time), user_id=user_id)
        db.session.add(new_image)
        db.session.commit()
        return jsonify({'message': 'Image published successfully'})
    except Exception as e:
        return jsonify({'message': f'Error al publicar la imagen: {e}'}), 500

@app.route('/api/gallery', methods=['GET'])
@jwt_required()
def api_gallery():
    user_id = get_jwt_identity()
    user = User.query.get(user_id)
    images = Image.query.filter_by(user_id=user_id).all()

    image_data = [
        {
            'id': image.id, 
            'filename': image.filename, 
            'filter_type': image.filter_type, 
            'processing_time': image.processing_time,
            'username': user.username
        } 
        for image in images
    ]
    return jsonify(image_data)

@app.route('/api/public_gallery', methods=['GET'])
def api_public_gallery():
    images = db.session.query(Image, User.username).join(User).all()
    images_list = [{
        'id': image.Image.id,
        'filename': image.Image.filename,
        'filter_type': image.Image.filter_type,
        'processing_time': image.Image.processing_time,
        'username': image.username,
        'url': url_for('static', filename=image.Image.filename, _external=True)
    } for image in images]
    return jsonify(images_list)

@app.route('/api/delete_image', methods=['POST'])
@jwt_required()
def api_delete_image():
    user_id = get_jwt_identity()
    data = request.get_json()
    image_id = data.get('image_id')

    image = Image.query.filter_by(id=image_id, user_id=user_id).first()
    if image:
        try:
            os.remove(os.path.join(STATIC_DIR, image.filename))
        except Exception as e:
            error_message = f"Error eliminando el archivo: {e}"
            return jsonify({'error': error_message}), 500
        
        db.session.delete(image)
        db.session.commit()
        return jsonify({'message': 'Image deleted successfully'})
    else:
        error_message = 'Image not found or unauthorized'
        return jsonify({'error': error_message}), 404


def allowed_file(filename):
    allowed_extensions = {'png', 'jpg', 'jpeg', 'gif'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(host="0.0.0.0", port=int("5000"), debug=True)
