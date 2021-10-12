# main app file

from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import json, os, shutil
from fastapi.templating import Jinja2Templates
from mangum import Mangum
import logging
from datetime import datetime
import base64

# Make logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create app, just like in Flask
app = FastAPI()

# region Get location of templates, models, static, imgs dirs from Lambda runtime.
try:
    templates_dir = os.path.join(os.environ.get("LAMBDA_RUNTIME_DIR"), "templates")
except:
    # Since this app can be run locally without Lambda on cloud or local Lambda runtime, make local paths
    templates_dir = "./templates"
templates = Jinja2Templates(directory=templates_dir)

try:
    models_dir1 = os.path.join(os.environ.get("LAMBDA_TASK_ROOT"), "models")
except:
    models_dir1 = "./models"

# try:  # don't need this as it does not work
#     static_dir = os.path.join(os.environ.get("LAMBDA_RUNTIME_DIR"), "static")
# except:
#     static_dir = "static"
# static_dir1 = f"/{static_dir}" if not static_dir.startswith("/") else static_dir  # adds preceeding / for route purposes

# Assign location to store uploaded images
imgs_dir = "/tmp/imgs"
if not os.path.exists(imgs_dir):
    os.mkdir(imgs_dir)
    logger.info(f'Made {imgs_dir}')
else:
    logger.info(f'{imgs_dir} exists, reusing it.')
    
# endregion

# Mount directories to the app to enable url_for in the Jinja2 template file
# app.mount(static_dir1, StaticFiles(directory=static_dir), name='static')
# app.mount('/tmp/imgs', StaticFiles(directory='/tmp/imgs'), name='imgs')

# copy model to writable dir
logger.info(f'Src models dir: {models_dir1}')
models_dir = "/tmp/models"
copy_flag = False
if not os.path.exists(models_dir):
    os.mkdir(models_dir)
    logger.info(f'Made {models_dir}')
else:
    logger.info(f'{models_dir} exists, reusing it.')

try:
    shutil.copyfile(f'{models_dir1}/export.pkl', f'{models_dir}/export.pkl')
    logger.info(f'Copied model using copyfile to {models_dir}')
    copy_flag = True
except Exception as copy_ex:
    logger.info(f'Cannot copy with copyfile. ' + str(copy_ex))

# Start to define REST APIs
@app.get('/hello')
def hello():
    logger.info('Logger /hello was called')
    print('/hello was called')
    return {'Hello': 'World. I am from within the FastAPI within Docker'}


@app.get('/listImgFiles')
async def list_uploaded_files():
    return os.listdir(imgs_dir)


@app.get('/getTimestamp')
async def get_timestamp():
    return datetime.now().strftime("%y%h%d_%H%M%S")


# Upload files
@app.post('/classifyImg', response_class=HTMLResponse)
async def upload_classify_img(request: Request, file: UploadFile = File(...)):
    logger.info("classifyImg is called")

    # Download image
    timestamp = await get_timestamp()
    file_name = f'{timestamp}.jpg'
    img_data = file.file.read()
    img_data_b64 = base64.b64encode(img_data)
    img_data_b64 = img_data_b64.decode()

    with open(f'{imgs_dir}/{file_name}', 'wb+') as fp:
        fp.write(img_data)
    logger.info('Wrote file to disk')

    # classify image
    img_class = classify_img(f'{imgs_dir}/{file_name}')
    
    return templates.TemplateResponse('response.html', {'request':request,
                                                        'output_filepath': img_data_b64,
                                                        'output_class': img_class['predicted_class'],
                                                        'output_probabilities': img_class['class_probabilities']})


# Classify images with DL
def classify_img(file:str) -> dict:
    from fastai.vision.image import open_image
    from fastai.basic_train import load_learner

    # load model
    learn = load_learner(models_dir)

    # open image file
    img = open_image(file)

    # predict
    pred_class, pred_idx, outputs = learn.predict(img)
    
    # compose output
    classes=['vada','bagel','donut']
    probabilities = [round(f,4) for f in outputs.tolist()]

    d = {'predicted_class': pred_class.obj,
        'class_probabilities': dict(list(zip(classes, probabilities)))}

    return d


@app.get('/', response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse('index.html', {'request':request})


handler = Mangum(app)
