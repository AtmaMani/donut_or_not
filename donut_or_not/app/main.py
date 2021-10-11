# main app file

from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import json, os, shutil
from fastapi.templating import Jinja2Templates
from mangum import Mangum
import logging
from datetime import datetime


# Create app, just like in Flask
app = FastAPI()
app.mount('/static', StaticFiles(directory='static'), name='static')
app.mount('/tmp/imgs', StaticFiles(directory='/tmp/imgs'), name='imgs')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

try:
    templates_dir = os.path.join(os.environ.get("LAMBDA_RUNTIME_DIR"), "templates")
except:
    templates_dir = "./templates"

try:
    models_dir1 = os.path.join(os.environ.get("LAMBDA_TASK_ROOT"), "models")
except:
    models_dir1 = "./models"

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

if not copy_flag:
    try:
        shutil.copy(f'{models_dir1}/export.pkl', f'{models_dir}/export.pkl')
        logger.info(f'Copied model using copy to {models_dir}')
        copy_flag = True
    except Exception as copy_ex:
        logger.info(f'Cannot copy with copy. ' + str(copy_ex))

if not copy_flag:
    try:
        shutil.copy2(f'{models_dir1}/export.pkl', f'{models_dir}/export.pkl')
        logger.info(f'Copied model using copy2 to {models_dir}')
        copy_flag = True
    except Exception as copy_ex:
        logger.info(f'Cannot copy with copy2 ' + str(copy_ex))
    

imgs_dir = "/tmp/imgs"
templates = Jinja2Templates(directory=templates_dir)


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
    print("UploadImg is called- within func")

    # Create download folder
    if not os.path.exists(imgs_dir):
        os.mkdir(imgs_dir)
        logger.info(f'Made {imgs_dir}')
        print(f'Made {imgs_dir}')
    else:
        logger.info(f'{imgs_dir} exists, reusing it.')
        print(f'{imgs_dir} exists, reusing it.')

    # Download image
    timestamp = await get_timestamp()
    file_name = f'{timestamp}.jpg'

    with open(f'{imgs_dir}/{file_name}', 'wb+') as fp:
        fp.write(file.file.read())
    logger.info('Wrote file to disk')
    print('Wrote file to disk')

    # classify image
    img_class = classify_img(f'{imgs_dir}/{file_name}')
    
    # return {'Status':'Uploaded',
    #         'uploaded_files': await list_uploaded_files(),
    #         'Image class: ': img_class
    #         }
    return templates.TemplateResponse('response.html', {'request':request,
                                                        'output_filepath': file_name,
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
