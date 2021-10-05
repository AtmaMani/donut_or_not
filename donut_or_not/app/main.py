# main app file

from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse
import json, os
from fastapi.templating import Jinja2Templates
from mangum import Mangum
import logging
from datetime import datetime


# Create app, just like in Flask
app = FastAPI()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

try:
    templates_dir = os.path.join(os.environ.get("LAMBDA_RUNTIME_DIR"), "templates")
except:
    templates_dir = "./templates"
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
@app.post('/classifyImg')
async def upload_classify_img(file: UploadFile = File(...)):
    logger.info("classifyImg is called")
    print("UploadImg is called- within func")
    if not os.path.exists(imgs_dir):
        os.mkdir(imgs_dir)
        logger.info(f'Made {imgs_dir}')
        print(f'Made {imgs_dir}')
    else:
        logger.info(f'{imgs_dir} exists, reusing it.')
        print(f'{imgs_dir} exists, reusing it.')

    timestamp = await get_timestamp()
    file_name = f'{timestamp}.jpg'

    with open(f'{imgs_dir}/{file_name}', 'wb+') as fp:
        fp.write(file.file.read())

    logger.info('Wrote file to disk')
    print('Wrote file to disk')
    return {'Status':'Uploaded',
            'uploaded_files': await list_uploaded_files()}


# Upload files
@app.post('/uploadImg2')
async def create_upload_img2(file: bytes = File(...)):
    print("UploadImg is called- within func2")
    if not os.path.exists(imgs_dir):
        os.mkdir(imgs_dir)
        print(f'Made {imgs_dir}')
    else:
        print(f'{imgs_dir} exists, reusing it.')

    file_list = os.listdir(imgs_dir)
    file_name = f'{len(file_list)+1}.jpg'

    # with open(f'{imgs_dir}/{file_name}', 'wb+') as fp:
    #     # fp.write(file.file.read())
    #     fp.write(file.)

    print('Wrote file to disk')
    # return {'Status':'Uploaded',
    #         'uploaded_files': await list_uploaded_files()}
    return {'Status':'Upload stopped',
            'file_size':len(file)}

@app.get('/', response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse('index.html', {'request':request})

handler = Mangum(app)
