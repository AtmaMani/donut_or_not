FROM public.ecr.aws/lambda/python:3.8

COPY main.py ${LAMBDA_TASK_ROOT}
COPY requirements3.txt ${LAMBDA_TASK_ROOT}
COPY templates ${LAMBDA_RUNTIME_DIR}/templates
# COPY models ${LAMBDA_RUNTIME_DIR}/models
COPY models ${LAMBDA_TASK_ROOT}/models

# RUN python3.8 -m pip install http://download.pytorch.org/whl/cpu/torch-1.9.0%2Bcpu-cp38-cp38-linux_x86_64.whl -t "${LAMBDA_TASK_ROOT}"
RUN python3.8 -m pip install -r requirements3.txt -t "${LAMBDA_TASK_ROOT}"
RUN python3.8 -m pip install --no-deps fastai==1.0.61 -t "${LAMBDA_TASK_ROOT}"

CMD ["main.handler"]
