FROM public.ecr.aws/lambda/python:3.8

COPY main.py ${LAMBDA_TASK_ROOT}
COPY requirements.txt ${LAMBDA_TASK_ROOT}
COPY imgs /tmp/imgs
COPY templates ${LAMBDA_RUNTIME_DIR}/templates
COPY model /opt/ml/model

RUN python3.8 -m pip install -r requirements.txt -t "${LAMBDA_TASK_ROOT}"

CMD ["main.handler"]
