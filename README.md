# Donut or Not
**About**: Simple project to run deep learning inferencing on AWS Lambda to distinguish donuts from bagels and vadas (South Indian savory dish).

## Layout of this repo:
Layout of this repo:

todo 

## Set up
The set up for this project consists of two parts. The first part here talks about the bits needed to push this model and inferencing function (built using [FastAPI](https://fastapi.tiangolo.com/) lib) to the cloud using AWS Lambdas. The second part talks about the bits needed to train the PyTorch - FastAI model. These two environments are different from each other. While they can be on the same machine (a GPU powered one), they don't have to necessarily be.

### Set up for AWS Lambda based DL inferencing
The inferencing would run on a Docker Image on the AWS Lambda FaaS platform. AWS recently (March 2020) announced support for Docker images for Lambdas. The documentation around this is slim. So we will use the [AWS SAM](https://docs.aws.amazon.com/serverless-application-model/latest/developerguide/serverless-sam-cli-using-build.html) CLI to give us a framework which we will flush out with the necessary code.

The AWS SAM CLI is used here to initialize a Python Docker Image based function. The CLI will create the necessary folders with `app.py`, a `Dockerfile`, a `template.yml` file for cloud formation (which is the mechanism to build this deployment stack on the cloud) among other misc files. It also gives tooling to invoke the function locally to allow the development cycle. Finally, it allows us to deploy the stack on the cloud by auto-generating the necessary S3 buckets (for storage), AWS ECR (Elastic Container Registry) entries to store the Docker image, the API gateway endpoints (for the client facing REST API) and gives us a public URL to invoke the function.

Thus, the prerequisites for this set up are

 * [AWS account - free tier](https://aws.amazon.com/free/?nc2=h_ql_pr_ft). For this project, just the free tier will do. The first account you create is the "Root administrator" account. I recommend creating another account (based on security best practices), perhaps with admin privileges, but enable programming or shell access. We will use this account for this project.
 * [AWS CLI](https://aws.amazon.com/cli/). After installation, you need to login to allow programmatic access. [See this help](https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-files.html).
 * [Docker desktop engine](https://www.docker.com/products/docker-desktop) to build your containers locally.
 * [AWS SAM CLI](https://docs.aws.amazon.com/serverless-application-model/latest/developerguide/serverless-sam-cli-install.html). This is different from the AWS CLI. The SAM CLI makes use of the AWS CLI internally.

#### Initialize with SAM
Use terminal to enter your project folder and then run `sam init`. SAM uses a guided approach to ask a series of questions and creates the necessary files as shown below. **Note**: If you are cloning this repo, you might get these files pre-made.

```cmd
(base) ➜  donut_or_not git:(main) sam init
Which template source would you like to use?
	1 - AWS Quick Start Templates
	2 - Custom Template Location
Choice: 1
What package type would you like to use?
	1 - Zip (artifact is a zip uploaded to S3)	
	2 - Image (artifact is an image uploaded to an ECR image repository)
Package type: 2

Which base image would you like to use?
	1 - amazon/nodejs14.x-base
	2 - amazon/nodejs12.x-base
	3 - amazon/nodejs10.x-base
	4 - amazon/python3.9-base
	5 - amazon/python3.8-base
	6 - amazon/python3.7-base
	7 - amazon/python3.6-base
	8 - amazon/python2.7-base
	9 - amazon/ruby2.7-base
	10 - amazon/ruby2.5-base
	11 - amazon/go1.x-base
	12 - amazon/java11-base
	13 - amazon/java8.al2-base
	14 - amazon/java8-base
	15 - amazon/dotnet5.0-base
	16 - amazon/dotnetcore3.1-base
	17 - amazon/dotnetcore2.1-base
Base image: 5

Project name [sam-app]: donut_or_not

Cloning from https://github.com/aws/aws-sam-cli-app-templates

AWS quick start application templates:
	1 - Hello World Lambda Image Example
	2 - PyTorch Machine Learning Inference API
	3 - Scikit-learn Machine Learning Inference API
	4 - Tensorflow Machine Learning Inference API
	5 - XGBoost Machine Learning Inference API
Template selection: 2

    -----------------------
    Generating application:
    -----------------------
    Name: donut_or_not
    Base Image: amazon/python3.8-base
    Dependency Manager: pip
    Output Directory: .

    Next steps can be found in the README file at ./donut_or_not/README.md
        

SAM CLI update available (1.33.0); (1.31.0 installed)
To download: https://docs.aws.amazon.com/serverless-application-model/latest/developerguide/serverless-sam-cli-install.html
```
This creates another folder by name `donut_or_not` in the current folder. There is an `app`, `events` folders, a `template.yml` and a `training.ipynb` file. For now, these files correspond to a simple MNIST example using PyTorch. We will delete or replace these files with ones needed for our workflow.

#### Create a virtual env for development
For the dev cycle, we need a faster build-deploy loop. For this, we will create a local virtual env.

```python
(base) ➜  donut_or_not git:(main) ✗ conda activate aws_lambda_default
(aws_lambda_default) ➜  donut_or_not git:(main) ✗ python -m venv donut_env
(aws_lambda_default) ➜  donut_or_not git:(main) ✗ . donut_env/bin/activate

# Start installing libs for the web server
(donut_env) (aws_lambda_default) ➜  donut_or_not git:(main) ✗ pip install fastapi
(donut_env) (aws_lambda_default) ➜  donut_or_not git:(main) ✗ pip install uvicorn[standard]
(donut_env) (aws_lambda_default) ➜  donut_or_not git:(main) ✗ pip install jinja2
(donut_env) (aws_lambda_default) ➜  donut_or_not git:(main) ✗ pip install mangum
(donut_env) (aws_lambda_default) ➜  donut_or_not git:(main) ✗ pip install aiofiles
(donut_env) (aws_lambda_default) ➜  donut_or_not git:(main) ✗ pip install python-multipart

# Install libs for DL inference
(donut_env) (aws_lambda_default) ➜  donut_or_not git:(main) ✗ pip install fastai==1.0.61

# Write the requirements.txt file
(donut_env) (aws_lambda_default) ➜  donut_or_not git:(main) ✗ pip freeze > ./donut_or_not/app/requirements.txt
```

## Building the DL application
See `donut_or_not_training.ipynb` notebook to understand how the model was trained. Toward the end, you can see how the model was exported to a pickle while, which we use here.

The `classify_img` function in `main.py` file shows how the learner is hydrated from the model and how the uploaded image is used for inference.

The `main.py` is written in such a way that it can be run locally (without even the Docker container env) using the `donut_env` using the same code. It would work just as well when run in the Docker container later when testing using SAM. To run it locally, from within the `app` folder, run 

```
(donut_env) (aws_lambda_default) ➜  app git:(main) ✗ uvicorn main:app --reload
INFO:     Will watch for changes in these directories: ['/.../donut_or_not/donut_or_not/app']
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     Started reloader process [7138] using watchgod
INFO:     Started server process [7140]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     127.0.0.1:54397 - "GET / HTTP/1.1" 200 OK
```
Once the app looks good, we can get on to the next stage, which is building the Docker image.

## Deploy to cloud
Now that the local env works, the next step is to **build** the Docker image. We use `sam build` command for this.

### Building the Docker image
The inference environments for DL apps like this one are finicky and hard to get right if we were to just go with requirements or environment files. Hence, the developer needs to take the onus of building the Docker image and deploying it on the server. The build instructions are maintained in the `app/Dockerfile` file. After much trail and error, I settled with the current mix of install steps split between the Dockerfile and ones specified in the `app/requirements3.txt` file. To build, you run 

```
(donut_env) (aws_lambda_default) ➜  donut_or_not git:(main) ✗ sam build          
Building codeuri: /Users/atma6951/Documents/code/pychakras/donut_or_not/donut_or_not runtime: None metadata: {'Dockerfile': 'Dockerfile', 'DockerContext': '/Users/atma6951/Documents/code/pychakras/donut_or_not/donut_or_not/app', 'DockerTag': 'python3.8-v1'} functions: ['DonutOrNotFunction']
Building image for DonutOrNotFunction function
Setting DockerBuildArgs: {} for DonutOrNotFunction function
Step 1/8 : FROM public.ecr.aws/lambda/python:3.8
 ---> 80342c69b467
Step 2/8 : COPY main.py ${LAMBDA_TASK_ROOT}
 ---> Using cache
 ---> 3ddf00ad5799
Step 3/8 : COPY requirements3.txt ${LAMBDA_TASK_ROOT}
 ---> 53661dd9063b
Step 4/8 : COPY templates ${LAMBDA_RUNTIME_DIR}/templates
 ---> b820476212a5
Step 5/8 : COPY models ${LAMBDA_RUNTIME_DIR}/models
 ---> cd972485a6af
Step 6/8 : RUN python3.8 -m pip install -r requirements3.txt -t "${LAMBDA_TASK_ROOT}"
 ---> Running in fa12b066d68c
Looking in links: https://download.pytorch.org/whl/torch_stable.html
Collecting torch==1.8.0+cpu
  Downloading https://download.pytorch.org/whl/cpu/torch-1.8.0%2Bcpu-cp38-cp38-linux_x86_64.whl (169.1 MB)
Collecting torchvision==0.9.0+cpu
  Downloading https://download.pytorch.org/whl/cpu/torchvision-0.9.0%2Bcpu-cp38-cp38-linux_x86_64.whl (13.3 MB)
Collecting aiofiles==0.7.0
  Downloading aiofiles-0.7.0-py3-none-any.whl (13 kB)

# Install truncated for brevity here

Installing collected packages: typing-extensions, six, numpy, urllib3, torch, tomli, starlette, soupsieve, regex, pytz, python-dateutil, pyparsing, pydantic, platformdirs, Pillow, pathspec, mypy-extensions, MarkupSafe, kiwisolver, idna, h11, cycler, click, charset-normalizer, certifi, asgiref, websockets, watchgod, uvloop, uvicorn, torchvision, scipy, requests, PyYAML, python-multipart, python-dotenv, pynvx, pandas, packaging, nvidia-ml-py3, numexpr, matplotlib, mangum, Jinja2, httptools, fastprogress, fastapi, black, beautifulsoup4, aiofiles
    Running setup.py install for python-multipart: started
    Running setup.py install for python-multipart: finished with status 'done'
    Running setup.py install for nvidia-ml-py3: started
    Running setup.py install for nvidia-ml-py3: finished with status 'done'
Successfully installed Jinja2-3.0.2 MarkupSafe-2.0.1 Pillow-8.3.2 PyYAML-5.4.1 aiofiles-0.7.0 asgiref-3.4.1 beautifulsoup4-4.10.0 black-21.9b0 certifi-2021.5.30 charset-normalizer-2.0.6 click-8.0.1 cycler-0.10.0 fastapi-0.68.1 fastprogress-1.0.0 h11-0.12.0 httptools-0.2.0 idna-3.2 kiwisolver-1.3.2 mangum-0.12.2 matplotlib-3.4.3 mypy-extensions-0.4.3 numexpr-2.7.3 numpy-1.21.2 nvidia-ml-py3-7.352.0 packaging-21.0 pandas-1.3.3 pathspec-0.9.0 platformdirs-2.4.0 pydantic-1.8.2 pynvx-1.0.0 pyparsing-2.4.7 python-dateutil-2.8.2 python-dotenv-0.19.0 python-multipart-0.0.5 pytz-2021.3 regex-2021.9.30 requests-2.26.0 scipy-1.7.1 six-1.16.0 soupsieve-2.2.1 starlette-0.14.2 tomli-1.2.1 torch-1.8.0+cpu torchvision-0.9.0+cpu typing-extensions-3.10.0.2 urllib3-1.26.7 uvicorn-0.15.0 uvloop-0.16.0 watchgod-0.7 websockets-10.0
WARNING: Running pip as root will break packages and permissions. You should install packages reliably by using venv: https://pip.pypa.io/warnings/venv
WARNING: You are using pip version 21.1.1; however, version 21.2.4 is available.
You should consider upgrading via the '/var/lang/bin/python3.8 -m pip install --upgrade pip' command.
 ---> e13ae5fb8fbf
Step 7/8 : RUN python3.8 -m pip install --no-deps fastai==1.0.61 -t "${LAMBDA_TASK_ROOT}"
 ---> Running in d1f6b0692068
Collecting fastai==1.0.61
  Downloading fastai-1.0.61-py3-none-any.whl (239 kB)
Installing collected packages: fastai
Successfully installed fastai-1.0.61
WARNING: Running pip as root will break packages and permissions. You should install packages reliably by using venv: https://pip.pypa.io/warnings/venv
WARNING: You are using pip version 21.1.1; however, version 21.2.4 is available.
You should consider upgrading via the '/var/lang/bin/python3.8 -m pip install --upgrade pip' command.
 ---> fa382927593b
Step 8/8 : CMD ["main.handler"]
 ---> Running in 5d3ee69f73a9
 ---> 0dd800d5144d
Successfully built 0dd800d5144d
Successfully tagged donutornotfunction:python3.8-v1

Build Succeeded

Built Artifacts  : .aws-sam/build
Built Template   : .aws-sam/build/template.yaml

Commands you can use next
=========================
[*] Invoke Function: sam local invoke
[*] Deploy: sam deploy --guided
```

### Test Docker image
You can test the image locally by running `sam local start-api` as shown below:

```
(donut_env) (aws_lambda_default) ➜  donut_or_not git:(main) ✗ sam local start-api
Mounting DonutOrNotFunction at http://127.0.0.1:3000/hello [GET]
Mounting DonutOrNotFunction at http://127.0.0.1:3000/ [GET]
Mounting DonutOrNotFunction at http://127.0.0.1:3000/getTimestamp [GET]
Mounting DonutOrNotFunction at http://127.0.0.1:3000/listImgFiles [GET]
Mounting DonutOrNotFunction at http://127.0.0.1:3000/classifyImg [POST]
You can now browse to the above endpoints to invoke your functions. You do not need to restart/reload SAM CLI while working on your functions, changes will be reflected instantly/automatically. You only need to restart SAM CLI if you update your AWS SAM template
2021-10-08 14:03:28  * Running on http://127.0.0.1:3000/ (Press CTRL+C to quit)
Invoking Container created from donutornotfunction:python3.8-v1
Building image.................
Skip pulling image and use local one: donutornotfunction:rapid-1.31.0.

START RequestId: 774ffad2-5327-41de-8d01-7977c56e4016 Version: $LATEST
END RequestId: 774ffad2-5327-41de-8d01-7977c56e4016
REPORT RequestId: 774ffad2-5327-41de-8d01-7977c56e4016	Init Duration: 0.11 ms	Duration: 491.71 ms	Billed Duration: 500 ms	Memory Size: 128 MB	Max Memory Used: 128 MB	
2021-10-08 14:03:34 127.0.0.1 - - [08/Oct/2021 14:03:34] "GET / HTTP/1.1" 200 -
Invoking Container created from donutornotfunction:python3.8-v1
Building image.................
Skip pulling image and use local one: donutornotfunction:rapid-1.31.0.
```

## Deploy application to the cloud
To deploy you run `sam deploy`. The first time, you can use a guided approach and run `sam deploy --guided`. In my case, I have multiple AWS IAM users. Hence I provide a profile that has the right privileges. During the deploy, SAM walks you through various questions. Most are safe to answer with a `y` or `Y` for yes. The process creates necessary S3 buckets, repositories in ECR, IAM users and appropriate API gateways are opened.

```
(base) ➜  donut_or_not git:(main) sam deploy --guided --profile atma_lambda

Configuring SAM deploy
======================

	Looking for config file [samconfig.toml] :  Not found

	Setting default arguments for 'sam deploy'
	=========================================
	Stack Name [sam-app]: donut-or-not
	AWS Region [us-west-2]: 
	#Shows you resources changes to be deployed and require a 'Y' to initiate deploy
	Confirm changes before deploy [y/N]: Y
	#SAM needs permission to be able to create roles to connect to the resources in your template
	Allow SAM CLI IAM role creation [Y/n]: Y
	DonutOrNotFunction may not have authorization defined, Is this okay? [y/N]: Y
	DonutOrNotFunction may not have authorization defined, Is this okay? [y/N]: Y
	DonutOrNotFunction may not have authorization defined, Is this okay? [y/N]: Y
	DonutOrNotFunction may not have authorization defined, Is this okay? [y/N]: Y
	DonutOrNotFunction may not have authorization defined, Is this okay? [y/N]: Y
	Save arguments to configuration file [Y/n]: Y
	SAM configuration file [samconfig.toml]: 
	SAM configuration environment [default]: 

	Looking for resources needed for deployment:
	 Managed S3 bucket: aws-sam-cli-managed-default-samclisourcebucket-e7w5utol4v72
	 A different default S3 bucket can be set in samconfig.toml
	 Image repositories: Not found.
	 #Managed repositories will be deleted when their functions are removed from the template and deployed
	 Create managed ECR repositories for all functions? [Y/n]: Y

	Saved arguments to config file
	Running 'sam deploy' for future deployments will use the parameters saved above.
	The above parameters can be changed by modifying samconfig.toml
	Learn more about samconfig.toml syntax at 
	https://docs.aws.amazon.com/serverless-application-model/latest/developerguide/serverless-sam-cli-config.html

The push refers to repository [939098422637.dkr.ecr.us-west-2.amazonaws.com/donutornot3159b213/donutornotfunction89144d83repo]

The push refers to repository [939098422637.dkr.ecr.us-west-2.amazonaws.com/donutornot3159b213/donutornotfunction89144d83repo]
e8463ca45505: Pushed 
c29f53611dcb: Pushed 
b1cb94c6d9d1: Pushed 
8030ec00e438: Pushed 
e6ef369ea50f: Pushed 
9b828246826b: Pushed 
a1f8e0568112: Pushed 
bcf453d1de13: Pushed 
f6ae2f36d5d7: Pushed 
5959c8f9752b: Pushed 
3e5452c20c48: Pushed 
9c4b6b04eac3: Pushed 
donutornotfunction-0dd800d5144d-python3.8-v1: digest: sha256:f6df036750adbe9a39e4486383fb3ea094558f3e210182523f3c2f276840d77d size: 2841


	Deploying with following values
	===============================
	Stack name                   : donut-or-not
	Region                       : us-west-2
	Confirm changeset            : True
	Deployment image repository  : 
                                       {
                                           "DonutOrNotFunction": "939098422637.dkr.ecr.us-west-2.amazonaws.com/donutornot3159b213/donutornotfunction89144d83repo"
                                       }
	Deployment s3 bucket         : aws-sam-cli-managed-default-samclisourcebucket-e7w5utol4v72
	Capabilities                 : ["CAPABILITY_IAM"]
	Parameter overrides          : {}
	Signing Profiles             : {}

Initiating deployment
=====================
Uploading to donut-or-not/4eb668d1f8cd57d339f709d83932edda.template  1635 / 1635  (100.00%)

Waiting for changeset to be created..

CloudFormation stack changeset
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Operation                                         LogicalResourceId                                 ResourceType                                      Replacement                                     
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
+ Add                                             DonutOrNotFunctionClassifyImgPermissionProd       AWS::Lambda::Permission                           N/A                                             
+ Add                                             DonutOrNotFunctionGenTimeStampPermissionProd      AWS::Lambda::Permission                           N/A                                             
+ Add                                             DonutOrNotFunctionHelloPermissionProd             AWS::Lambda::Permission                           N/A                                             
+ Add                                             DonutOrNotFunctionListImgFilesPermissionProd      AWS::Lambda::Permission                           N/A                                             
+ Add                                             DonutOrNotFunctionRole                            AWS::IAM::Role                                    N/A                                             
+ Add                                             DonutOrNotFunctionRootPermissionProd              AWS::Lambda::Permission                           N/A                                             
+ Add                                             DonutOrNotFunction                                AWS::Lambda::Function                             N/A                                             
+ Add                                             ServerlessRestApiDeployment53b600a676             AWS::ApiGateway::Deployment                       N/A                                             
+ Add                                             ServerlessRestApiProdStage                        AWS::ApiGateway::Stage                            N/A                                             
+ Add                                             ServerlessRestApi                                 AWS::ApiGateway::RestApi                          N/A                                             
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Changeset created successfully. arn:aws:cloudformation:us-west-2:939098422637:changeSet/samcli-deploy1633729019/6fc074ea-cd21-4ad6-bd01-195f7a631170


Previewing CloudFormation changeset before deployment
======================================================
Deploy this changeset? [y/N]: y

2021-10-08 15:05:38 - Waiting for stack create/update to complete

CloudFormation events from changeset
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
ResourceStatus                                    ResourceType                                      LogicalResourceId                                 ResourceStatusReason                            
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
CREATE_IN_PROGRESS                                AWS::IAM::Role                                    DonutOrNotFunctionRole                            Resource creation Initiated                     
CREATE_IN_PROGRESS                                AWS::IAM::Role                                    DonutOrNotFunctionRole                            -                                               
CREATE_COMPLETE                                   AWS::IAM::Role                                    DonutOrNotFunctionRole                            -                                               
CREATE_IN_PROGRESS                                AWS::Lambda::Function                             DonutOrNotFunction                                -                                               
CREATE_IN_PROGRESS                                AWS::Lambda::Function                             DonutOrNotFunction                                Resource creation Initiated                     
CREATE_COMPLETE                                   AWS::Lambda::Function                             DonutOrNotFunction                                -                                               
CREATE_IN_PROGRESS                                AWS::ApiGateway::RestApi                          ServerlessRestApi                                 -                                               
CREATE_IN_PROGRESS                                AWS::ApiGateway::RestApi                          ServerlessRestApi                                 Resource creation Initiated                     
CREATE_COMPLETE                                   AWS::ApiGateway::RestApi                          ServerlessRestApi                                 -                                               
CREATE_IN_PROGRESS                                AWS::Lambda::Permission                           DonutOrNotFunctionClassifyImgPermissionProd       -                                               
CREATE_IN_PROGRESS                                AWS::ApiGateway::Deployment                       ServerlessRestApiDeployment53b600a676             -                                               
CREATE_IN_PROGRESS                                AWS::Lambda::Permission                           DonutOrNotFunctionGenTimeStampPermissionProd      -                                               
CREATE_IN_PROGRESS                                AWS::ApiGateway::Deployment                       ServerlessRestApiDeployment53b600a676             Resource creation Initiated                     
CREATE_IN_PROGRESS                                AWS::Lambda::Permission                           DonutOrNotFunctionListImgFilesPermissionProd      -                                               
CREATE_IN_PROGRESS                                AWS::Lambda::Permission                           DonutOrNotFunctionRootPermissionProd              Resource creation Initiated                     
CREATE_IN_PROGRESS                                AWS::Lambda::Permission                           DonutOrNotFunctionHelloPermissionProd             Resource creation Initiated                     
CREATE_IN_PROGRESS                                AWS::Lambda::Permission                           DonutOrNotFunctionClassifyImgPermissionProd       Resource creation Initiated                     
CREATE_IN_PROGRESS                                AWS::Lambda::Permission                           DonutOrNotFunctionHelloPermissionProd             -                                               
CREATE_IN_PROGRESS                                AWS::Lambda::Permission                           DonutOrNotFunctionRootPermissionProd              -                                               
CREATE_IN_PROGRESS                                AWS::Lambda::Permission                           DonutOrNotFunctionGenTimeStampPermissionProd      Resource creation Initiated                     
CREATE_COMPLETE                                   AWS::ApiGateway::Deployment                       ServerlessRestApiDeployment53b600a676             -                                               
CREATE_IN_PROGRESS                                AWS::Lambda::Permission                           DonutOrNotFunctionListImgFilesPermissionProd      Resource creation Initiated                     
CREATE_IN_PROGRESS                                AWS::ApiGateway::Stage                            ServerlessRestApiProdStage                        -                                               
CREATE_IN_PROGRESS                                AWS::ApiGateway::Stage                            ServerlessRestApiProdStage                        Resource creation Initiated                     
CREATE_COMPLETE                                   AWS::ApiGateway::Stage                            ServerlessRestApiProdStage                        -                                               
CREATE_COMPLETE                                   AWS::Lambda::Permission                           DonutOrNotFunctionGenTimeStampPermissionProd      -                                               
CREATE_COMPLETE                                   AWS::Lambda::Permission                           DonutOrNotFunctionListImgFilesPermissionProd      -                                               
CREATE_COMPLETE                                   AWS::Lambda::Permission                           DonutOrNotFunctionClassifyImgPermissionProd       -                                               
CREATE_COMPLETE                                   AWS::Lambda::Permission                           DonutOrNotFunctionRootPermissionProd              -                                               
CREATE_COMPLETE                                   AWS::Lambda::Permission                           DonutOrNotFunctionHelloPermissionProd             -                                               
CREATE_COMPLETE                                   AWS::CloudFormation::Stack                        donut-or-not                                      -                                               
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

CloudFormation outputs from deployed stack
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Outputs                                                                                                                                                                                               
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Key                 DonutOrNotFunctionApi                                                                                                                                                             
Description         API Gateway endpoint URL for Prod stage for DonutOrNot function                                                                                                                   
Value               https://r4eajdv9i1.execute-api.us-west-2.amazonaws.com/Prod/                                                                                                                      

Key                 DonutOrNotFunctionIamRole                                                                                                                                                         
Description         Implicit IAM Role created for DonutOrNot Function                                                                                                                                 
Value               arn:aws:iam::939098422637:role/donut-or-not-DonutOrNotFunctionRole-11RF2YZ088JE4                                                                                                  

Key                 DonutOrNotFunction                                                                                                                                                                
Description         DonutOrNot ARN                                                                                                                                                                    
Value               arn:aws:lambda:us-west-2:939098422637:function:donut-or-not-DonutOrNotFunction-lomdtdVWL2UY                                                                                       
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Successfully created/updated stack - donut-or-not in us-west-2
```

## Caveats when working with AWS Lambda
There are a few gotchas to keep in mind when running a web server via Lambda

 1. The API gateway configured on the [cloud vs local runtime is quite different](https://github.com/aws/aws-sam-cli/issues/1216). If you want your API to accept uploaded files, you need to add ```Api:
    BinaryMediaTypes: ['*~1*']
```
to the `template.yaml` file. This allows the app to accept binary files of all types that are uploaded by the user. Without this, any upload command will be blocked, but return an un-helpful `Internal Server Error` response.

 2. Lambda runs your container with a different set of permission on the cloud compared to the local runtime. This can lead to all sorts of [permission denial issues](https://docs.aws.amazon.com/lambda/latest/dg/troubleshooting-deployment.html). You can address this by elevating permissions for certain folders or files using the `chmod` command. See the `Dockerfile` for examples.