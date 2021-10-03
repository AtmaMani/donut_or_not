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
