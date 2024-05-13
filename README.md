# Image Classification FastAPI

## Introduction
This repository offers an implementation of cat vs dog image classification where resnet 50 network is employed for binary classification. Furthermore, a FastAPI is implemented
for prediction.

# API

| Method  | Route | Functionality | Access |
| ------------- | ------------- | ------------- | ------------- |
| POST | */predict/* | Predict image | All user |

# Training/Evaluation

| Flags  | Usage |
| ------------- | ------------- |
| ```--demo``` | run demo | 
| ```--train```  | initiate training method	|                                                                   
| ```--epoch```  | set number of epoch |
| ```--data_path```  | set dataset path | 
| ```--img_path```  | set image path | 

# Training 
 ```
 python demo.py --train --epoch <set>
 ```

# Evaluation
 ```
 python demo.py --demo
 ```

# Run API
 ```
 uvicorn main:app --reload
 ```
