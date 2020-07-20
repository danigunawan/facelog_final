# facelog_project
****
* Face recognition and face verification

****
# First
you show create train_dir folder 
```
mkdir train_dir
```
and then put your dataset to train_dir
```
train_dir
│      
│
└───folder1
│   │   file011.jpg
│   │   file012.jpg
│   │   file013.jpg
│   
└───folder2
    │   file021.jpg
    │   file022.jpg
    |   file023.jpg

```
# Two steps for training
 ## step_1: Load image data and convert to numpy extract to face embedding(128 )
 ```
 Create_embedding_.py
 ```
 ## step_2: Train for classifier 
 ``` 
 Train_Classifier_.py
 ```
# For testing 
``` 
  python3 inference_.py
```
# Start system using docker
```
docker-compose up
```


