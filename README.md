# facelog__face
you show create traind dir data set
`mkdir train_dir`
And then you should put your dataset to train_dir
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
 * step_1
 # load image data and convert to numpy extract to face embedding(128 )
 ```
 Create_embedding_.py
 ```
 * step_2
 # train for classifier 
 ``` 
 Train_Classifier_.py
 ```
# For Testing 
``` 
  python3 inference_.py
```


