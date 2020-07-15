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
 * 1_run
 # load image data to numpy and extract to face embedding(128 )
 ```
 Create_embedding_.py
 ```
 * 2_run
 # train for classifier 
 ``` 
 Train_Classifier_.py
 ```
# For Testing 
``` 
  python3 inference_.py
```


