# Berkeley Deep Drive Object Detection


This is the implementation of YOLOv3 for object detection using the darknet framework in Google Colab on the Berkeley Deep Drive dataset. The goal of this project is to successfully train the YOLOv3 network to detect the 10 classes of the BDD. These classes include:
1. bike
2. bus
3. car
4. motorcycle
5. person
6. rider
7. traffic light
8. traffic sign
9. train
10. truck

The inspiration for this project was the book [*Autonomy: The Quest to Build the Driverless Car - and How It Will Reshape Our World*](https://www.amazon.com/Autonomy-Quest-Driverless-Car-Reshape/dp/0062661124) by Lawrence D. Burn. I wanted to explore some of the deep learning that would be required for self-driving cars and object detection is a good starting point. 

### YOLOv3
  [YOLOv3](https://pjreddie.com/darknet/yolo/) is a state-of-the-art convolutional neural network ideal for real-time object detection. Fast and accurate detection is necessary for self-driving cars. Convolutional neural networks(CNN) are great for pattern recognition. They filter images so that basic patterns are recognized first, such as edges, circles, etc., and as the network goes deeper, complex patterns are recognized, say an eye, ear or nose. The mechanism by which this works is convolving the images. This is accomplished by filter nodes in a hidden layer made up of a matrix of random numbers. The dot product of the random number matrix and a matrix from the image is calculated which forms another matrix. For example, if the filter node is a 3x3 matrix, the dot product is calculated over every 3x3 matrix within the image. 
  The structure of YOLOv3 contains 53 convolutional layers and is considered fully convolutional. Every layer uses batch normalization and a 'Leaky ReLU' activation. Leaky ReLU is similar to ReLU but the goal is to avoid 'dying' nodes that occur when weights never activate a node. This is accomplished by having a slight negative output when x<0 instead of an output of zero. YOLOv3 is known for its speed and not sacrificing too much mAP to achieve this. 
  
### Training
Several resources were used to effectively implement YOLOv3 in Google Colab. This included the following:
  
  [The Darknet Repository](https://github.com/AlexeyAB/darknet)
  
  [YOLOv3 in the Cloud](https://www.youtube.com/watch?v=10joRJt39Ns&t=2175s)
  
  [YOLOv3: An Incremental Improvement](https://pjreddie.com/media/files/papers/YOLOv3.pdf)
    
and a lot of googling. I used Google Colab for the code of this project. Google Colab is very similar to Jupyter Notebooks. It allows us to use a GPU for
training in the cloud and that saves a lot of time on training as using a CPU would be extremely slow. Bash commands are used within the notebook as well which is extremely convenient.
  
We begin by cloning in the darknet repository.

    #Clone Darknet Repo to Environment
    !git clone https://github.com/AlexeyAB/darknet

Edit the makefile to have GPU and OpenCV enabled and make darknet.

    #Edit Makefile to have GPU, OpenCV Enabled
    %cd darknet
    !sed -i 's/OPENCV=0/OPENCV=1/' Makefile
    !sed -i 's/GPU=0/GPU=1/' Makefile
    !sed -i 's/CUDNN=0/CUDNN=1/' Makefile
    
    #Make Darknet
    !make

Training requires the image training set and annotations for the image in .txt files. The annotation format is as follows:
    
    (<object_class>, <x_center>, <y_center>, <image_width>, <image_height>)

All the annotated bounding boxes in the image are stored in .txt files that are named the exact same as the image file. We also need train.txt and test.txt that store the file paths of all the images in the training and test sets, respectively. We then will need the .data file that lists the number of classes, the path to the train.txt file, path to the test.txt file, path to the classes.names file and path to where to save weights during training. 

    data.data file:
      classes = 10
      train = path/to/train.txt
      val = path/to/test.txt
      names = path/to/classes.names
      backup = path/to/backup/folder
      
We will also need to edit the .cfg configuration file for training. We need batch=64, max_batches = 2000 x num_classes, steps = 0.8 x max_batches, 0.9 x max_batches and classes = num_classes in the "yolo" layers. 

To train our model we will begin with the darknet53.conv.74 weights. To run training the following command is executed.

      #Training on BDD dataset.
      !./darknet detector train path/to/data.data path/to/custom_cfg.cfg darnet53.conv.74 -dont_show
      
After 8000 iterations, a MSE loss of ~6 is achieved. 

### Test

To calculate the mAP score the following command can be run.

    #Calculate mAP score for saved weights
    !./darknet detector map path/to/data.data path/to/custom_cfg.cfg path/to/weights.weights

Our results by class and overall were as follows:
    
    bike = 25.27%
    bus = 43.60%
    car = 58.96%
    motor = 15.04%
    person = 31.73%
    rider = 23.94%
    traffic light = 36.12%
    traffic sign = 50.01%
    train = 0.00%
    truck = 47.27%
    
    mAP = 33.19%
    
The results are pretty solid for no parameter optimization. The train class AP of 0.00% hurt the performance but this was expected as there were only 136 instances of a train in the training dataset. As opposed to car which had 713,211 instances and performed the best @ 58.96%.

### Example Video

An example of a detected video can be found [here]<https://drive.google.com/file/d/1--sB-B52x4NEsir9XkJZKQeYIplMaQep/view?usp=sharing>. 

### Future Work
1. Explore parameter optimization to see how performance can be improved.

2. Use other self-driving datasets in unison with BDD for training so that it might be improved (KITTI, etc.)

3. Train for longer. 

### References

1. [Berkeley Deep Drive]<https://bdd-data.berkeley.edu/>
2. [Autonomy : The Quest to Build the Driverless Car - And How It Will Reshape Our World]<https://www.amazon.com/Autonomy-Quest-Driverless-Car-Reshape/dp/0062661124>
3. [YOLOv3: An Incremental Improvement]<https://pjreddie.com/media/files/papers/YOLOv3.pdf>
4. [YOLOv3 in the Cloud]<https://www.youtube.com/watch?v=10joRJt39Ns&t=2175s>
5. [Darknet Repository]<https://github.com/AlexeyAB/darknet>
6. [Real-time object detection for autonmous vehicles using deep learning]<https://uu.diva-portal.org/smash/get/diva2:1356309/FULLTEXT01.pdf>
  
      
  
  
  
  
