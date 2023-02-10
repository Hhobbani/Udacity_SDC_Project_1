# Project 1: Object Detection in an Urban Environment

## General notes and issues

During the course of the project, I was forced to come up with several tricks to compile my code. I attempted to work in the environment provided by you (as described in the instructions). Unfortunately, this did not work. The Mozilla browser kept crashing.

<img src="Pictures/Mozilla_Error.png" width=80% height=60%>

Even after I installed the Chromium browser, it also kept crashing.

<img src="Pictures/Chromium_Error.png" width=100% height=60%>

I was only able to run my code, which I implemented on my personal laptop, once out of 10 attempts. Even the Jupyter Notebook `Explore augmentations.ipynb` could not be opened.

<img src="Pictures/Notebook_Error.png" width=100% height=60%>

It was only possible to open the notebook directly in Github. As a result, I created another notebook `My Explore augmentations.ipynb`.

I tried to download my workspace to work on a laptop with a graphic card. However, I was not able to set up the environment and to make matters worse, the download failed. I had to download everything separatly and then add the files to my GitHub. If something is missing everything is available in my workspace and all implemented parts work there with no issues if you start it from terminal and avoid using jupyter notebook in my workspace.

<img src="Pictures/Download_Error.png" width=100% height=60%>

I was truly confused throughout the entire project. I apologize if the project is not well documented enough. It was due to my frustration and headaches during the project.

Finally, I tried to upload everything to my workspace, including this README file.

## Goal

Classification and detection of cars, pedestrians and cyclists in images.

### Available data

The data we will use is organized as follow:
```
/home/workspace/data/
    - train: contain the train data 
    - val: contain the val data 
    - test: contains the test data
```

## Step 1a - Exploratory Data Analysis (EDA)


### Implementing the function `display_images.py`

It was mentioned using the folder /data/waymo/training_and_validation/ which I could not find in the workspace. So I just used the folder /data/train/ for the exploration.

<img src="Pictures/exploring_1.png" width=60% height=60%>
<img src="Pictures/exploring_2.png" width=60% height=60%>
<img src="Pictures/exploring_3.png" width=60% height=60%>
<img src="Pictures/exploring_4.png" width=60% height=60%>

### Analysis

I analyzed a random sample of 30,000 from the dataset.

1. Class distribution analysis

<img src="Pictures/object_count.png" width=50% height=50%>

i

4. Distribution of class frequency in an image

<img src="Pictures/object_frequency.png" width=150% height=50%> 

## Step 1b - Create the training - validation splits

I don't understand this step because the data in my workspace was already split. That's why I didn't run the script `create_splits.py`.

## Step 2 - Edit the config file

Executed as discribed in the project instructions and a new config file `pipeline_new.config` has been created and moved to the folder `/home/workspace/experiments/reference/`.

## Step 3 - Model Training and Evaluation

The training process was launched as discribed in the project instructions and tensorboard had the following output:

<img src="Pictures/tensorboard_1_1.png" width=150% height=50%> 
<img src="Pictures/tensorboard_1_2.png" width=150% height=50%>

## Step 4 - Improuve the performance

In order to improve the classification goodness I explored the Object Detection API and applied many different augmentations using different strategies.

These strategies were added directly to the config file `pipeline_new.config` in the folder `/home/workspace/experiments/reference/`. 

For instance, I added the augmentation for adjusting the saturation of the image with the following command lines:

```
data_augmentation_options {
  	random_adjust_saturation{
    }
  }
```
I also changed the base learning rate from 0.04 to 5^10-4.

|<img src="Pictures/augmentation_1.png" width=50% height=50%> | <img src="Pictures/augmentation_2.png" width=50% height=50%>|

After I modified the configuration file, I restarted the training and the following TensorBoard output was produced.

<img src="Pictures/tensorboard_2_1.png" width=150% height=50%> 
<img src="Pictures/tensorboard_2_2.png" width=150% height=50%>

After the training process is done, I executed the evaluation command

```
python experiments/model_main_tf2.py --model_dir=experiments/reference/ --pipeline_config_path=experiments/reference/pipeline_new.config --checkpoint_dir=experiments/reference/
```

and started again TensorBoard with the following output:

<img src="Pictures/tensorboard_3_1.png" width=150% height=50%> 
<img src="Pictures/tensorboard_3_2.png" width=150% height=50%>
<img src="Pictures/tensorboard_3_3.png" width=150% height=50%> 

### Creating animations

<img src="Pictures/animation_1.gif" width=150% height=50%> 
<img src="Pictures/animation_2.gif" width=150% height=50%>
<img src="Pictures/animation_3.gif" width=150% height=50%> 
