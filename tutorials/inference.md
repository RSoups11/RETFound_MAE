# Tutorial to use RETFound

## Git clone 

First you'll have to git clone from [my repo](https://github.com/RSoups11/RETFound_MAE)
 : 

```bash
git clone https://github.com/RSoups11/RETFound_MAE.git
```

And you should install the depedencies from the requirements.txt

```bash
pip install -r requirements.txt
```


## Folder tree

__To facilitate your work, I have provide a notebook called "copy_images.ipynb" at the root of RETFound_MAE. Please keep reading this tutorial to understand how the folder tree works in RETFound.__

Now that you have the folder RETFound_MAE, you should create a folder called "data" in it's root.

```bash
mkdir data
```

And inside this data folder you should create 3 sub-folder, "train", "val" and "test".

```bash
cd data/
mkdir train
mkdir val
mkdir test
```

Next we are going to create a "dummy" subfolder in each folder. RETfound by default is still waiting for each split to have folder with the class name with at least one images in each classes. 

Since the goal is to simulate the inference we are going to create a false class called "dummy" in each folder to avoid any crash from the code.

At the end of the day, since we are going to evaluate an already fine_tuned model for classification, only the test set will be used and the class doesn't matter since we are not trying to compute any kind of metrics with unlabelled data (We will need to manually labelled the images if we want to double check). 

## Populate the folder

For inference RETFound is waiting for each folder to have at least 1 images. So you should put 1 random images in the train/dummy and val/dummy subfolder at least (it will have no impact for the inference)

Then you have to put all of your unlabelled data in the test/dummy subfolder

## Weights 

Now you should create at the root of RETFound_MAE a folder called "weights" :

```bash
mkdir weights
```

Inside this folder you are going to put the weights that you are going to use (e.g RETFound_binary_messidor.pth)

## Inference

Now to run the inference from RETFound, you should place yourself at the root of the folder 

```bash
cd ~/RETFound_MAE
```

And you run this command : 

```bash
torchrun --nproc_per_node=1 --master_port=48798 main_finetune.py \
  --model RETFound_mae \
  --eval \
  --global_pool \
  --batch_size 32 \
  --nb_classes 2 \
  --input_size 224 \
  --data_path ./data \
  --task <task_name> \
  --best_threshold <threshold_value> \
  --inference \
  --resume ./weights/<weights_name>.pth
```

You should chose a task name as you please. After the run this will create 2 folders : output_dir and output_logs. 

Inside the output_dir you will find a folder with the task name that you have chosen. The prediction.csv file will be in this folder.checkpoint-best

It is imporant to inclue the inference argument or else the prediction.csv won't be produced.

_N.B : The threshold value is related to which threshold is the best for a given .pth file_
