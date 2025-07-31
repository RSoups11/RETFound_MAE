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

## Populate the folder

For the finetuning, you should take into consideration that RETFound will be using alphanumerical order to determine the classes.

For example, if you have a __data/train/DR__ folder where you put your labelled DR images and a __data/train/NoDR__ folder where you put you labelled NoDR images. RETFound is actually going to put the images of the DR folder as class 0 since the folder name start with an _D_ which comes before _N_. 

A good practice is to always start your folder with a number, such as __1_DR__ to ensure that the folder is link to the class that you want.

## Weights 

Now you should create at the root of RETFound_MAE a folder called "weights" :

```bash
mkdir weights
```

Inside this folder you are going to put the weights that you are going to use if you are going to use a custom RETFound's weight (e.g RETFound_binary_messidor.pth)

However if you are planing to use a weight provided by the original authors from hugginface, please follow step 1 and 2 of Fine-Tuning Section [here](https://github.com/rmaphoh/RETFound_MAE/blob/main/README.md).

After that you can either download the official weight from huggingface or specify the path of your custom weight in the command line.

## Fine-Tuning

### Train

Now to run the finetuning from RETFound, you should place yourself at the root of the folder 

```bash
cd ~/RETFound_MAE
```

And you run this command : 

```bash
torchrun --nproc_per_node=1 --master_port=48798 main_finetune.py \
    --model RETFound_mae \
    --savemodel \
    --global_pool \
    --batch_size 32 \
    --epochs 50 \
    --blr 6e-4 --layer_decay 0.8 \
    --weight_decay 0.1 --drop_path 0.1 \
    --nb_classes 2 \
    --input_size 224 \
    --data_path ./data \
    --task <task_name> \
    --finetune RETFound_mae_natureCFP # specify .pth path for custom
```
You should chose a task name as you please. After the run this will create 2 folders : output_dir and output_logs. 

Inside the output_dir you will find a folder with the task name that you have chosen. This folder will contain the loss curve and the confusion matrix of the train/validation set.

_N.B : This is an example of command for binary classification in the [usage](./usage.md) file you can find ALL of the possible arguments._

### Test

After the training is done, you can now proceed to the evaluation on the test set.

To do so run : 

```bash
torchrun --nproc_per_node=1 --master_port=48798 main_finetune.py \
      --model RETFound_mae \
      --eval \
      --global_pool \
      --batch_size 32 \
      --nb_classes <number_of_classes> \
      --input_size 224 \
      --data_path ./data \
      --task <task_name> \
      --best_threshold <treshold_value> \
      --resume <path/to/the/trained/weight.pth>
```

Once again this will create a folder with your task_name within the output_dir folder. Inside this folder you'll find the loss curves and the confusion matrix on the test set, as well as a csv with the metrics. 

_N.B : I am not sure about how RETFound is computing the loss per batch/epoch, so the loss curves may need to be double checked in the __engine_finetune.py__ code_

## Hyperparameter search

__optuna_search.py__ wraps Optuna, a Bayesian-optimisation library that explores hyper-parameter combinations in a smart way and prunes trials that look unpromising (early stopping).

With the script you can search on : 

- Base learning rate (blr)
- Weight decay
- drop_path
- layer_decay
- smoothing
- mixup
- cutmix
- epoch
- batch_size
- reprob

As well as the other classic argument to launch the RETFound __main_finetune.py__ script. The hyperparameter were arbitraly chosen in the optuna's script, so you should modify the search space in __build_args()__.

### Running the script

To run the script place yourself at the root of RETFound_MAE

```bash
cd ~/RETFound_MAE
```

I advise you to run the script with nohup since it can take some times to do the search 

```bash
nohup python optuna_search.py > optuna/optuna.log 2>&1 &
```

The script should then create an __optuna__ folder, in which, you will find a csv with all of the non-pruned trial hyperparameters and their AUC. Also, the weight of the best combinaison should be saved up as `best_checkpoint.pth` in the folder.

If you want an interactive dashboard : 

```bash
pip install optuna-dashboard
optuna-dashboard sqlite:///optuna/retfound.db
```

_N.B : if you encountered locked database error, just modify the DB_FILE line in the configuration section with __DB_FILE = "sqlite:///:memory:"__. This is specially true if you run this script on Azure ML studio_
