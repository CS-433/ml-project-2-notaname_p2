[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/UDdkOEMs)


# Machine Learning Project 2  

The overall goal of this project is perform an image segmentation on aerial/satellite image to find the roads. 
 

### <b>Project overview</b>
This project aims to develop a machine learning model to predict the likelihood of a pixel in an image to be a road or not. Based on that prediction we will attribute a label to each pixel of a satellite image. With road beeing set to white and non-road set to black  (road=1, background=0).

Example : 

<img width="839" alt="road example" src="https://github.com/user-attachments/assets/9de1fb79-7a19-49e2-ac62-e50491e1212f">

## How to Run the Code 

You may want to run the code with either run.py or run.ipynb (the latter is more readable in our opinion, the choice is yours).

1. **Clone the repository**  
   Open your terminal and run the following command to clone this repository:

   ```bash
   cd YOUR_CLONING_DIRECTORY
   git clone https://github.com/CS-433/ml-project-2-notaname_p2
   cd ml-project-2-notaname_p2
2. **Get the enviroment and activate it**   

   ```bash
   conda env create -f environment.yml
   conda activate ML_project2

3. **Get pretrained Models**

   You will need to redownload the UNet_model.pth and RC_params_opti.pth files and place them in the project folder. If you don't you might enconter an error stating the files are corrupted. By using a pretrained model the prediction time would be around 10 minutes, if you want to train the model your self it might take several hours.

4. **Get the best prediction**
   
   open UNet_pred.ipynb on any editor, selected the ML_project2 enviromnent and run all cells

  ## Best Prediction and Getting other Predictions

  To get best prediction that we obtained, run UNet_pred.ipynb with all parameters set to False (already done by default). 
  
  If you want to predict best parameters (best number of layers/bases,...), set UNET_SEARCH to True. 
  
  If you want in addition to save them, set both SAVE_UNET and UNET_TRAIN to True. For the hyperparameter search for postprocessing part, set ROAD_CORRECTION_SEARCH to True (to save them set SAVE_RC_PARAMS to True).
  
  The last parameter THRESHOLD_SEARCH is for searching optimal threshold to minimize F1 loss. 




##### Authors: Joana Pires, Leonardo Tredici, Antonin HUDRY
