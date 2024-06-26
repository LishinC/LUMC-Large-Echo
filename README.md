# LUMC-Large-Echo

## Planning
1. **Model inference**: apply the pretrained model on the 200 labeled LUMC validation cases
    * For the 23 previously-seen views	-> see accuracy
    * For all other unseen 2D views		-> see embedding & uncertainty
    * For all other unseen image types	-> see embedding & uncertainty
2. **Decision making**:
    * If the accuracy on the 23 views is low -> proceed to step 3 to improve the accuracy on the 23 class
    * If the accuracy on the 23 views is high but the uncertainty for the unseen views is low that they're often confidently mis-classified into one of the 23 classes -> proceed to step 3 to better classify the unseen
    * If accuracy on the 23 views is high & uncertainty for the unseen views is high -> The model can already be applied on the whole LUMC data to identify the 23 views and the uncertain (potentially the unseen) ones
3. **Model improving**: label a few more cases for training
    * Keep the model architecture:
        * Fine tune the model
        * Semi-supervised learning
        * Identify cluster centers ...
        * Self-supervised learning
        * Few-shot learning
    * Altering the model architecture:
        * Corase classification -> detailed classification (e.g. first identify PLAX/PSAX/Apical/Subcostal then identify subclasses with 4 dedicated models; Or, first identify image type then identify subclasses with dedicated mdoels)
        * Better utlization of the temporal information in the clips (e.g. use 3D CNN or 2D CNN + LSTM for the clips.)

## Replicating the echocv project
* link is https://bitbucket.org/rahuldeo/echocv/src/master/
1. Clone the repository and enter the folder
   ```
   git clone https://bitbucket.org/rahuldeo/echocv.git
   cd echocv
   ```
2. Inside the echocv folder, manually create a folder called "models". Then download the three checkpoint files whose names start from "view_23_e5_class_11-Mar-2018" from https://www.dropbox.com/sh/0tkcf7e0ljgs0b8/AACBnNiXZ7PetYeCcvb-Z9MSa?dl=0 , and put them inside the "models" folder that was just created.
3. Now create the virtual environment according to the instructions on echocv project. Python 2.7.11 wasn't easy to install, so install Python 2.7.18 for now.
   ```
   conda create -n echocv python=2.7
   conda activate echocv
   pip install -r requirements.txt
   ```
4. Now you are free to reproduce the project. For the specific focus of saving the trained tensorflow view-classification model into a Pytorch replicate, put save_as_torch_model.py and pytorch_vgg.py into the echocv folder, and run save_as_torch_model.py . These scripts are derived from predict_viewclass_v2.py and nets/vgg.py in the original echocv repository.


## Types of images
* 2D view
    * Gray scale 2D+time videos
* Color Doppler overlay
    * The window size of the color Doppler overlay can be adjusted. Color bar in general is yellow - red - blue - green representing flow towards the transducer all the way to away from the transducer.
* Spectral Doppler
    * 1D+time, focusing on a line from the 2D Doppler, or a small fragment of the line. Presented in yellow, many annotations are done on this view, either by labeling the peak (a small plus sign will be put on the peak) or tracing the envelop. The measurements will be presented in the top left corner of the image. During this view the signal will also be played in audio, resembling a heat-beat like sound.
* M-mode
    * 1D+time, which is a line from the 2D image. Often used to see the tissue motion. In the past when the machine wasn't that powerful to get high resolution 2D+time videos, 1D+time image is very popular.
* Tissue Doppler
    * Often in blue and red if the color bar is not adjusted, because the movement of the tissue is not as significant as the blood.
* Strain
    * Myocardium is separated into fragments and the movement of each fragment is tracked.
* 3D view
    * 3D+time gray scale image. If there is a movement of the patient or the probe, then there will be artifacts and the image quality is not always great. If acquired nicely then it's used to derive ejection fraction.

## Acquisition ordering of different views in general
0. (patient lay on the side facing its left)
1. Parasternal
2. Apical
3. (patient lay on the back facing the ceiling)
4. Subcostical
5. Supersternal

## Acquisition ordering of different types of images at each position
1. The 2D view  is fist used to get the position of the probe correctly and thus the view correctly
2. Then the color Doppler overlay can be turned on, focusing on an overview of regurgitation at the valve or the tissue Doppler
3. Then switch to spectral Doppler for quantification
4. M-mode image or tissue Doppler can also be acquired

## Other
* Frame rate
    * The power from the transducer is limited. Depending on how wide the window is, the machine automatically adjust the frame rate to the maximum it can get. To capture the motion better, the window has to be more focused (narrower field of view), so that details won't be missed in between two frames.

* Annotations
    * Annotations were done right during the exam or after the exam in the image room. Automatic annotation is possible on some of the machine but is not gauranteed to be accurate. In automatic annotation the machine recognize the view, does the tracing, and determine what parameter is drived from this view and it's value. It can average the value over several heartbeat. The sonographer can approve or reject the automatically derived values. Tracing of LV and RV for the calculation of volume and ejection fraction are usually done manually on A2C and A4C views.


## Postprocessing
* This can significantly alter the image apprance, which might be a problem for the deep learning model. Hopefully there are sufficient varieties in the dataset which let the model familarized with all kinds of post-processing variations preffered by different sonographers.
* Color bar can be adjusted to better distinguish flow in certain direction.
* Gain, compressing, smoothing to remove noize, sharpening to increase contrast. Zooming.
