# LUMC-Large-Echo

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
(patient lay on the side facing its left)
    1. Parasternal
    2. Apical
(patient lay on the back facing the ceiling)
    3. Subcostical, aka 肋骨下
    4. Supersternal, aka 胸骨上喉嚨附近

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
