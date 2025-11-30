## Automatic Cardiac CT Myocardial and Aortic Valve Segmentation Using Deep Learning to Support Preoperative Simulation

#### Document
The complete report can be obtained from [Google Drive](https://drive.google.com/file/d/1BcTWPRWM-KDBQLAvjLFeE12Nd5EDWvNy/view?usp=sharing).

#### Introduction  
Severe aortic stenosis treated with Transcatheter Aortic Valve Replacement (TAVR) requires precise preoperative evaluation of the aortic root and surrounding myocardium on cardiac CT, yet current measurements are largely manual, time-consuming, and operator-dependent. This study proposes a deep learning framework that combines a YOLOv8m detector for automatic aortic valve localization with myocardial segmentation to provide a complete anatomical context for TAVR planning. Trained on a cardiac CT dataset from the AI CUP challenge, the detector operates on 2D axial slices and is evaluated using standard object detection metrics. In the internal validation set, the model achieves a precision of 0.958, recall of 0.96, mAP@0.5 of 0.987, and mAP@0.5:0.95 of 0.783, and obtains a leaderboard score of 0.9635 in the official hidden test set. These results indicate that a detection-driven pipeline can accurately and efficiently localize the aortic valve while providing anatomically meaningful outputs to support preoperative simulation and reduce clinical workload.

#### Collaborators
Fan-Jia Huang, Hsueh-Po Lu, Mu-En Chiu, and Ya-Chi Tu.
