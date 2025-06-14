# RipOperation
## A python code gallery for flash rip detection, tracking, and characterization
This is the code for tracking rip currents and characterizing their features from web-cam images.

## Background
### Flash Rips
Rip currents are narrow, fast-moving flows that move seaward and can quickly carry swimmers away from the shoreline, causing numerous drowning fatalities. They are often driven by bathymetric features such as rip channels and nearshore structures, but can also form purely due to hydrodynamic conditions—these are known as flash rips. Flash rips are particularly dangerous because they are intermittent, transient, and can occur on seemingly featureless beaches, making them difficult to predict and detect.
<p align="center">
  <img src="https://github.com/wwang487/RipOperation/blob/main/FlashRip.jpg" width="50%">
</p>

The visual cues of flash rips differ from those of other types of rip currents. Unlike bathymetrically induced rips, flash rips often lack visible bubble trails, making them difficult to detect. However, previous studies have noted that seaward-moving sediment plumes—such as the one shown in the image above—can serve as indicators of flash rip activity.

### LOCKS
The Lifeguarding Operational Camera Kiosk System (LOCKS) is a real-time monitoring system for rip current detection and warning, deployed at Port Washington, Wisconsin, on the western shore of Lake Michigan. LOCKS captures water surface images every 10 seconds and transmits them to a backend PC station for processing. If hazardous flash rip currents are detected, the system automatically issues alerts through an on-site warning system.
<p align="center">
  <img src="https://github.com/wwang487/RipOperation/blob/main/LOCKS_Site.jpg">
</p>

## Refined Cascade R-CNN Detection Framework
Flash rip detection is performed using a Refined Cascade R-CNN framework. The model is first trained on a manually labeled dataset, where bounding boxes delineate flash rip regions versus background. Training and inference are implemented using mmDetection, an open-source toolbox developed by OpenMMLab. Once trained, the best-performing Cascade R-CNN model is used to infer flash rips in newly acquired images. The cascade structure is configured with Intersection over Union (IoU) thresholds of 0.45, 0.55, and 0.65.
<p align="center">
  <img src="https://github.com/wwang487/RipOperation/blob/main/Cascade.jpg">
</p>

The optimal model obtained from training is then applied to new incoming images for flash rip detection. As illustrated in [RefinedCascadeRCNN](https://github.com/wwang487/RefinedCascadeRCNN), a post-processing step is implemented to enhance detection precision. In this step, flash rip candidates with low confidence scores (p-values) or overly flat shapes are removed, and overlapping or intersecting detections are merged. The thresholds used in this refinement process are informed by field observations and expert knowledge. This refinement strategy significantly improves precision, increasing it by more than 10%.
<p align="center">
  <img src="https://github.com/wwang487/RipOperation/blob/main/Refinement.jpg">
</p>

Once flash rip events are detected—or environmental conditions suggest a high likelihood of their occurrence—the warning system alerts beachgoers. A red light is activated, and warning messages are displayed on the Kiosk screen.
<p align="center">
  <img src="https://github.com/wwang487/RipOperation/blob/main/LOCKS_Inside.jpg">
</p>

## Contributor
Wei Wang, University of Wisconsin-Madison

Yuli Liu, University of Wisconsin-Madison; Nanjing University of Information Science and Technology

Boyuan Lu, University of Wisconsin-Madison

Daniel Wright, University of Wisconsin-Madison

Chin H. Wu, University of Wisconsin-Madison
