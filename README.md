# RipOperation-A python code gallery for analyzing flash rips and associated environmental variables.
This is the code for tracking rip currents and characterizing their features from web-cam images.

## Background
### Flash Rips
Rip currents are narrow, fast-moving flows that move seaward and can quickly carry swimmers away from the shoreline, causing numerous drowning fatalities. They are often driven by bathymetric features such as rip channels and nearshore structures, but can also form purely due to hydrodynamic conditions—these are known as flash rips. Flash rips are particularly dangerous because they are intermittent, transient, and can occur on seemingly featureless beaches, making them difficult to predict and detect.
!['./FlashRip.jpg'](https://github.com/wwang487/RipOperation/blob/main/FlashRip.jpg)

The visual cues of flash rips differ from those of other types of rip currents. Unlike bathymetrically induced rips, flash rips often lack visible bubble trails, making them difficult to detect. However, previous studies have noted that seaward-moving sediment plumes—such as the one shown in the image above—can serve as indicators of flash rip activity.

### LOCKS
The Lifeguarding Operational Camera Kiosk System (LOCKS) is a real-time monitoring system for rip current detection and warning, deployed at Port Washington, Wisconsin, on the western shore of Lake Michigan. LOCKS captures water surface images every 10 seconds and transmits them to a backend PC station for processing. If hazardous flash rip currents are detected, the system automatically issues alerts through an on-site warning system.

!['./LOCKS_Site.jpg'](https://github.com/wwang487/RipOperation/blob/main/LOCKS_Site.jpg)

## Contributor
Wei Wang, University of Wisconsin-Madison

Yuli Liu, University of Wisconsin-Madison; Nanjing University of Information Science and Technology

Boyuan Lu, University of Wisconsin-Madison

Daniel Wright, University of Wisconsin-Madison

Chin H. Wu, University of Wisconsin-Madison
