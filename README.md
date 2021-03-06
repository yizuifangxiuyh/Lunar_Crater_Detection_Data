# Lunar_Crater_Detection_Data
A new lunar crater dataset that contains nearly 20,000 craters captured from different NAC CDR images for crater detection studies.

We use the NAC (Narrow Angle Camera) images produced by the space probe LRO (Lunar Reconnaissance Orbital). NAC products have a panchromatic band with a resolution of 0.5 meters per pixel, covering approximately 5 kilometers. We collected 22 pairs of NAC CDR images with different illumination conditions, which were near the Chang’E-4 landing site between latitudes 45° and 46° S and longitudes 176.4° and 178.8° E. Among them, there were 4 pairs with insufficient illumination conditions to identify the craters. In fact, when the solar incidence angle is near 0, no resolvable shadows are cast, and surface albedo variations stand out. In contrast, the high solar incidence angle images have a large part hidden by shadows.

Note that, craters with diameters below 8 pixels were ignored due to their hardly detectable nature in our data set.


# Citation
If you find this project useful for your research, please cite our work.

Huan Yang, Xinchao Xu, Youqing Ma, Yaming Xu, Shaochuang Liu, “CraterDANet: A Convolutional Neural Network for Small-Scale Crater Detection via Synthetic-to-Real Domain Adaptation,” IEEE Transactions on Geoscience and Remote Sensing.
