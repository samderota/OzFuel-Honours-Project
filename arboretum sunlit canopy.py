# In[]: Init and open data file

import numpy as np
import matplotlib.pyplot as plt
import bisect
import time
import os 
import skimage.transform as ski
import pandas as pd

os.chdir('C:/Users/Sam/Google Drive/UNI/2022/ENGN4350/Python/') #set CWD
start = time.time()

# Import and open file
file1 = "C:/Users/Sam/Google Drive/UNI/2022/ENGN4350/Python/Reorder_288_2430_384_U16.bin"       #arboretum file address
wavelengths = [-8.6306, 0.9195025, 10.469605, 20.0197075, 29.56981, 39.1199125, 48.670015, 58.2201175, 67.77022, 77.3203225, 86.870425, 96.4205275, 105.97063, 115.5207325, 125.070835, 134.6209375, 144.17104, 153.7211425, 163.271245, 172.8213475, 182.37145, 191.9215525, 201.471655, 211.0217575, 220.57186, 230.1219625, 239.672065, 249.2221675, 258.77227, 268.3223725, 277.872475, 287.4225775, 296.97268, 306.5227825, 316.072885, 325.6229875, 335.17309, 344.7231925, 354.273295, 363.8233975, 373.3735, 382.9236025, 392.473705, 402.0238075, 411.57391, 421.1240125, 430.674115, 440.2242175, 449.77432, 459.3244225, 468.874525, 478.4246275, 487.97473, 497.5248325, 507.074935, 516.6250375, 526.17514, 535.7252425, 545.275345, 554.8254475, 564.37555, 573.9256525, 583.475755, 593.0258575, 602.57596, 612.1260625, 621.676165, 631.2262675, 640.77637, 650.3264725, 659.876575, 669.4266775, 678.97678, 688.5268825, 698.076985, 707.6270875, 717.17719, 726.7272925, 736.277395, 745.8274975, 755.3776, 764.9277025, 774.477805, 784.0279075, 793.57801, 803.1281125, 812.678215, 822.2283175, 831.77842, 841.3285225, 850.878625, 860.4287275, 869.97883, 879.5289325, 889.079035, 898.6291375, 908.17924, 917.7293425, 927.279445, 936.8295475, 946.37965, 955.9297525, 965.479855, 975.0299575, 984.58006, 994.1301625, 1003.680265, 1013.2303675, 1022.78047, 1032.3305725, 1041.880675, 1051.4307775, 1060.98088, 1070.5309825, 1080.081085, 1089.6311875, 1099.18129, 1108.7313925, 1118.281495, 1127.8315975, 1137.3817, 1146.9318025, 1156.481905, 1166.0320075, 1175.58211, 1185.1322125, 1194.682315, 1204.2324175, 1213.78252, 1223.3326225, 1232.882725, 1242.4328275, 1251.98293, 1261.5330325, 1271.083135, 1280.6332375, 1290.18334, 1299.7334425, 1309.283545, 1318.8336475, 1328.38375, 1337.9338525, 1347.483955, 1357.0340575, 1366.58416, 1376.1342625, 1385.684365, 1395.2344675, 1404.78457, 1414.3346725, 1423.884775, 1433.4348775, 1442.98498, 1452.5350825, 1462.085185, 1471.6352875, 1481.18539, 1490.7354925, 1500.285595, 1509.8356975, 1519.3858, 1528.9359025, 1538.486005, 1548.0361075, 1557.58621, 1567.1363125, 1576.686415, 1586.2365175, 1595.78662, 1605.3367225, 1614.886825, 1624.4369275, 1633.98703, 1643.5371325, 1653.087235, 1662.6373375, 1672.18744, 1681.7375425, 1691.287645, 1700.8377475, 1710.38785, 1719.9379525, 1729.488055, 1739.0381575, 1748.58826, 1758.1383625, 1767.688465, 1777.2385675, 1786.78867, 1796.3387725, 1805.888875, 1815.4389775, 1824.98908, 1834.5391825, 1844.089285, 1853.6393875, 1863.18949, 1872.7395925, 1882.289695, 1891.8397975, 1901.3899, 1910.9400025, 1920.490105, 1930.0402075, 1939.59031, 1949.1404125, 1958.690515, 1968.2406175, 1977.79072, 1987.3408225, 1996.890925, 2006.4410275, 2015.99113, 2025.5412325, 2035.091335, 2044.6414375, 2054.19154, 2063.7416425, 2073.291745, 2082.8418475, 2092.39195, 2101.9420525, 2111.492155, 2121.0422575, 2130.59236, 2140.1424625, 2149.692565, 2159.2426675, 2168.79277, 2178.3428725, 2187.892975, 2197.4430775, 2206.99318, 2216.5432825, 2226.093385, 2235.6434875, 2245.19359, 2254.7436925, 2264.293795, 2273.8438975, 2283.394, 2292.9441025, 2302.494205, 2312.0443075, 2321.59441, 2331.1445125, 2340.694615, 2350.2447175, 2359.79482, 2369.3449225, 2378.895025, 2388.4451275, 2397.99523, 2407.5453325, 2417.095435, 2426.6455375, 2436.19564, 2445.7457425, 2455.295845, 2464.8459475, 2474.39605, 2483.9461525, 2493.496255, 2503.0463575, 2512.59646, 2522.1465625, 2531.696665, 2541.2467675, 2550.79687, 2560.3469725, 2569.897075, 2579.4471775, 2588.99728, 2598.5473825, 2608.097485, 2617.6475875, 2627.19769, 2636.7477925, 2646.297895, 2655.8479975, 2665.3981, 2674.9482025, 2684.498305, 2694.0484075, 2703.59851, 2713.1486125, 2722.698715, 2732.2488175] #wavelengths captured accroding to metadata
bands = len(wavelengths)
length = 2430       #file specifc
height = 384
sensor_max = 2**16  #SWIR sensor has a 16 bit readout
with open(file1) as f:
    dt = np.dtype('<u2')   #< is for little endian, u2 is for unsigned int 2 bytes (16-bit)
    rawdata = np.fromfile(f, dtype=dt)
rawdata = np.reshape(rawdata, (len(wavelengths), length, height))  #mould bitstream into data cube
data = np.rot90(rawdata, 1, (1,2)) #rotate for correct orientation

bins = np.linspace(0, 1, 20)  #20x bins across 0-1 normalised data, need same bins to compare between bands and sensors

#Cropping!
data = data[:, 145:215, 1190:2010]                 
length = data.shape[2]
height = data.shape[1]


# Process hyperspectral data to generate image simulations

#nested list containing each band for every platform, note - only contains comparison bands. full list in archive.py
base_bands = list(np.arange(900,2500,1))
ozfuel_bands = [ list(np.linspace(1200,1210,1210-1200+1)) , list(np.linspace(1655, 1665, 1665-1655+1)) , list(np.linspace(2095, 2105, 2105-2095+1)) , list(np.linspace(2255, 2265, 2265-2255+1))]
modis_bands = [ list(np.linspace(1230,1250,1+1250-1230)) , list(np.linspace(1628,1652,1+1652-1628)) , list(np.linspace(2105,2155,1+2155-2105))]
sentinel_bands = [ list(np.linspace(1358,1389,1+1389-1358)), list(np.linspace(2115,2290,1+2290-2115)) ]
landsat_bands = [ list(np.linspace(1566,1651,1+1651-1566)) , list(np.linspace(2107,2294,1+2294-2107)) ]
himawari_bands = [ list(np.linspace(1600,1620,1+1620-1600)) , list(np.linspace(2250,2270,1+2270-2250)) ]

#Sensor spatial resolutions
ozfuel_scaling = 50 
modis_scaling = 250
sentinel_scaling = 20
landsat_scaling = 30
himawari_scaling = 2000

#calculate dimensions of simulated images according to scaling factor
ozfuel_dims = [int(height/ozfuel_scaling), int(length/ozfuel_scaling)]
modis_dims = [1,3]#[int(height/modis_scaling), int(length/modis_scaling)]
sentinel_dims = [int(height/sentinel_scaling), int(length/sentinel_scaling)]
landsat_dims = [int(height/landsat_scaling), int(length/landsat_scaling)]
himawari_dims = [1,1]# int(length/himawari_scaling)]                               #causes issues because floor of 0.xx is 0

#figure out SWIR data indices that correspond to each sensor band
base_hyper = []
ozfuel_hyper = []
modis_hyper = []
sentinel_hyper = []
landsat_hyper = []
himawari_hyper = []

   
for band in ozfuel_bands:     
    temp = []    
    for wave in band:
        temp.append(bisect.bisect_left(wavelengths, wave))
    ozfuel_hyper.append(temp)

for band in modis_bands:
    temp = []     
    for wave in band:
        temp.append(bisect.bisect_left(wavelengths, wave))
    modis_hyper.append(temp)

for band in sentinel_bands:     
    temp = []     
    for wave in band:
        temp.append(bisect.bisect_left(wavelengths, wave))
    sentinel_hyper.append(temp)

for band in landsat_bands:     
    temp = []     
    for wave in band:
        temp.append(bisect.bisect_left(wavelengths, wave))
    landsat_hyper.append(temp)
    
for band in himawari_bands:     
    temp = []     
    for wave in band:
        temp.append(bisect.bisect_left(wavelengths, wave))
    himawari_hyper.append(temp)

#empty data cubes for each sensor
ozfuel_vari = []
modis_vari = []
sentinel_vari = []
landsat_vari = []
himawari_vari = []

ozfuel_std = []
modis_std = []
sentinel_std = []
landsat_std = []
himawari_std = []

# In[]: Preview Crop
img = np.sum(data, axis=0)

plt.figure(dpi=200)    
plt.imshow(img)
plt.imsave('imgs/swir/sum.png', img)

# In[]: Per band images and histograms

ozfuel_imgs = []

#Ozfuel
for band in ozfuel_hyper:
    resized_imgs = []
    for wave in band:
        resized_imgs.append( ski.resize(data[wave, :, :], (ozfuel_dims[0], ozfuel_dims[1]), preserve_range=True))
    cube = np.stack(resized_imgs, axis=2)
    raw_img = np.median(cube, axis=2)
    norm_img = (raw_img - np.min(raw_img)) / (np.max(raw_img) - np.min(raw_img))
    ozfuel_imgs.append(norm_img)
    ozfuel_vari.append(np.var(cube, axis=2))
    ozfuel_std.append(np.std(cube, axis=2))
    
#save simualted images  
# plt.figure(dpi=1200)
# plt.imshow(ozfuel_imgs[0], vmin=0.0, vmax=1.0)
# plt.savefig('imgs/ozfuel/sim_ozfuel_band_1.png', bbox_inches='tight')
# plt.figure(dpi=1200)
# plt.imshow(ozfuel_imgs[1], vmin=0.0, vmax=1.0)
# plt.savefig('imgs/ozfuel/sim_ozfuel_band_2.png', bbox_inches='tight')
# plt.figure(dpi=1200)
# plt.imshow(ozfuel_imgs[2], vmin=0.0, vmax=1.0)
# plt.savefig('imgs/ozfuel/sim_ozfuel_band_3.png', bbox_inches='tight')
# plt.figure(dpi=1200)
# plt.imshow(ozfuel_imgs[3], vmin=0.0, vmax=1.0)
# plt.savefig('imgs/ozfuel/sim_ozfuel_band_4.png', bbox_inches='tight')

#save variance maps
# plt.figure(dpi=1200)
# plt.imshow(ozfuel_vari[0])
# plt.savefig('imgs/ozfuel/vari_ozfuel_band_1.png', bbox_inches='tight')
# plt.figure(dpi=1200)
# plt.imshow(ozfuel_vari[1])
# plt.savefig('imgs/ozfuel/vari_ozfuel_band_2.png', bbox_inches='tight')
# plt.figure(dpi=1200)
# plt.imshow(ozfuel_vari[2])
# plt.savefig('imgs/ozfuel/vari_ozfuel_band_3.png', bbox_inches='tight')
# plt.figure(dpi=1200)
# plt.imshow(ozfuel_vari[3])
# plt.savefig('imgs/ozfuel/vari_ozfuel_band_4.png', bbox_inches='tight')

#save std maps
# plt.figure(dpi=1200)
# plt.imshow(ozfuel_std[0])
# plt.savefig('imgs/ozfuel/std_ozfuel_band_1.png', bbox_inches='tight')
# plt.figure(dpi=1200)
# plt.imshow(ozfuel_std[1])
# plt.savefig('imgs/ozfuel/std_ozfuel_band_2.png', bbox_inches='tight')
# plt.figure(dpi=1200)
# plt.imshow(ozfuel_std[2])
# plt.savefig('imgs/ozfuel/std_ozfuel_band_3.png', bbox_inches='tight')
# plt.figure(dpi=1200)
# plt.imshow(ozfuel_std[3])
# plt.savefig('imgs/ozfuel/std_ozfuel_band_4.png', bbox_inches='tight')

#img histograms and boxes
# count = 1
# while count < 5:
#     plt.figure(dpi=1200)
#     plt.title(f'OzFuel Band {count}')
#     plt.xlabel('Pixel value (0-1)')
#     plt.ylabel('Count')
#     hist = plt.hist(ozfuel_imgs[count-1].flatten(), bins)
#     plt.savefig(f'imgs/ozfuel/hist_ozfuel_band_{count}.png')
#     count += 1

# fig = plt.figure(dpi=1200)
# ax = fig.add_subplot(111)
# plt.boxplot([ozfuel_imgs[0].flatten(), ozfuel_imgs[1].flatten(), ozfuel_imgs[2].flatten(), ozfuel_imgs[3].flatten()])
# ax.set_xticklabels(['Band 1', 'Band 2', 'Band 3', 'Band 4'])
# ax.set_xlabel('Platform Bands')
# ax.set_ylabel('Pixel value')
# ax.set_title('OzFuel Bands')
# plt.savefig('imgs/ozfuel/box_ozfuel_bands.png')


#vari histograms and boxes
# count = 1
# while count < 5:
#     plt.figure(dpi=1200)
#     plt.title(f'Variance of OzFuel Band {count} Pixels')
#     plt.xlabel('Variance')
#     plt.ylabel('Count')
#     hist = plt.hist(ozfuel_vari[count-1].flatten())
#     plt.savefig(f'imgs/ozfuel/his_vari_ozfuel_band_{count}.png')
#     count += 1

# fig = plt.figure(dpi=1200)
# ax = fig.add_subplot(111)
# plt.boxplot([ozfuel_vari[0].flatten(), ozfuel_vari[1].flatten(), ozfuel_vari[2].flatten(), ozfuel_vari[3].flatten()])
# ax.set_xticklabels(['Band 1', 'Band 2', 'Band 3', 'Band 4'])
# ax.set_xlabel('Platform Bands')
# ax.set_ylabel('Variance')
# ax.set_title('Variance of OzFuel Bands')
# plt.savefig('imgs/ozfuel/box_vari_ozfuel_bands.png')

#std dev histograms and boxes
# count = 1
# while count < 5:
#     plt.figure(dpi=1200)
#     plt.title(f'Standard Deviation of OzFuel Band {count} Pixels')
#     plt.xlabel('Standard Deviation')
#     plt.ylabel('Count')
#     hist = plt.hist(ozfuel_std[count-1].flatten())
#     plt.savefig(f'imgs/ozfuel/hist_std_ozfuel_band_{count}.png')
#     count += 1

# fig = plt.figure(dpi=1200)
# ax = fig.add_subplot(111)
# plt.boxplot([ozfuel_std[0].flatten(), ozfuel_std[1].flatten(), ozfuel_std[2].flatten(), ozfuel_std[3].flatten()])
# ax.set_xticklabels(['Band 1', 'Band 2', 'Band 3', 'Band 4'])
# ax.set_xlabel('Platform Bands')
# ax.set_ylabel('Standard Deviation')
# ax.set_title('Standard Deviation of OzFuel Bands')
# plt.savefig('imgs/ozfuel/box_std_ozfuel_bands.png')
    


# In[] modis

modis_imgs = []

for band in modis_hyper:
    resized_imgs = []
    for wave in band:
        resized_imgs.append( ski.resize(data[wave, :, :], (modis_dims[0], modis_dims[1]), preserve_range=True))
    cube = np.stack(resized_imgs, axis=2)
    raw_img = np.median(cube, axis=2)
    norm_img = (raw_img - np.min(raw_img)) / (np.max(raw_img) - np.min(raw_img))
    # norm_img = raw_img / sensor_max
    modis_imgs.append(norm_img)
    modis_vari.append(np.var(cube, axis=2))
    modis_std.append(np.std(cube, axis=2))
      
#save simualted images  
# plt.figure(dpi=1200)
# plt.imshow(modis_imgs[0], vmin=0.0, vmax=1.0)
# plt.savefig('imgs/modis/sim_modis_band_4.png', bbox_inches='tight')
# plt.figure(dpi=1200)
# plt.imshow(modis_imgs[1], vmin=0.0, vmax=1.0)
# plt.savefig('imgs/modis/sim_modis_band_5.png', bbox_inches='tight')
# plt.figure(dpi=1200)
# plt.imshow(modis_imgs[2], vmin=0.0, vmax=1.0)
# plt.savefig('imgs/modis/sim_modis_band_6.png', bbox_inches='tight')


#save variance maps
# plt.figure(dpi=1200)
# plt.imshow(modis_vari[0])
# plt.savefig('imgs/modis/vari_modis_band_4.png', bbox_inches='tight')
# plt.figure(dpi=1200)
# plt.imshow(modis_vari[1])
# plt.savefig('imgs/modis/vari_modis_band_5.png', bbox_inches='tight')
# plt.figure(dpi=1200)
# plt.imshow(modis_vari[2])
# plt.savefig('imgs/modis/vari_modis_band_6.png', bbox_inches='tight')


#save std maps
# plt.figure(dpi=1200)
# plt.imshow(modis_std[0])
# plt.savefig('imgs/modis/std_modis_band_4.png', bbox_inches='tight')
# plt.figure(dpi=1200)
# plt.imshow(modis_std[1])
# plt.savefig('imgs/modis/std_modis_band_5.png', bbox_inches='tight')
# plt.figure(dpi=1200)
# plt.imshow(modis_std[2])
# plt.savefig('imgs/modis/std_modis_band_6.png', bbox_inches='tight')

#img histograms and boxes
# plt.figure(dpi=1200)
# plt.title('MODIS Band 5')
# plt.xlabel('Pixel value (0-1)')
# plt.ylabel('Count')
# hist = plt.hist(modis_imgs[0].flatten(), bins)
# plt.savefig('imgs/modis/hist_modis_band_5.png')

# plt.figure(dpi=1200)
# plt.title('MODIS Band 6')
# plt.xlabel('Pixel value (0-1)')
# plt.ylabel('Count')
# hist = plt.hist(modis_imgs[1].flatten(), bins)
# plt.savefig('imgs/modis/hist_modis_band_6.png')

# plt.figure(dpi=1200)
# plt.title('MODIS Band 7')
# plt.xlabel('Pixel value (0-1)')
# plt.ylabel('Count')
# hist = plt.hist(modis_imgs[2].flatten(), bins)
# plt.savefig('imgs/modis/hist_modis_band_7.png')


# fig = plt.figure(dpi=1200)
# ax = fig.add_subplot(111)
# plt.boxplot([modis_imgs[0].flatten(), modis_imgs[1].flatten(), modis_imgs[2].flatten()])
# ax.set_xticklabels(['Band 5', 'Band 6', 'Band 7'])
# ax.set_xlabel('Platform Bands')
# ax.set_ylabel('Pixel value')
# ax.set_title('MODIS SWIR Bands')
# plt.ylim(0, 1)
# plt.savefig('imgs/modis/box_modis_SWIR_bands.png')


#vari histograms and boxes
# count = 1
# while count < 5:
#     plt.figure(dpi=1200)
#     plt.title(f'Variance of modis Band {count} Pixels')
#     plt.xlabel('Variance')
#     plt.ylabel('Count')
#     hist = plt.hist(modis_vari[count-1].flatten())
#     plt.savefig(f'imgs/modis/his_vari_modis_band_{count}.png')
#     count += 1

# fig = plt.figure(dpi=1200)
# ax = fig.add_subplot(111)
# plt.boxplot([modis_vari[0].flatten(), modis_vari[1].flatten(), modis_vari[2].flatten()])
# ax.set_xticklabels(['Band 5', 'Band 6', 'Band 7'])
# ax.set_xlabel('Platform Bands')
# ax.set_ylabel('Variance')
# ax.set_title('Variance of MODIS Bands')
# plt.savefig('imgs/modis/box_vari_modis_bands.png')

#std dev histograms and boxes
# count = 1
# while count < 5:
#     plt.figure(dpi=1200)
#     plt.title(f'Standard Deviation of modis Band {count} Pixels')
#     plt.xlabel('Standard Deviation')
#     plt.ylabel('Count')
#     hist = plt.hist(modis_std[count-1].flatten())
#     plt.savefig(f'imgs/modis/hist_std_modis_band_{count}.png')
#     count += 1

# fig = plt.figure(dpi=1200)
# ax = fig.add_subplot(111)
# plt.boxplot([modis_std[0].flatten(), modis_std[1].flatten(), modis_std[2].flatten()])
# ax.set_xticklabels(['Band 5', 'Band 6', 'Band 7'])
# ax.set_xlabel('Platform Bands')
# ax.set_ylabel('Standard Deviation')
# ax.set_title('Standard Deviation of MODIS Bands')
# plt.savefig('imgs/modis/box_std_modis_bands.png')

# In[] Sentinel band sims, hists, and box
sentinel_imgs = []

for band in sentinel_hyper:
    resized_imgs = []
    for wave in band:
        resized_imgs.append( ski.resize(data[wave, :, :], (sentinel_dims[0], sentinel_dims[1]), preserve_range=True))
    cube = np.stack(resized_imgs, axis=2)
    raw_img = np.median(cube, axis=2)
    norm_img = (raw_img - np.min(raw_img)) / (np.max(raw_img) - np.min(raw_img))
    sentinel_imgs.append(norm_img)
    sentinel_vari.append(np.var(cube, axis=2))
    sentinel_std.append(np.std(cube, axis=2))
      
#save simualted images  
# plt.figure(dpi=1200)
# plt.imshow(sentinel_imgs[0], vmin=0.0, vmax=1.0)
# plt.savefig('imgs/sentinel/sim_sentinel_band_10.png', bbox_inches='tight')
# plt.figure(dpi=1200)
# plt.imshow(sentinel_imgs[1], vmin=0.0, vmax=1.0)
# plt.savefig('imgs/sentinel/sim_sentinel_band_12.png', bbox_inches='tight')

#save variance maps
# plt.figure(dpi=1200)
# plt.imshow(sentinel_vari[0])
# plt.savefig('imgs/sentinel/vari_sentinel_band_10.png', bbox_inches='tight')
# plt.figure(dpi=1200)
# plt.imshow(sentinel_vari[1])
# plt.savefig('imgs/sentinel/vari_sentinel_band_12.png', bbox_inches='tight')

#save std maps
# plt.figure(dpi=1200)
# plt.imshow(sentinel_std[0])
# plt.savefig('imgs/sentinel/std_sentinel_band_10.png', bbox_inches='tight')
# plt.figure(dpi=1200)
# plt.imshow(sentinel_std[1])
# plt.savefig('imgs/sentinel/std_sentinel_band_12.png', bbox_inches='tight')

#img histograms and boxes
# plt.figure(dpi=1200)
# plt.title('Sentinel-2 Band 10')
# plt.xlabel('Pixel value (0-1)')
# plt.ylabel('Count')
# hist = plt.hist(sentinel_imgs[0].flatten(), bins)
# plt.savefig('imgs/sentinel/hist_sentinel_band_10.png')

# plt.figure(dpi=1200)
# plt.title('Sentinel-2 Band 12')
# plt.xlabel('Pixel value (0-1)')
# plt.ylabel('Count')
# hist = plt.hist(sentinel_imgs[1].flatten(), bins)
# plt.savefig('imgs/sentinel/hist_sentinel_band_12.png')

# fig = plt.figure(dpi=1200)
# ax = fig.add_subplot(111)
# plt.boxplot([sentinel_imgs[0].flatten(), sentinel_imgs[1].flatten()])
# ax.set_xticklabels(['Band 10', 'Band 12'])
# ax.set_xlabel('Platform Bands')
# ax.set_ylabel('Pixel value')
# ax.set_title('Sentinel-2 SWIR Bands')
# plt.ylim(0, 1)
# plt.savefig('imgs/sentinel/box_sentinel_SWIR_bands.png')


#vari histograms and boxes
# count = 1
# while count < 5:
#     plt.figure(dpi=1200)
#     plt.title(f'Variance of sentinel Band {count} Pixels')
#     plt.xlabel('Variance')
#     plt.ylabel('Count')
#     hist = plt.hist(sentinel_vari[count-1].flatten())
#     plt.savefig(f'imgs/sentinel/his_vari_sentinel_band_{count}.png')
#     count += 1

# fig = plt.figure(dpi=1200)
# ax = fig.add_subplot(111)
# plt.boxplot([sentinel_vari[0].flatten(), sentinel_vari[1].flatten()])
# ax.set_xticklabels(['Band 10', 'Band 12'])
# ax.set_xlabel('Platform Bands')
# ax.set_ylabel('Variance')
# ax.set_title('Variance of Sentinel-2 Bands')
# plt.savefig('imgs/sentinel/box_vari_sentinel_bands.png')

#std dev histograms and boxes
# count = 1
# while count < 5:
#     plt.figure(dpi=1200)
#     plt.title(f'Standard Deviation of sentinel Band {count} Pixels')
#     plt.xlabel('Standard Deviation')
#     plt.ylabel('Count')
#     hist = plt.hist(sentinel_std[count-1].flatten())
#     plt.savefig(f'imgs/sentinel/hist_std_sentinel_band_{count}.png')
#     count += 1

# fig = plt.figure(dpi=1200)
# ax = fig.add_subplot(111)
# plt.boxplot([sentinel_std[0].flatten(), sentinel_std[1].flatten()])
# ax.set_xticklabels(['Band 10', 'Band 12'])
# ax.set_xlabel('Platform Bands')
# ax.set_ylabel('Standard Deviation')
# ax.set_title('Standard Deviation of Sentinel-2 Bands')
# plt.savefig('imgs/sentinel/box_std_sentinel_bands.png')

# In[] Landsat
landsat_imgs = []

for band in landsat_hyper:
    resized_imgs = []
    for wave in band:
        resized_imgs.append( ski.resize(data[wave, :, :], (landsat_dims[0], landsat_dims[1]), preserve_range=True))
    cube = np.stack(resized_imgs, axis=2)
    raw_img = np.median(cube, axis=2)
    norm_img = (raw_img - np.min(raw_img)) / (np.max(raw_img) - np.min(raw_img))
    landsat_imgs.append(norm_img)
    landsat_vari.append(np.var(cube, axis=2))
    landsat_std.append(np.std(cube, axis=2))
      
#save simualted images  
# plt.figure(dpi=1200)
# plt.imshow(landsat_imgs[0], vmin=0.0, vmax=1.0)
# plt.savefig('imgs/landsat/sim_landsat_band_6.png', bbox_inches='tight')
# plt.figure(dpi=1200)
# plt.imshow(landsat_imgs[1], vmin=0.0, vmax=1.0)
# plt.savefig('imgs/landsat/sim_landsat_band_7.png', bbox_inches='tight')

#save variance maps
# plt.figure(dpi=1200)
# plt.imshow(landsat_vari[0])
# plt.savefig('imgs/landsat/vari_landsat_band_6.png', bbox_inches='tight')
# plt.figure(dpi=1200)
# plt.imshow(landsat_vari[1])
# plt.savefig('imgs/landsat/vari_landsat_band_7.png', bbox_inches='tight')

#save std maps
# plt.figure(dpi=1200)
# plt.imshow(landsat_std[0])
# plt.savefig('imgs/landsat/std_landsat_band_6.png', bbox_inches='tight')
# plt.figure(dpi=1200)
# plt.imshow(landsat_std[1])
# plt.savefig('imgs/landsat/std_landsat_band_7.png', bbox_inches='tight')

#img histograms and boxes
# plt.figure(dpi=1200)
# plt.title('Landsat-8 Band 6')
# plt.xlabel('Pixel value (0-1)')
# plt.ylabel('Count')
# hist = plt.hist(landsat_imgs[0].flatten(), bins)
# plt.savefig('imgs/landsat/hist_landsat_band_6.png')

# plt.figure(dpi=1200)
# plt.title('Landsat-8 Band 7')
# plt.xlabel('Pixel value (0-1)')
# plt.ylabel('Count')
# hist = plt.hist(landsat_imgs[1].flatten(), bins)
# plt.savefig('imgs/landsat/hist_landsat_band_7.png')

# fig = plt.figure(dpi=1200)
# ax = fig.add_subplot(111)
# plt.boxplot([landsat_imgs[0].flatten(), landsat_imgs[1].flatten()])
# ax.set_xticklabels(['Band 6', 'Band 7'])
# ax.set_xlabel('Platform Bands')
# ax.set_ylabel('Pixel value')
# ax.set_title('Landsat-8 SWIR Bands')
# plt.ylim(0, 1)
# plt.savefig('imgs/landsat/box_landsat_SWIR_bands.png')


#vari histograms and boxes
# count = 1
# while count < 5:
#     plt.figure(dpi=1200)
#     plt.title(f'Variance of landsat Band {count} Pixels')
#     plt.xlabel('Variance')
#     plt.ylabel('Count')
#     hist = plt.hist(landsat_vari[count-1].flatten())
#     plt.savefig(f'imgs/landsat/his_vari_landsat_band_{count}.png')
#     count += 1

# fig = plt.figure(dpi=1200)
# ax = fig.add_subplot(111)
# plt.boxplot([landsat_vari[0].flatten(), landsat_vari[1].flatten()])
# ax.set_xticklabels(['Band 6', 'Band 7'])
# ax.set_xlabel('Platform Bands')
# ax.set_ylabel('Variance')
# ax.set_title('Variance of Landsat-8 Bands')
# plt.savefig('imgs/landsat/box_vari_landsat_bands.png')

#std dev histograms and boxes
# count = 1
# while count < 5:
#     plt.figure(dpi=1200)
#     plt.title(f'Standard Deviation of landsat Band {count} Pixels')
#     plt.xlabel('Standard Deviation')
#     plt.ylabel('Count')
#     hist = plt.hist(landsat_std[count-1].flatten())
#     plt.savefig(f'imgs/landsat/hist_std_landsat_band_{count}.png')
#     count += 1

# fig = plt.figure(dpi=1200)
# ax = fig.add_subplot(111)
# plt.boxplot([landsat_std[0].flatten(), landsat_std[1].flatten()])
# ax.set_xticklabels(['Band 6', 'Band 7'])
# ax.set_xlabel('Platform Bands')
# ax.set_ylabel('Standard Deviation')
# ax.set_title('Standard Deviation of Landsat-8 Bands')
# plt.savefig('imgs/landsat/box_std_landsat_bands.png')

# In[]   Himawari
himawari_imgs = []

for band in himawari_hyper:
    resized_imgs = []
    for wave in band:
        resized_imgs.append( ski.resize(data[wave, :, :], (himawari_dims[0], himawari_dims[1]), preserve_range=True))
    cube = np.stack(resized_imgs, axis=2)
    raw_img = np.median(cube, axis=2)
    # norm_img = (raw_img - np.min(raw_img)) / (np.max(raw_img) - np.min(raw_img))
    norm_img = raw_img / sensor_max
    himawari_imgs.append(norm_img)
    himawari_vari.append(np.var(cube, axis=2))
    himawari_std.append(np.std(cube, axis=2))
      
#save simualted images  
# plt.figure(dpi=1200)
# plt.imshow(himawari_imgs[0], vmin=0.0, vmax=1.0)
# plt.savefig('imgs/himawari/sim_himawari_band_5.png', bbox_inches='tight')
# plt.figure(dpi=1200)
# plt.imshow(himawari_imgs[1], vmin=0.0, vmax=1.0)
# plt.savefig('imgs/himawari/sim_himawari_band_6.png', bbox_inches='tight')

#save variance maps
# plt.figure(dpi=1200)
# plt.imshow(himawari_vari[0])
# plt.savefig('imgs/himawari/vari_himawari_band_5.png', bbox_inches='tight')
# plt.figure(dpi=1200)
# plt.imshow(himawari_vari[1])
# plt.savefig('imgs/himawari/vari_himawari_band_6.png', bbox_inches='tight')

#save std maps
# plt.figure(dpi=1200)
# plt.imshow(himawari_std[0])
# plt.savefig('imgs/himawari/std_himawari_band_5.png', bbox_inches='tight')
# plt.figure(dpi=1200)
# plt.imshow(himawari_std[1])
# plt.savefig('imgs/himawari/std_himawari_band_6.png', bbox_inches='tight')

#img histograms and boxes
# plt.figure(dpi=1200)
# plt.title('Himawari-8 Band 5')
# plt.xlabel('Pixel value (0-1)')
# plt.ylabel('Count')
# hist = plt.hist(himawari_imgs[0].flatten(), bins)
# plt.savefig('imgs/himawari/hist_himawari_band_6.png')

# plt.figure(dpi=1200)
# plt.title('Himawari-8 Band 6')
# plt.xlabel('Pixel value (0-1)')
# plt.ylabel('Count')
# hist = plt.hist(himawari_imgs[1].flatten(), bins)
# plt.savefig('imgs/himawari/hist_himawari_band_7.png')

# fig = plt.figure(dpi=1200)
# ax = fig.add_subplot(111)
# plt.boxplot([himawari_imgs[0].flatten(), himawari_imgs[1].flatten()])
# ax.set_xticklabels(['Band 5', 'Band 6'])
# ax.set_xlabel('Platform Bands')
# ax.set_ylabel('Pixel value')
# ax.set_title('Himawari-8 SWIR Bands')
# plt.ylim(0, 1)
# plt.savefig('imgs/himawari/box_himawari_SWIR_bands.png')


#vari histograms and boxes
# count = 1
# while count < 5:
#     plt.figure(dpi=1200)
#     plt.title(f'Variance of himawari Band {count} Pixels')
#     plt.xlabel('Variance')
#     plt.ylabel('Count')
#     hist = plt.hist(himawari_vari[count-1].flatten())
#     plt.savefig(f'imgs/himawari/his_vari_himawari_band_{count}.png')
#     count += 1

# fig = plt.figure(dpi=1200)
# ax = fig.add_subplot(111)
# plt.boxplot([himawari_vari[0].flatten(), himawari_vari[1].flatten()])
# ax.set_xticklabels(['Band 5', 'Band 6'])
# ax.set_xlabel('Platform Bands')
# ax.set_ylabel('Variance')
# ax.set_title('Variance of Himawari-8 Bands')
# plt.savefig('imgs/himawari/box_vari_himawari_bands.png')

#std dev histograms and boxes
# count = 1
# while count < 5:
#     plt.figure(dpi=1200)
#     plt.title(f'Standard Deviation of himawari Band {count} Pixels')
#     plt.xlabel('Standard Deviation')
#     plt.ylabel('Count')
#     hist = plt.hist(himawari_std[count-1].flatten())
#     plt.savefig(f'imgs/himawari/hist_std_himawari_band_{count}.png')
#     count += 1

# fig = plt.figure(dpi=1200)
# ax = fig.add_subplot(111)
# plt.boxplot([himawari_std[0].flatten(), himawari_std[1].flatten()])
# ax.set_xticklabels(['Band 5', 'Band 6'])
# ax.set_xlabel('Platform Bands')
# ax.set_ylabel('Standard Deviation')
# ax.set_title('Standard Deviation of Himawari-8 Bands')
# plt.savefig('imgs/himawari/box_std_himawari_bands.png')


# In[]: Img box plots

# #Ozfuel band 1
# fig = plt.figure(dpi=1200)
# ax = fig.add_subplot(111)
# plt.boxplot([ozfuel_imgs[0].flatten(), modis_imgs[0].flatten(), sentinel_imgs[0].flatten()])
# ax.set_xticklabels(['OzFuel 1', 'MODIS 5', 'Sentinel-2 10'])
# ax.set_xlabel('Platform Bands')
# ax.set_ylabel('Pixel value')
# ax.set_title('OzFuel Band 1 vs other bands')
# plt.savefig('imgs/swir/box_OzFuel_band_1.png')

# #Ozfuel band 2
# fig = plt.figure(dpi=1200)
# ax = fig.add_subplot(111)
# plt.boxplot([ozfuel_imgs[1].flatten(), modis_imgs[1].flatten(), landsat_imgs[0].flatten(), himawari_imgs[0].flatten()])
# ax.set_xticklabels(['OzFuel 2', 'MODIS 6', 'Landsat-8 6', 'Himawari-8 5'])
# ax.set_xlabel('Platform Bands')
# ax.set_ylabel('Pixel value')
# ax.set_title('OzFuel Band 2 vs other bands')
# plt.savefig('imgs/swir/box_OzFuel_band_2.png')

# #Ozfuel band 3
# fig = plt.figure(dpi=1200)
# ax = fig.add_subplot(111)
# plt.boxplot([ozfuel_imgs[2].flatten(), modis_imgs[2].flatten()])
# ax.set_xticklabels(['OzFuel 3', 'MODIS 7'])
# ax.set_xlabel('Platform Bands')
# ax.set_ylabel('Pixel value')
# ax.set_title('OzFuel Band 3 vs other bands')
# plt.savefig('imgs/swir/box_OzFuel_band_3.png')

# #Ozfuel band 1
# fig = plt.figure(dpi=1200)
# ax = fig.add_subplot(111)
# plt.boxplot([ozfuel_imgs[3].flatten(), sentinel_imgs[1].flatten(), landsat_imgs[1].flatten(), himawari_imgs[1].flatten()])
# ax.set_xticklabels(['OzFuel 4', 'Sentinel-2 12', 'Landsat-8 7', 'Himawari-8 6'])
# ax.set_xlabel('Platform Bands')
# ax.set_ylabel('Pixel value')
# ax.set_title('OzFuel Band 4 vs other bands')
# plt.savefig('imgs/swir/box_OzFuel_band_4.png')


# In[]: Vari box plots

# #Ozfuel band 1
# fig = plt.figure(dpi=1200)
# ax = fig.add_subplot(111)
# plt.boxplot([ozfuel_vari[0].flatten(), modis_vari[0].flatten(), sentinel_vari[0].flatten()])
# ax.set_xticklabels(['OzFuel 1', 'MODIS 5', 'Sentinel-2 10'])
# ax.set_xlabel('Platform Bands')
# ax.set_ylabel('Pixel variance')
# ax.set_title('OzFuel Band 1 vs other bands')
# plt.savefig('imgs/swir/box_vari_OzFuel_band_1.png')

# #Ozfuel band 2
# fig = plt.figure(dpi=1200)
# ax = fig.add_subplot(111)
# plt.boxplot([ozfuel_vari[1].flatten(), modis_vari[1].flatten(), landsat_vari[0].flatten(), himawari_vari[0].flatten()])
# ax.set_xticklabels(['OzFuel 2', 'MODIS 6', 'Landsat-8 6', 'Himawari-8 5'])
# ax.set_xlabel('Platform Bands')
# ax.set_ylabel('Pixel variance')
# ax.set_title('OzFuel Band 2 vs other bands')
# plt.savefig('imgs/swir/box_vari_OzFuel_band_2.png')

# #Ozfuel band 3
# fig = plt.figure(dpi=1200)
# ax = fig.add_subplot(111)
# plt.boxplot([ozfuel_vari[2].flatten(), modis_vari[2].flatten()])
# ax.set_xticklabels(['OzFuel 3', 'MODIS 7'])
# ax.set_xlabel('Platform Bands')
# ax.set_ylabel('Pixel variance')
# ax.set_title('OzFuel Band 3 vs other bands')
# plt.savefig('imgs/swir/box_vari_OzFuel_band_3.png')

# #Ozfuel band 1
# fig = plt.figure(dpi=1200)
# ax = fig.add_subplot(111)
# plt.boxplot([ozfuel_vari[3].flatten(), sentinel_vari[1].flatten(), landsat_vari[1].flatten(), himawari_vari[1].flatten()])
# ax.set_xticklabels(['OzFuel 4', 'Sentinel-2 12', 'Landsat-8 7', 'Himawari-8 6'])
# ax.set_xlabel('Platform Bands')
# ax.set_ylabel('Pixel variance')
# ax.set_title('OzFuel Band 4 vs other bands')
# plt.savefig('imgs/swir/box_vari_OzFuel_band_4.png')



# In[]: Std box plots

# #Ozfuel band 1
# fig = plt.figure(dpi=1200)
# ax = fig.add_subplot(111)
# plt.boxplot([ozfuel_std[0].flatten(), modis_std[0].flatten(), sentinel_std[0].flatten()])
# ax.set_xticklabels(['OzFuel 1', 'MODIS 5', 'Sentinel-2 10'])
# ax.set_xlabel('Platform Bands')
# ax.set_ylabel('Pixel Standard Deviation')
# ax.set_title('OzFuel Band 1 vs other bands')
# plt.savefig('imgs/swir/box_std_OzFuel_band_1.png')

# #Ozfuel band 2
# fig = plt.figure(dpi=1200)
# ax = fig.add_subplot(111)
# plt.boxplot([ozfuel_std[1].flatten(), modis_std[1].flatten(), landsat_std[0].flatten(), himawari_std[0].flatten()])
# ax.set_xticklabels(['OzFuel 2', 'MODIS 6', 'Landsat-8 6', 'Himawari-8 5'])
# ax.set_xlabel('Platform Bands')
# ax.set_ylabel('Pixel Standard Deviation')
# ax.set_title('OzFuel Band 2 vs other bands')
# plt.savefig('imgs/swir/box_std_OzFuel_band_2.png')

# #Ozfuel band 3
# fig = plt.figure(dpi=1200)
# ax = fig.add_subplot(111)
# plt.boxplot([ozfuel_std[2].flatten(), modis_std[2].flatten()])
# ax.set_xticklabels(['OzFuel 3', 'MODIS 7'])
# ax.set_xlabel('Platform Bands')
# ax.set_ylabel('Pixel Standard Deviation')
# ax.set_title('OzFuel Band 3 vs other bands')
# plt.savefig('imgs/swir/box_std_OzFuel_band_3.png')

# #Ozfuel band 1
# fig = plt.figure(dpi=1200)
# ax = fig.add_subplot(111)
# plt.boxplot([ozfuel_std[3].flatten(), sentinel_std[1].flatten(), landsat_std[1].flatten(), himawari_std[1].flatten()])
# ax.set_xticklabels(['OzFuel 4', 'Sentinel-2 12', 'Landsat-8 7', 'Himawari-8 6'])
# ax.set_xlabel('Platform Bands')
# ax.set_ylabel('Pixel Standard Deviation')
# ax.set_title('OzFuel Band 4 vs other bands')
# plt.savefig('imgs/swir/box_std_OzFuel_band_4.png')


# In[]: Comparison with measured spectra
plt.rcParams.update({'errorbar.capsize': 5}) #set error bar cap size in advance

#generate spectral signature of SWIR sensor and normalise for comparison
base_hyper = []
for band in base_bands:        
    base_hyper.append(bisect.bisect_left(wavelengths, band))
swir_spectra = []
for band in base_hyper:
    raw_img = data[band, :, :]
    norm_img = (raw_img - np.min(raw_img)) / (np.max(raw_img) - np.min(raw_img))
    swir_spectra.append(np.median(norm_img))  

#import measured spectra and remove noise
measured_spectra = pd.read_csv('C:/Users/Sam/Google Drive/UNI/2022/ENGN4350/Python/MAC_ARB_098_044_210422_L00001.asd.txt', sep='\t')
measured_spectra.set_index('Wavelength', inplace=True)
measured_spectra = measured_spectra.truncate(before=400, after=2400)

#compute band centres, spectra, and error bars
ozfuel_ctrs = [np.mean(ozfuel_bands[0]), np.mean(ozfuel_bands[1]), np.mean(ozfuel_bands[2]), np.mean(ozfuel_bands[3])]
ozfuel_spectra = [np.median(ozfuel_imgs[0]), np.median(ozfuel_imgs[1]), np.median(ozfuel_imgs[2]), np.median(ozfuel_imgs[3])]
ozfuel_err = [np.std(ozfuel_imgs[0]), np.std(ozfuel_imgs[1]), np.std(ozfuel_imgs[2]), np.std(ozfuel_imgs[3])]

modis_ctrs = [np.mean(modis_bands[0]), np.mean(modis_bands[1]), np.mean(modis_bands[2])]
modis_spectra = [np.median(modis_imgs[0]), np.median(modis_imgs[1]), np.median(modis_imgs[2])]
modis_err = [np.std(modis_imgs[0]), np.std(modis_imgs[1]), np.std(modis_imgs[2])]

sentinel_ctrs = [np.mean(sentinel_bands[0]), np.mean(sentinel_bands[1])]
sentinel_spectra = [np.median(sentinel_imgs[0]), np.median(sentinel_imgs[1])]
sentinel_err = [np.std(sentinel_imgs[0]), np.std(sentinel_imgs[1])]

landsat_ctrs = [np.mean(landsat_bands[0]), np.mean(landsat_bands[1])]
landsat_spectra = [np.median(landsat_imgs[0]), np.median(landsat_imgs[1])]
landsat_err = [np.std(landsat_imgs[0]), np.std(landsat_imgs[1])]

himawari_ctrs = [np.mean(himawari_bands[0]), np.mean(himawari_bands[1])]
himawari_spectra = [np.median(himawari_imgs[0]), np.median(himawari_imgs[1])]
himawari_err = [np.std(himawari_imgs[0]), np.std(himawari_imgs[1])]

#Plot measured and simulated spectra 
fig, ax = plt.subplots(nrows=5, ncols=1, sharex=True, sharey=True, dpi=200, figsize=(8,14))
ax[0].set_ylim([-0.1,1.1])
plt.subplot(5,1,1)
plt.plot(range(400,2401), measured_spectra.values.tolist())
plt.errorbar(ozfuel_ctrs, ozfuel_spectra, yerr=ozfuel_err, fmt='rx-')
plt.plot(base_bands, swir_spectra, 'g')
plt.legend(['Measured Spectra', 'SWIR Sensor', 'OzFuel'])
plt.title('Comparison of measured eucalpyt spectra and simulated spectra from imagery')

plt.subplot(5,1,2)
plt.plot(range(400,2401), measured_spectra.values.tolist())
plt.errorbar(modis_ctrs, modis_spectra, yerr=modis_err, fmt='cx-')
plt.plot(base_bands, swir_spectra, 'g')
plt.legend(['Measured Spectra', 'SWIR Sensor', 'MODIS'])

plt.subplot(5,1,3)
plt.plot(range(400,2401), measured_spectra.values.tolist())
plt.errorbar(sentinel_ctrs, sentinel_spectra, yerr=sentinel_err, fmt='yx-')
plt.plot(base_bands, swir_spectra, 'g')
plt.legend(['Measured Spectra', 'SWIR Sensor', 'Sentinel-2'])

plt.subplot(5,1,4)
plt.plot(range(400,2401), measured_spectra.values.tolist())
plt.errorbar(landsat_ctrs, landsat_spectra, yerr=landsat_err, fmt='mx-')
plt.plot(base_bands, swir_spectra, 'g')
plt.legend(['Measured Spectra', 'SWIR Sensor', 'Landsat-8'])

plt.subplot(5,1,5)
plt.plot(range(400,2401), measured_spectra.values.tolist())
plt.errorbar(himawari_ctrs, himawari_spectra, yerr=himawari_err, fmt='x-')
plt.plot(base_bands, swir_spectra, 'g')
plt.legend(['Measured Spectra', 'SWIR Sensor', 'Himawari-8'])
ax[4].set(xlabel='Wavelength')

for ax in ax.flat:
    ax.set(ylabel='Pixel Value/Reflectiveness')

#save figure for external processing
plt.savefig('imgs/swir/spectra.png', bbox_inches='tight')


# In[]:
# data_file.close()
end = time.time()
print(end-start)