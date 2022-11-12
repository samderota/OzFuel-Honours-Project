# In[]: Init and open data file

import numpy as np
import matplotlib.pyplot as plt
import bisect
import time
import os 
import skimage.transform as ski 
from scipy import stats

os.chdir('C:/Users/Sam/Google Drive/UNI/2022/ENGN4350/Python/')

start = time.time()

wavelengths = [-8.6306, 0.9195025, 10.469605, 20.0197075, 29.56981, 39.1199125, 48.670015, 58.2201175, 67.77022, 77.3203225, 86.870425, 96.4205275, 105.97063, 115.5207325, 125.070835, 134.6209375, 144.17104, 153.7211425, 163.271245, 172.8213475, 182.37145, 191.9215525, 201.471655, 211.0217575, 220.57186, 230.1219625, 239.672065, 249.2221675, 258.77227, 268.3223725, 277.872475, 287.4225775, 296.97268, 306.5227825, 316.072885, 325.6229875, 335.17309, 344.7231925, 354.273295, 363.8233975, 373.3735, 382.9236025, 392.473705, 402.0238075, 411.57391, 421.1240125, 430.674115, 440.2242175, 449.77432, 459.3244225, 468.874525, 478.4246275, 487.97473, 497.5248325, 507.074935, 516.6250375, 526.17514, 535.7252425, 545.275345, 554.8254475, 564.37555, 573.9256525, 583.475755, 593.0258575, 602.57596, 612.1260625, 621.676165, 631.2262675, 640.77637, 650.3264725, 659.876575, 669.4266775, 678.97678, 688.5268825, 698.076985, 707.6270875, 717.17719, 726.7272925, 736.277395, 745.8274975, 755.3776, 764.9277025, 774.477805, 784.0279075, 793.57801, 803.1281125, 812.678215, 822.2283175, 831.77842, 841.3285225, 850.878625, 860.4287275, 869.97883, 879.5289325, 889.079035, 898.6291375, 908.17924, 917.7293425, 927.279445, 936.8295475, 946.37965, 955.9297525, 965.479855, 975.0299575, 984.58006, 994.1301625, 1003.680265, 1013.2303675, 1022.78047, 1032.3305725, 1041.880675, 1051.4307775, 1060.98088, 1070.5309825, 1080.081085, 1089.6311875, 1099.18129, 1108.7313925, 1118.281495, 1127.8315975, 1137.3817, 1146.9318025, 1156.481905, 1166.0320075, 1175.58211, 1185.1322125, 1194.682315, 1204.2324175, 1213.78252, 1223.3326225, 1232.882725, 1242.4328275, 1251.98293, 1261.5330325, 1271.083135, 1280.6332375, 1290.18334, 1299.7334425, 1309.283545, 1318.8336475, 1328.38375, 1337.9338525, 1347.483955, 1357.0340575, 1366.58416, 1376.1342625, 1385.684365, 1395.2344675, 1404.78457, 1414.3346725, 1423.884775, 1433.4348775, 1442.98498, 1452.5350825, 1462.085185, 1471.6352875, 1481.18539, 1490.7354925, 1500.285595, 1509.8356975, 1519.3858, 1528.9359025, 1538.486005, 1548.0361075, 1557.58621, 1567.1363125, 1576.686415, 1586.2365175, 1595.78662, 1605.3367225, 1614.886825, 1624.4369275, 1633.98703, 1643.5371325, 1653.087235, 1662.6373375, 1672.18744, 1681.7375425, 1691.287645, 1700.8377475, 1710.38785, 1719.9379525, 1729.488055, 1739.0381575, 1748.58826, 1758.1383625, 1767.688465, 1777.2385675, 1786.78867, 1796.3387725, 1805.888875, 1815.4389775, 1824.98908, 1834.5391825, 1844.089285, 1853.6393875, 1863.18949, 1872.7395925, 1882.289695, 1891.8397975, 1901.3899, 1910.9400025, 1920.490105, 1930.0402075, 1939.59031, 1949.1404125, 1958.690515, 1968.2406175, 1977.79072, 1987.3408225, 1996.890925, 2006.4410275, 2015.99113, 2025.5412325, 2035.091335, 2044.6414375, 2054.19154, 2063.7416425, 2073.291745, 2082.8418475, 2092.39195, 2101.9420525, 2111.492155, 2121.0422575, 2130.59236, 2140.1424625, 2149.692565, 2159.2426675, 2168.79277, 2178.3428725, 2187.892975, 2197.4430775, 2206.99318, 2216.5432825, 2226.093385, 2235.6434875, 2245.19359, 2254.7436925, 2264.293795, 2273.8438975, 2283.394, 2292.9441025, 2302.494205, 2312.0443075, 2321.59441, 2331.1445125, 2340.694615, 2350.2447175, 2359.79482, 2369.3449225, 2378.895025, 2388.4451275, 2397.99523, 2407.5453325, 2417.095435, 2426.6455375, 2436.19564, 2445.7457425, 2455.295845, 2464.8459475, 2474.39605, 2483.9461525, 2493.496255, 2503.0463575, 2512.59646, 2522.1465625, 2531.696665, 2541.2467675, 2550.79687, 2560.3469725, 2569.897075, 2579.4471775, 2588.99728, 2598.5473825, 2608.097485, 2617.6475875, 2627.19769, 2636.7477925, 2646.297895, 2655.8479975, 2665.3981, 2674.9482025, 2684.498305, 2694.0484075, 2703.59851, 2713.1486125, 2722.698715, 2732.2488175]
bands = len(wavelengths)
sensor_max = 2**16  

#nested list containing each band, note - only contains SWIR bands for comparison
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


# In[]: OzFuel band 1

ozfuel_band1_spectra = []


#File 1
file_1 = "C:/Users/Sam/Google Drive/UNI/2022/ENGN4350/Python/FMC/20220228_Reorder_288_2349_384_U16.bin"       
with open(file_1) as f:
    dt = np.dtype('<u2')   
    rawdata = np.fromfile(f, dtype=dt)

length = 2349       #file specifc
height = 384
rawdata = np.reshape(rawdata, (len(wavelengths), length, height))  
master_data_1 = np.rot90(rawdata, 1, (1,2))

# cropping
mac1 = master_data_1[:, 49:49+333, 1293:1293+178] 
mac2 = master_data_1[:, 41:41+333, 1068:1068+178] 
mac3 = master_data_1[:, 49:49+334, 899:899+124] 
mac4 = master_data_1[:, 102:102+276, 791:791+85] 
lengths = [mac1.shape[2], mac2.shape[2], mac3.shape[2], mac4.shape[2]] 
heights = [mac1.shape[1], mac2.shape[1], mac3.shape[1], mac4.shape[1]] 

#calculate dimensions of simulated images according to scaling factor
ozfuel_dims_mac1 = [int(heights[0]/ozfuel_scaling), int(lengths[0]/ozfuel_scaling)]
ozfuel_dims_mac2 = [int(heights[1]/ozfuel_scaling), int(lengths[1]/ozfuel_scaling)]
ozfuel_dims_mac3 = [int(heights[2]/ozfuel_scaling), int(lengths[2]/ozfuel_scaling)]
ozfuel_dims_mac4 = [int(heights[3]/ozfuel_scaling), int(lengths[3]/ozfuel_scaling)]

measured_fmc_file1 = [106.92, 112.29, 76.245, 100.335]      #fuel moisture content of cropped region (float)              

# OzFuel MAC 1
resized_imgs = []
for wave in ozfuel_hyper[0]:
    resized_imgs.append( ski.resize(mac1[wave, :, :], (ozfuel_dims_mac1[0], ozfuel_dims_mac1[1]), preserve_range=True))
cube = np.stack(resized_imgs, axis=2)
raw_img = np.median(cube, axis=2)
norm_img = (raw_img - np.min(raw_img)) / (np.max(raw_img) - np.min(raw_img))
ozfuel_band1_spectra.append(np.median(norm_img))
# OzFuel MAC 2
resized_imgs = []
for wave in ozfuel_hyper[0]:
    resized_imgs.append( ski.resize(mac2[wave, :, :], (ozfuel_dims_mac2[0], ozfuel_dims_mac2[1]), preserve_range=True))
cube = np.stack(resized_imgs, axis=2)
raw_img = np.median(cube, axis=2)
norm_img = (raw_img - np.min(raw_img)) / (np.max(raw_img) - np.min(raw_img))
ozfuel_band1_spectra.append(np.median(norm_img))   
# OzFuel MAC 3
resized_imgs = []
for wave in ozfuel_hyper[0]:
    resized_imgs.append( ski.resize(mac3[wave, :, :], (ozfuel_dims_mac3[0], ozfuel_dims_mac3[1]), preserve_range=True))
cube = np.stack(resized_imgs, axis=2)
raw_img = np.median(cube, axis=2)
norm_img = (raw_img - np.min(raw_img)) / (np.max(raw_img) - np.min(raw_img))
ozfuel_band1_spectra.append(np.median(norm_img))
# OzFuel MAC 4
resized_imgs = []
for wave in ozfuel_hyper[0]:
    resized_imgs.append( ski.resize(mac4[wave, :, :], (ozfuel_dims_mac4[0], ozfuel_dims_mac4[1]), preserve_range=True))
cube = np.stack(resized_imgs, axis=2)
raw_img = np.median(cube, axis=2)
norm_img = (raw_img - np.min(raw_img)) / (np.max(raw_img) - np.min(raw_img))
ozfuel_band1_spectra.append(np.median(norm_img))
  


  
#File 2
file_2 = "C:/Users/Sam/Google Drive/UNI/2022/ENGN4350/Python/FMC/20211206_Reorder_288_2108_384_U16.bin"       
with open(file_2) as f:
    dt = np.dtype('<u2')   
    rawdata = np.fromfile(f, dtype=dt)

length = 2108       #file specifc
height = 384
rawdata = np.reshape(rawdata, (len(wavelengths), length, height))  
master_data_2 = np.rot90(rawdata, 1, (1,2))

# cropping
mac1 = master_data_2[:, 3:3+257, 1003:1003+207] 
mac2 = master_data_2[:, 0:257, 733:733+207] 
mac3 = master_data_2[:, 3:336, 523:523+207] 
lengths = [mac1.shape[2], mac2.shape[2], mac3.shape[2], mac4.shape[2]] 
heights = [mac1.shape[1], mac2.shape[1], mac3.shape[1], mac4.shape[1]] 

#calculate dimensions of simulated images according to scaling factor
ozfuel_dims_mac1 = [int(heights[0]/ozfuel_scaling), int(lengths[0]/ozfuel_scaling)]
ozfuel_dims_mac2 = [int(heights[1]/ozfuel_scaling), int(lengths[1]/ozfuel_scaling)]
ozfuel_dims_mac3 = [int(heights[2]/ozfuel_scaling), int(lengths[2]/ozfuel_scaling)]

measured_fmc_file2 = [91.005, 88.56, 103.16]      #fuel moisture content of cropped region (float)              

# OzFuel MAC 1
resized_imgs = []
for wave in ozfuel_hyper[0]:
    resized_imgs.append( ski.resize(mac1[wave, :, :], (ozfuel_dims_mac1[0], ozfuel_dims_mac1[1]), preserve_range=True))
cube = np.stack(resized_imgs, axis=2)
raw_img = np.median(cube, axis=2)
norm_img = (raw_img - np.min(raw_img)) / (np.max(raw_img) - np.min(raw_img))
ozfuel_band1_spectra.append(np.median(norm_img))
# OzFuel MAC 2
resized_imgs = []
for wave in ozfuel_hyper[0]:
    resized_imgs.append( ski.resize(mac2[wave, :, :], (ozfuel_dims_mac2[0], ozfuel_dims_mac2[1]), preserve_range=True))
cube = np.stack(resized_imgs, axis=2)
raw_img = np.median(cube, axis=2)
norm_img = (raw_img - np.min(raw_img)) / (np.max(raw_img) - np.min(raw_img))
ozfuel_band1_spectra.append(np.median(norm_img))   
# OzFuel MAC 3
resized_imgs = []
for wave in ozfuel_hyper[0]:
    resized_imgs.append( ski.resize(mac3[wave, :, :], (ozfuel_dims_mac3[0], ozfuel_dims_mac3[1]), preserve_range=True))
cube = np.stack(resized_imgs, axis=2)
raw_img = np.median(cube, axis=2)
norm_img = (raw_img - np.min(raw_img)) / (np.max(raw_img) - np.min(raw_img))
ozfuel_band1_spectra.append(np.median(norm_img))
    

#File 3
file_3 = "C:/Users/Sam/Google Drive/UNI/2022/ENGN4350/Python/FMC/20220113_Reorder_288_2431_384_U16.bin"       
with open(file_3) as f:
    dt = np.dtype('<u2')   
    rawdata = np.fromfile(f, dtype=dt)

length = 2431       #file specifc
height = 384
rawdata = np.reshape(rawdata, (len(wavelengths), length, height))  
master_data_3 = np.rot90(rawdata, 1, (1,2))

# cropping
mac1 = master_data_3[:, 0:384, 1126:1126+177] 
mac2 = master_data_3[:, 130:130+248, 830:830+219] 
mac3 = master_data_3[:, 148:148+235, 671:671+139] 
mac4 = master_data_3[:, 177:177+266, 577:577+85] 
lengths = [mac1.shape[2], mac2.shape[2], mac3.shape[2], mac4.shape[2]] 
heights = [mac1.shape[1], mac2.shape[1], mac3.shape[1], mac4.shape[1]] 

#calculate dimensions of simulated images according to scaling factor
ozfuel_dims_mac1 = [int(heights[0]/ozfuel_scaling), int(lengths[0]/ozfuel_scaling)]
ozfuel_dims_mac2 = [int(heights[1]/ozfuel_scaling), int(lengths[1]/ozfuel_scaling)]
ozfuel_dims_mac3 = [int(heights[2]/ozfuel_scaling), int(lengths[2]/ozfuel_scaling)]
ozfuel_dims_mac4 = [int(heights[3]/ozfuel_scaling), int(lengths[3]/ozfuel_scaling)]

measured_fmc_file3 = [102.84, 75.26, 137.735, 68.125]      #fuel moisture content of cropped region (float)              

# OzFuel MAC 1
resized_imgs = []
for wave in ozfuel_hyper[0]:
    resized_imgs.append( ski.resize(mac1[wave, :, :], (ozfuel_dims_mac1[0], ozfuel_dims_mac1[1]), preserve_range=True))
cube = np.stack(resized_imgs, axis=2)
raw_img = np.median(cube, axis=2)
norm_img = (raw_img - np.min(raw_img)) / (np.max(raw_img) - np.min(raw_img))
ozfuel_band1_spectra.append(np.median(norm_img))
# OzFuel MAC 2
resized_imgs = []
for wave in ozfuel_hyper[0]:
    resized_imgs.append( ski.resize(mac2[wave, :, :], (ozfuel_dims_mac2[0], ozfuel_dims_mac2[1]), preserve_range=True))
cube = np.stack(resized_imgs, axis=2)
raw_img = np.median(cube, axis=2)
norm_img = (raw_img - np.min(raw_img)) / (np.max(raw_img) - np.min(raw_img))
ozfuel_band1_spectra.append(np.median(norm_img))   
# OzFuel MAC 3
resized_imgs = []
for wave in ozfuel_hyper[0]:
    resized_imgs.append( ski.resize(mac3[wave, :, :], (ozfuel_dims_mac3[0], ozfuel_dims_mac3[1]), preserve_range=True))
cube = np.stack(resized_imgs, axis=2)
raw_img = np.median(cube, axis=2)
norm_img = (raw_img - np.min(raw_img)) / (np.max(raw_img) - np.min(raw_img))
ozfuel_band1_spectra.append(np.median(norm_img))
# OzFuel MAC 4
resized_imgs = []
for wave in ozfuel_hyper[0]:
    resized_imgs.append( ski.resize(mac4[wave, :, :], (ozfuel_dims_mac4[0], ozfuel_dims_mac4[1]), preserve_range=True))
cube = np.stack(resized_imgs, axis=2)
raw_img = np.median(cube, axis=2)
norm_img = (raw_img - np.min(raw_img)) / (np.max(raw_img) - np.min(raw_img))
ozfuel_band1_spectra.append(np.median(norm_img))

plt.scatter(measured_fmc_file1+measured_fmc_file2+measured_fmc_file3, ozfuel_band1_spectra)



#File 4
file_4 = "C:/Users/Sam/Google Drive/UNI/2022/ENGN4350/Python/FMC/20220202_Reorder_288_2351_384_U16.bin"       
with open(file_4) as f:
    dt = np.dtype('<u2')   
    rawdata = np.fromfile(f, dtype=dt)

length = 2351       #file specifc
height = 384
rawdata = np.reshape(rawdata, (len(wavelengths), length, height))  
master_data_4 = np.rot90(rawdata, 1, (1,2))

# cropping
mac1 = master_data_4[:, 0:384, 1168:1168+192] 
mac2 = master_data_4[:, 0:384, 920:920+192] 
mac3 = master_data_4[:, 60:60+324, 718:718+153] 
mac4 = master_data_4[:, 58:58+324, 640:640+75] 
lengths = [mac1.shape[2], mac2.shape[2], mac3.shape[2], mac4.shape[2]] 
heights = [mac1.shape[1], mac2.shape[1], mac3.shape[1], mac4.shape[1]] 

#calculate dimensions of simulated images according to scaling factor
ozfuel_dims_mac1 = [int(heights[0]/ozfuel_scaling), int(lengths[0]/ozfuel_scaling)]
ozfuel_dims_mac2 = [int(heights[1]/ozfuel_scaling), int(lengths[1]/ozfuel_scaling)]
ozfuel_dims_mac3 = [int(heights[2]/ozfuel_scaling), int(lengths[2]/ozfuel_scaling)]
ozfuel_dims_mac4 = [int(heights[3]/ozfuel_scaling), int(lengths[3]/ozfuel_scaling)]

measured_fmc_file4 = [112.52, 81.535, 100.495, 85.185]      #fuel moisture content of cropped region (float)              

# OzFuel MAC 1
resized_imgs = []
for wave in ozfuel_hyper[0]:
    resized_imgs.append( ski.resize(mac1[wave, :, :], (ozfuel_dims_mac1[0], ozfuel_dims_mac1[1]), preserve_range=True))
cube = np.stack(resized_imgs, axis=2)
raw_img = np.median(cube, axis=2)
norm_img = (raw_img - np.min(raw_img)) / (np.max(raw_img) - np.min(raw_img))
ozfuel_band1_spectra.append(np.median(norm_img))
# OzFuel MAC 2
resized_imgs = []
for wave in ozfuel_hyper[0]:
    resized_imgs.append( ski.resize(mac2[wave, :, :], (ozfuel_dims_mac2[0], ozfuel_dims_mac2[1]), preserve_range=True))
cube = np.stack(resized_imgs, axis=2)
raw_img = np.median(cube, axis=2)
norm_img = (raw_img - np.min(raw_img)) / (np.max(raw_img) - np.min(raw_img))
ozfuel_band1_spectra.append(np.median(norm_img))   
# OzFuel MAC 3
resized_imgs = []
for wave in ozfuel_hyper[0]:
    resized_imgs.append( ski.resize(mac3[wave, :, :], (ozfuel_dims_mac3[0], ozfuel_dims_mac3[1]), preserve_range=True))
cube = np.stack(resized_imgs, axis=2)
raw_img = np.median(cube, axis=2)
norm_img = (raw_img - np.min(raw_img)) / (np.max(raw_img) - np.min(raw_img))
ozfuel_band1_spectra.append(np.median(norm_img))
# OzFuel MAC 4
resized_imgs = []
for wave in ozfuel_hyper[0]:
    resized_imgs.append( ski.resize(mac4[wave, :, :], (ozfuel_dims_mac4[0], ozfuel_dims_mac4[1]), preserve_range=True))
cube = np.stack(resized_imgs, axis=2)
raw_img = np.median(cube, axis=2)
norm_img = (raw_img - np.min(raw_img)) / (np.max(raw_img) - np.min(raw_img))
ozfuel_band1_spectra.append(np.median(norm_img))

plt.scatter(measured_fmc_file1+measured_fmc_file2+measured_fmc_file3+measured_fmc_file4, ozfuel_band1_spectra)



#File 5
file_5 = "C:/Users/Sam/Google Drive/UNI/2022/ENGN4350/Python/FMC/20211102_Reorder_288_3109_384_U16.bin"       
with open(file_5) as f:
    dt = np.dtype('<u2')   
    rawdata = np.fromfile(f, dtype=dt)

length = 3109       #file specifc
height = 384
rawdata = np.reshape(rawdata, (len(wavelengths), length, height))  
master_data_5 = np.rot90(rawdata, 1, (1,2))


# cropping
mac1 = master_data_5[:, 0:384, 1080:1080+216] 
mac2 = master_data_5[:, 0:310, 834:834+216] 
mac3 = master_data_5[:, 0:373, 659:659+153] 
mac4 = master_data_5[:, :252, 622:622+61] 
lengths = [mac1.shape[2], mac2.shape[2], mac3.shape[2], mac4.shape[2]] 
heights = [mac1.shape[1], mac2.shape[1], mac3.shape[1], mac4.shape[1]] 

#calculate dimensions of simulated images according to scaling factor
ozfuel_dims_mac1 = [int(heights[0]/ozfuel_scaling), int(lengths[0]/ozfuel_scaling)]
ozfuel_dims_mac2 = [int(heights[1]/ozfuel_scaling), int(lengths[1]/ozfuel_scaling)]
ozfuel_dims_mac3 = [int(heights[2]/ozfuel_scaling), int(lengths[2]/ozfuel_scaling)]
ozfuel_dims_mac4 = [int(heights[3]/ozfuel_scaling), int(lengths[3]/ozfuel_scaling)]

measured_fmc_file5 = [74.36, 73.41, 76.56, 63.405]      #fuel moisture content of cropped region (float)              

# OzFuel MAC 1
resized_imgs = []
for wave in ozfuel_hyper[0]:
    resized_imgs.append( ski.resize(mac1[wave, :, :], (ozfuel_dims_mac1[0], ozfuel_dims_mac1[1]), preserve_range=True))
cube = np.stack(resized_imgs, axis=2)
raw_img = np.median(cube, axis=2)
norm_img = (raw_img - np.min(raw_img)) / (np.max(raw_img) - np.min(raw_img))
ozfuel_band1_spectra.append(np.median(norm_img))
# OzFuel MAC 2
resized_imgs = []
for wave in ozfuel_hyper[0]:
    resized_imgs.append( ski.resize(mac2[wave, :, :], (ozfuel_dims_mac2[0], ozfuel_dims_mac2[1]), preserve_range=True))
cube = np.stack(resized_imgs, axis=2)
raw_img = np.median(cube, axis=2)
norm_img = (raw_img - np.min(raw_img)) / (np.max(raw_img) - np.min(raw_img))
ozfuel_band1_spectra.append(np.median(norm_img))   
# OzFuel MAC 3
resized_imgs = []
for wave in ozfuel_hyper[0]:
    resized_imgs.append( ski.resize(mac3[wave, :, :], (ozfuel_dims_mac3[0], ozfuel_dims_mac3[1]), preserve_range=True))
cube = np.stack(resized_imgs, axis=2)
raw_img = np.median(cube, axis=2)
norm_img = (raw_img - np.min(raw_img)) / (np.max(raw_img) - np.min(raw_img))
ozfuel_band1_spectra.append(np.median(norm_img))
# OzFuel MAC 4
resized_imgs = []
for wave in ozfuel_hyper[0]:
    resized_imgs.append( ski.resize(mac4[wave, :, :], (ozfuel_dims_mac4[0], ozfuel_dims_mac4[1]), preserve_range=True))
cube = np.stack(resized_imgs, axis=2)
raw_img = np.median(cube, axis=2)
norm_img = (raw_img - np.min(raw_img)) / (np.max(raw_img) - np.min(raw_img))
ozfuel_band1_spectra.append(np.median(norm_img))

# In[]: OzFuel band 2

ozfuel_band2_spectra = []

#File 1
file_1 = "C:/Users/Sam/Google Drive/UNI/2022/ENGN4350/Python/FMC/20220228_Reorder_288_2349_384_U16.bin"       
with open(file_1) as f:
    dt = np.dtype('<u2')   
    rawdata = np.fromfile(f, dtype=dt)

length = 2349       #file specifc
height = 384
rawdata = np.reshape(rawdata, (len(wavelengths), length, height))  
master_data_1 = np.rot90(rawdata, 1, (1,2))

# cropping
mac1 = master_data_1[:, 49:49+333, 1293:1293+178] 
mac2 = master_data_1[:, 41:41+333, 1068:1068+178] 
mac3 = master_data_1[:, 49:49+334, 899:899+124] 
mac4 = master_data_1[:, 102:102+276, 791:791+85] 
lengths = [mac1.shape[2], mac2.shape[2], mac3.shape[2], mac4.shape[2]] 
heights = [mac1.shape[1], mac2.shape[1], mac3.shape[1], mac4.shape[1]] 

#calculate dimensions of simulated images according to scaling factor
ozfuel_dims_mac1 = [int(heights[0]/ozfuel_scaling), int(lengths[0]/ozfuel_scaling)]
ozfuel_dims_mac2 = [int(heights[1]/ozfuel_scaling), int(lengths[1]/ozfuel_scaling)]
ozfuel_dims_mac3 = [int(heights[2]/ozfuel_scaling), int(lengths[2]/ozfuel_scaling)]
ozfuel_dims_mac4 = [int(heights[3]/ozfuel_scaling), int(lengths[3]/ozfuel_scaling)]

#File 1           
# OzFuel MAC 1
resized_imgs = []
for wave in ozfuel_hyper[1]:
    resized_imgs.append( ski.resize(mac1[wave, :, :], (ozfuel_dims_mac1[0], ozfuel_dims_mac1[1]), preserve_range=True))
cube = np.stack(resized_imgs, axis=2)
raw_img = np.median(cube, axis=2)
norm_img = (raw_img - np.min(raw_img)) / (np.max(raw_img) - np.min(raw_img))
ozfuel_band2_spectra.append(np.median(norm_img))
# OzFuel MAC 2
resized_imgs = []
for wave in ozfuel_hyper[1]:
    resized_imgs.append( ski.resize(mac2[wave, :, :], (ozfuel_dims_mac2[0], ozfuel_dims_mac2[1]), preserve_range=True))
cube = np.stack(resized_imgs, axis=2)
raw_img = np.median(cube, axis=2)
norm_img = (raw_img - np.min(raw_img)) / (np.max(raw_img) - np.min(raw_img))
ozfuel_band2_spectra.append(np.median(norm_img))   
# OzFuel MAC 3
resized_imgs = []
for wave in ozfuel_hyper[1]:
    resized_imgs.append( ski.resize(mac3[wave, :, :], (ozfuel_dims_mac3[0], ozfuel_dims_mac3[1]), preserve_range=True))
cube = np.stack(resized_imgs, axis=2)
raw_img = np.median(cube, axis=2)
norm_img = (raw_img - np.min(raw_img)) / (np.max(raw_img) - np.min(raw_img))
ozfuel_band2_spectra.append(np.median(norm_img))
# OzFuel MAC 4
resized_imgs = []
for wave in ozfuel_hyper[1]:
    resized_imgs.append( ski.resize(mac4[wave, :, :], (ozfuel_dims_mac4[0], ozfuel_dims_mac4[1]), preserve_range=True))
cube = np.stack(resized_imgs, axis=2)
raw_img = np.median(cube, axis=2)
norm_img = (raw_img - np.min(raw_img)) / (np.max(raw_img) - np.min(raw_img))
ozfuel_band2_spectra.append(np.median(norm_img))
  


  
#File 2
file_2 = "C:/Users/Sam/Google Drive/UNI/2022/ENGN4350/Python/FMC/20211206_Reorder_288_2108_384_U16.bin"       
with open(file_2) as f:
    dt = np.dtype('<u2')   
    rawdata = np.fromfile(f, dtype=dt)

length = 2108       #file specifc
height = 384
rawdata = np.reshape(rawdata, (len(wavelengths), length, height))  
master_data_2 = np.rot90(rawdata, 1, (1,2))

# cropping
mac1 = master_data_2[:, 3:3+257, 1003:1003+207] 
mac2 = master_data_2[:, 0:257, 733:733+207] 
mac3 = master_data_2[:, 3:336, 523:523+207] 
lengths = [mac1.shape[2], mac2.shape[2], mac3.shape[2], mac4.shape[2]] 
heights = [mac1.shape[1], mac2.shape[1], mac3.shape[1], mac4.shape[1]] 

#calculate dimensions of simulated images according to scaling factor
ozfuel_dims_mac1 = [int(heights[0]/ozfuel_scaling), int(lengths[0]/ozfuel_scaling)]
ozfuel_dims_mac2 = [int(heights[1]/ozfuel_scaling), int(lengths[1]/ozfuel_scaling)]
ozfuel_dims_mac3 = [int(heights[2]/ozfuel_scaling), int(lengths[2]/ozfuel_scaling)]

measured_fmc_file2 = [91.005, 88.56, 103.16]      #fuel moisture content of cropped region (float)              

# OzFuel MAC 1
resized_imgs = []
for wave in ozfuel_hyper[1]:
    resized_imgs.append( ski.resize(mac1[wave, :, :], (ozfuel_dims_mac1[0], ozfuel_dims_mac1[1]), preserve_range=True))
cube = np.stack(resized_imgs, axis=2)
raw_img = np.median(cube, axis=2)
norm_img = (raw_img - np.min(raw_img)) / (np.max(raw_img) - np.min(raw_img))
ozfuel_band2_spectra.append(np.median(norm_img))
# OzFuel MAC 2
resized_imgs = []
for wave in ozfuel_hyper[1]:
    resized_imgs.append( ski.resize(mac2[wave, :, :], (ozfuel_dims_mac2[0], ozfuel_dims_mac2[1]), preserve_range=True))
cube = np.stack(resized_imgs, axis=2)
raw_img = np.median(cube, axis=2)
norm_img = (raw_img - np.min(raw_img)) / (np.max(raw_img) - np.min(raw_img))
ozfuel_band2_spectra.append(np.median(norm_img))   
# OzFuel MAC 3
resized_imgs = []
for wave in ozfuel_hyper[1]:
    resized_imgs.append( ski.resize(mac3[wave, :, :], (ozfuel_dims_mac3[0], ozfuel_dims_mac3[1]), preserve_range=True))
cube = np.stack(resized_imgs, axis=2)
raw_img = np.median(cube, axis=2)
norm_img = (raw_img - np.min(raw_img)) / (np.max(raw_img) - np.min(raw_img))
ozfuel_band2_spectra.append(np.median(norm_img))
    

#File 3
file_3 = "C:/Users/Sam/Google Drive/UNI/2022/ENGN4350/Python/FMC/20220113_Reorder_288_2431_384_U16.bin"       
with open(file_3) as f:
    dt = np.dtype('<u2')   
    rawdata = np.fromfile(f, dtype=dt)

length = 2431       #file specifc
height = 384
rawdata = np.reshape(rawdata, (len(wavelengths), length, height))  
master_data_3 = np.rot90(rawdata, 1, (1,2))

# cropping
mac1 = master_data_3[:, 0:384, 1126:1126+177] 
mac2 = master_data_3[:, 130:130+248, 830:830+219] 
mac3 = master_data_3[:, 148:148+235, 671:671+139] 
mac4 = master_data_3[:, 177:177+266, 577:577+85] 
lengths = [mac1.shape[2], mac2.shape[2], mac3.shape[2], mac4.shape[2]] 
heights = [mac1.shape[1], mac2.shape[1], mac3.shape[1], mac4.shape[1]] 

#calculate dimensions of simulated images according to scaling factor
ozfuel_dims_mac1 = [int(heights[0]/ozfuel_scaling), int(lengths[0]/ozfuel_scaling)]
ozfuel_dims_mac2 = [int(heights[1]/ozfuel_scaling), int(lengths[1]/ozfuel_scaling)]
ozfuel_dims_mac3 = [int(heights[2]/ozfuel_scaling), int(lengths[2]/ozfuel_scaling)]
ozfuel_dims_mac4 = [int(heights[3]/ozfuel_scaling), int(lengths[3]/ozfuel_scaling)]

measured_fmc_file3 = [102.84, 75.26, 137.735, 68.125]      #fuel moisture content of cropped region (float)              

# OzFuel MAC 1
resized_imgs = []
for wave in ozfuel_hyper[1]:
    resized_imgs.append( ski.resize(mac1[wave, :, :], (ozfuel_dims_mac1[0], ozfuel_dims_mac1[1]), preserve_range=True))
cube = np.stack(resized_imgs, axis=2)
raw_img = np.median(cube, axis=2)
norm_img = (raw_img - np.min(raw_img)) / (np.max(raw_img) - np.min(raw_img))
ozfuel_band2_spectra.append(np.median(norm_img))
# OzFuel MAC 2
resized_imgs = []
for wave in ozfuel_hyper[1]:
    resized_imgs.append( ski.resize(mac2[wave, :, :], (ozfuel_dims_mac2[0], ozfuel_dims_mac2[1]), preserve_range=True))
cube = np.stack(resized_imgs, axis=2)
raw_img = np.median(cube, axis=2)
norm_img = (raw_img - np.min(raw_img)) / (np.max(raw_img) - np.min(raw_img))
ozfuel_band2_spectra.append(np.median(norm_img))   
# OzFuel MAC 3
resized_imgs = []
for wave in ozfuel_hyper[1]:
    resized_imgs.append( ski.resize(mac3[wave, :, :], (ozfuel_dims_mac3[0], ozfuel_dims_mac3[1]), preserve_range=True))
cube = np.stack(resized_imgs, axis=2)
raw_img = np.median(cube, axis=2)
norm_img = (raw_img - np.min(raw_img)) / (np.max(raw_img) - np.min(raw_img))
ozfuel_band2_spectra.append(np.median(norm_img))
# OzFuel MAC 4
resized_imgs = []
for wave in ozfuel_hyper[1]:
    resized_imgs.append( ski.resize(mac4[wave, :, :], (ozfuel_dims_mac4[0], ozfuel_dims_mac4[1]), preserve_range=True))
cube = np.stack(resized_imgs, axis=2)
raw_img = np.median(cube, axis=2)
norm_img = (raw_img - np.min(raw_img)) / (np.max(raw_img) - np.min(raw_img))
ozfuel_band2_spectra.append(np.median(norm_img))


#File 4
file_4 = "C:/Users/Sam/Google Drive/UNI/2022/ENGN4350/Python/FMC/20220202_Reorder_288_2351_384_U16.bin"       
with open(file_4) as f:
    dt = np.dtype('<u2')   
    rawdata = np.fromfile(f, dtype=dt)

length = 2351       #file specifc
height = 384
rawdata = np.reshape(rawdata, (len(wavelengths), length, height))  
master_data_4 = np.rot90(rawdata, 1, (1,2))

# cropping
mac1 = master_data_4[:, 0:384, 1168:1168+192] 
mac2 = master_data_4[:, 0:384, 920:920+192] 
mac3 = master_data_4[:, 60:60+324, 718:718+153] 
mac4 = master_data_4[:, 58:58+324, 640:640+75] 
lengths = [mac1.shape[2], mac2.shape[2], mac3.shape[2], mac4.shape[2]] 
heights = [mac1.shape[1], mac2.shape[1], mac3.shape[1], mac4.shape[1]] 

#calculate dimensions of simulated images according to scaling factor
ozfuel_dims_mac1 = [int(heights[0]/ozfuel_scaling), int(lengths[0]/ozfuel_scaling)]
ozfuel_dims_mac2 = [int(heights[1]/ozfuel_scaling), int(lengths[1]/ozfuel_scaling)]
ozfuel_dims_mac3 = [int(heights[2]/ozfuel_scaling), int(lengths[2]/ozfuel_scaling)]
ozfuel_dims_mac4 = [int(heights[3]/ozfuel_scaling), int(lengths[3]/ozfuel_scaling)]

measured_fmc_file4 = [112.52, 81.535, 100.495, 85.185]      #fuel moisture content of cropped region (float)              

# OzFuel MAC 1
resized_imgs = []
for wave in ozfuel_hyper[1]:
    resized_imgs.append( ski.resize(mac1[wave, :, :], (ozfuel_dims_mac1[0], ozfuel_dims_mac1[1]), preserve_range=True))
cube = np.stack(resized_imgs, axis=2)
raw_img = np.median(cube, axis=2)
norm_img = (raw_img - np.min(raw_img)) / (np.max(raw_img) - np.min(raw_img))
ozfuel_band2_spectra.append(np.median(norm_img))
# OzFuel MAC 2
resized_imgs = []
for wave in ozfuel_hyper[1]:
    resized_imgs.append( ski.resize(mac2[wave, :, :], (ozfuel_dims_mac2[0], ozfuel_dims_mac2[1]), preserve_range=True))
cube = np.stack(resized_imgs, axis=2)
raw_img = np.median(cube, axis=2)
norm_img = (raw_img - np.min(raw_img)) / (np.max(raw_img) - np.min(raw_img))
ozfuel_band2_spectra.append(np.median(norm_img))   
# OzFuel MAC 3
resized_imgs = []
for wave in ozfuel_hyper[1]:
    resized_imgs.append( ski.resize(mac3[wave, :, :], (ozfuel_dims_mac3[0], ozfuel_dims_mac3[1]), preserve_range=True))
cube = np.stack(resized_imgs, axis=2)
raw_img = np.median(cube, axis=2)
norm_img = (raw_img - np.min(raw_img)) / (np.max(raw_img) - np.min(raw_img))
ozfuel_band2_spectra.append(np.median(norm_img))
# OzFuel MAC 4
resized_imgs = []
for wave in ozfuel_hyper[1]:
    resized_imgs.append( ski.resize(mac4[wave, :, :], (ozfuel_dims_mac4[0], ozfuel_dims_mac4[1]), preserve_range=True))
cube = np.stack(resized_imgs, axis=2)
raw_img = np.median(cube, axis=2)
norm_img = (raw_img - np.min(raw_img)) / (np.max(raw_img) - np.min(raw_img))
ozfuel_band2_spectra.append(np.median(norm_img))


#File 5
file_5 = "C:/Users/Sam/Google Drive/UNI/2022/ENGN4350/Python/FMC/20211102_Reorder_288_3109_384_U16.bin"       
with open(file_5) as f:
    dt = np.dtype('<u2')   
    rawdata = np.fromfile(f, dtype=dt)

length = 3109       #file specifc
height = 384
rawdata = np.reshape(rawdata, (len(wavelengths), length, height))  
master_data_5 = np.rot90(rawdata, 1, (1,2))


# cropping
mac1 = master_data_5[:, 0:384, 1080:1080+216] 
mac2 = master_data_5[:, 0:310, 834:834+216] 
mac3 = master_data_5[:, 0:373, 659:659+153] 
mac4 = master_data_5[:, :252, 622:622+61] 
lengths = [mac1.shape[2], mac2.shape[2], mac3.shape[2], mac4.shape[2]] 
heights = [mac1.shape[1], mac2.shape[1], mac3.shape[1], mac4.shape[1]] 

#calculate dimensions of simulated images according to scaling factor
ozfuel_dims_mac1 = [int(heights[0]/ozfuel_scaling), int(lengths[0]/ozfuel_scaling)]
ozfuel_dims_mac2 = [int(heights[1]/ozfuel_scaling), int(lengths[1]/ozfuel_scaling)]
ozfuel_dims_mac3 = [int(heights[2]/ozfuel_scaling), int(lengths[2]/ozfuel_scaling)]
ozfuel_dims_mac4 = [int(heights[3]/ozfuel_scaling), int(lengths[3]/ozfuel_scaling)]

measured_fmc_file5 = [74.36, 73.41, 76.56, 63.405]      #fuel moisture content of cropped region (float)              

# OzFuel MAC 1
resized_imgs = []
for wave in ozfuel_hyper[1]:
    resized_imgs.append( ski.resize(mac1[wave, :, :], (ozfuel_dims_mac1[0], ozfuel_dims_mac1[1]), preserve_range=True))
cube = np.stack(resized_imgs, axis=2)
raw_img = np.median(cube, axis=2)
norm_img = (raw_img - np.min(raw_img)) / (np.max(raw_img) - np.min(raw_img))
ozfuel_band2_spectra.append(np.median(norm_img))
# OzFuel MAC 2
resized_imgs = []
for wave in ozfuel_hyper[1]:
    resized_imgs.append( ski.resize(mac2[wave, :, :], (ozfuel_dims_mac2[0], ozfuel_dims_mac2[1]), preserve_range=True))
cube = np.stack(resized_imgs, axis=2)
raw_img = np.median(cube, axis=2)
norm_img = (raw_img - np.min(raw_img)) / (np.max(raw_img) - np.min(raw_img))
ozfuel_band2_spectra.append(np.median(norm_img))   
# OzFuel MAC 3
resized_imgs = []
for wave in ozfuel_hyper[1]:
    resized_imgs.append( ski.resize(mac3[wave, :, :], (ozfuel_dims_mac3[0], ozfuel_dims_mac3[1]), preserve_range=True))
cube = np.stack(resized_imgs, axis=2)
raw_img = np.median(cube, axis=2)
norm_img = (raw_img - np.min(raw_img)) / (np.max(raw_img) - np.min(raw_img))
ozfuel_band2_spectra.append(np.median(norm_img))
# OzFuel MAC 4
resized_imgs = []
for wave in ozfuel_hyper[1]:
    resized_imgs.append( ski.resize(mac4[wave, :, :], (ozfuel_dims_mac4[0], ozfuel_dims_mac4[1]), preserve_range=True))
cube = np.stack(resized_imgs, axis=2)
raw_img = np.median(cube, axis=2)
norm_img = (raw_img - np.min(raw_img)) / (np.max(raw_img) - np.min(raw_img))
ozfuel_band2_spectra.append(np.median(norm_img))

# In[]: OzFuel band 3

ozfuel_band3_spectra = []

#File 1
file_1 = "C:/Users/Sam/Google Drive/UNI/2022/ENGN4350/Python/FMC/20220228_Reorder_288_2349_384_U16.bin"       
with open(file_1) as f:
    dt = np.dtype('<u2')   
    rawdata = np.fromfile(f, dtype=dt)

length = 2349       #file specifc
height = 384
rawdata = np.reshape(rawdata, (len(wavelengths), length, height))  
master_data_1 = np.rot90(rawdata, 1, (1,2))

# cropping
mac1 = master_data_1[:, 49:49+333, 1293:1293+178] 
mac2 = master_data_1[:, 41:41+333, 1068:1068+178] 
mac3 = master_data_1[:, 49:49+334, 899:899+124] 
mac4 = master_data_1[:, 102:102+276, 791:791+85] 
lengths = [mac1.shape[2], mac2.shape[2], mac3.shape[2], mac4.shape[2]] 
heights = [mac1.shape[1], mac2.shape[1], mac3.shape[1], mac4.shape[1]] 

#calculate dimensions of simulated images according to scaling factor
ozfuel_dims_mac1 = [int(heights[0]/ozfuel_scaling), int(lengths[0]/ozfuel_scaling)]
ozfuel_dims_mac2 = [int(heights[1]/ozfuel_scaling), int(lengths[1]/ozfuel_scaling)]
ozfuel_dims_mac3 = [int(heights[2]/ozfuel_scaling), int(lengths[2]/ozfuel_scaling)]
ozfuel_dims_mac4 = [int(heights[3]/ozfuel_scaling), int(lengths[3]/ozfuel_scaling)]

#File 1           
# OzFuel MAC 1
resized_imgs = []
for wave in ozfuel_hyper[2]:
    resized_imgs.append( ski.resize(mac1[wave, :, :], (ozfuel_dims_mac1[0], ozfuel_dims_mac1[1]), preserve_range=True))
cube = np.stack(resized_imgs, axis=2)
raw_img = np.median(cube, axis=2)
norm_img = (raw_img - np.min(raw_img)) / (np.max(raw_img) - np.min(raw_img))
ozfuel_band3_spectra.append(np.median(norm_img))
# OzFuel MAC 2
resized_imgs = []
for wave in ozfuel_hyper[2]:
    resized_imgs.append( ski.resize(mac2[wave, :, :], (ozfuel_dims_mac2[0], ozfuel_dims_mac2[1]), preserve_range=True))
cube = np.stack(resized_imgs, axis=2)
raw_img = np.median(cube, axis=2)
norm_img = (raw_img - np.min(raw_img)) / (np.max(raw_img) - np.min(raw_img))
ozfuel_band3_spectra.append(np.median(norm_img))   
# OzFuel MAC 3
resized_imgs = []
for wave in ozfuel_hyper[2]:
    resized_imgs.append( ski.resize(mac3[wave, :, :], (ozfuel_dims_mac3[0], ozfuel_dims_mac3[1]), preserve_range=True))
cube = np.stack(resized_imgs, axis=2)
raw_img = np.median(cube, axis=2)
norm_img = (raw_img - np.min(raw_img)) / (np.max(raw_img) - np.min(raw_img))
ozfuel_band3_spectra.append(np.median(norm_img))
# OzFuel MAC 4
resized_imgs = []
for wave in ozfuel_hyper[2]:
    resized_imgs.append( ski.resize(mac4[wave, :, :], (ozfuel_dims_mac4[0], ozfuel_dims_mac4[1]), preserve_range=True))
cube = np.stack(resized_imgs, axis=2)
raw_img = np.median(cube, axis=2)
norm_img = (raw_img - np.min(raw_img)) / (np.max(raw_img) - np.min(raw_img))
ozfuel_band3_spectra.append(np.median(norm_img))
  


  
#File 2
file_2 = "C:/Users/Sam/Google Drive/UNI/2022/ENGN4350/Python/FMC/20211206_Reorder_288_2108_384_U16.bin"       
with open(file_2) as f:
    dt = np.dtype('<u2')   
    rawdata = np.fromfile(f, dtype=dt)

length = 2108       #file specifc
height = 384
rawdata = np.reshape(rawdata, (len(wavelengths), length, height))  
master_data_2 = np.rot90(rawdata, 1, (1,2))

# cropping
mac1 = master_data_2[:, 3:3+257, 1003:1003+207] 
mac2 = master_data_2[:, 0:257, 733:733+207] 
mac3 = master_data_2[:, 3:336, 523:523+207] 
lengths = [mac1.shape[2], mac2.shape[2], mac3.shape[2], mac4.shape[2]] 
heights = [mac1.shape[1], mac2.shape[1], mac3.shape[1], mac4.shape[1]] 

#calculate dimensions of simulated images according to scaling factor
ozfuel_dims_mac1 = [int(heights[0]/ozfuel_scaling), int(lengths[0]/ozfuel_scaling)]
ozfuel_dims_mac2 = [int(heights[1]/ozfuel_scaling), int(lengths[1]/ozfuel_scaling)]
ozfuel_dims_mac3 = [int(heights[2]/ozfuel_scaling), int(lengths[2]/ozfuel_scaling)]

measured_fmc_file2 = [91.005, 88.56, 103.16]      #fuel moisture content of cropped region (float)              

# OzFuel MAC 1
resized_imgs = []
for wave in ozfuel_hyper[2]:
    resized_imgs.append( ski.resize(mac1[wave, :, :], (ozfuel_dims_mac1[0], ozfuel_dims_mac1[1]), preserve_range=True))
cube = np.stack(resized_imgs, axis=2)
raw_img = np.median(cube, axis=2)
norm_img = (raw_img - np.min(raw_img)) / (np.max(raw_img) - np.min(raw_img))
ozfuel_band3_spectra.append(np.median(norm_img))
# OzFuel MAC 2
resized_imgs = []
for wave in ozfuel_hyper[2]:
    resized_imgs.append( ski.resize(mac2[wave, :, :], (ozfuel_dims_mac2[0], ozfuel_dims_mac2[1]), preserve_range=True))
cube = np.stack(resized_imgs, axis=2)
raw_img = np.median(cube, axis=2)
norm_img = (raw_img - np.min(raw_img)) / (np.max(raw_img) - np.min(raw_img))
ozfuel_band3_spectra.append(np.median(norm_img))   
# OzFuel MAC 3
resized_imgs = []
for wave in ozfuel_hyper[2]:
    resized_imgs.append( ski.resize(mac3[wave, :, :], (ozfuel_dims_mac3[0], ozfuel_dims_mac3[1]), preserve_range=True))
cube = np.stack(resized_imgs, axis=2)
raw_img = np.median(cube, axis=2)
norm_img = (raw_img - np.min(raw_img)) / (np.max(raw_img) - np.min(raw_img))
ozfuel_band3_spectra.append(np.median(norm_img))
    

#File 3
file_3 = "C:/Users/Sam/Google Drive/UNI/2022/ENGN4350/Python/FMC/20220113_Reorder_288_2431_384_U16.bin"       
with open(file_3) as f:
    dt = np.dtype('<u2')   
    rawdata = np.fromfile(f, dtype=dt)

length = 2431       #file specifc
height = 384
rawdata = np.reshape(rawdata, (len(wavelengths), length, height))  
master_data_3 = np.rot90(rawdata, 1, (1,2))

# cropping
mac1 = master_data_3[:, 0:384, 1126:1126+177] 
mac2 = master_data_3[:, 130:130+248, 830:830+219] 
mac3 = master_data_3[:, 148:148+235, 671:671+139] 
mac4 = master_data_3[:, 177:177+266, 577:577+85] 
lengths = [mac1.shape[2], mac2.shape[2], mac3.shape[2], mac4.shape[2]] 
heights = [mac1.shape[1], mac2.shape[1], mac3.shape[1], mac4.shape[1]] 

#calculate dimensions of simulated images according to scaling factor
ozfuel_dims_mac1 = [int(heights[0]/ozfuel_scaling), int(lengths[0]/ozfuel_scaling)]
ozfuel_dims_mac2 = [int(heights[1]/ozfuel_scaling), int(lengths[1]/ozfuel_scaling)]
ozfuel_dims_mac3 = [int(heights[2]/ozfuel_scaling), int(lengths[2]/ozfuel_scaling)]
ozfuel_dims_mac4 = [int(heights[3]/ozfuel_scaling), int(lengths[3]/ozfuel_scaling)]

measured_fmc_file3 = [102.84, 75.26, 137.735, 68.125]      #fuel moisture content of cropped region (float)              

# OzFuel MAC 1
resized_imgs = []
for wave in ozfuel_hyper[2]:
    resized_imgs.append( ski.resize(mac1[wave, :, :], (ozfuel_dims_mac1[0], ozfuel_dims_mac1[1]), preserve_range=True))
cube = np.stack(resized_imgs, axis=2)
raw_img = np.median(cube, axis=2)
norm_img = (raw_img - np.min(raw_img)) / (np.max(raw_img) - np.min(raw_img))
ozfuel_band3_spectra.append(np.median(norm_img))
# OzFuel MAC 2
resized_imgs = []
for wave in ozfuel_hyper[2]:
    resized_imgs.append( ski.resize(mac2[wave, :, :], (ozfuel_dims_mac2[0], ozfuel_dims_mac2[1]), preserve_range=True))
cube = np.stack(resized_imgs, axis=2)
raw_img = np.median(cube, axis=2)
norm_img = (raw_img - np.min(raw_img)) / (np.max(raw_img) - np.min(raw_img))
ozfuel_band3_spectra.append(np.median(norm_img))   
# OzFuel MAC 3
resized_imgs = []
for wave in ozfuel_hyper[2]:
    resized_imgs.append( ski.resize(mac3[wave, :, :], (ozfuel_dims_mac3[0], ozfuel_dims_mac3[1]), preserve_range=True))
cube = np.stack(resized_imgs, axis=2)
raw_img = np.median(cube, axis=2)
norm_img = (raw_img - np.min(raw_img)) / (np.max(raw_img) - np.min(raw_img))
ozfuel_band3_spectra.append(np.median(norm_img))
# OzFuel MAC 4
resized_imgs = []
for wave in ozfuel_hyper[2]:
    resized_imgs.append( ski.resize(mac4[wave, :, :], (ozfuel_dims_mac4[0], ozfuel_dims_mac4[1]), preserve_range=True))
cube = np.stack(resized_imgs, axis=2)
raw_img = np.median(cube, axis=2)
norm_img = (raw_img - np.min(raw_img)) / (np.max(raw_img) - np.min(raw_img))
ozfuel_band3_spectra.append(np.median(norm_img))


#File 4
file_4 = "C:/Users/Sam/Google Drive/UNI/2022/ENGN4350/Python/FMC/20220202_Reorder_288_2351_384_U16.bin"       
with open(file_4) as f:
    dt = np.dtype('<u2')   
    rawdata = np.fromfile(f, dtype=dt)

length = 2351       #file specifc
height = 384
rawdata = np.reshape(rawdata, (len(wavelengths), length, height))  
master_data_4 = np.rot90(rawdata, 1, (1,2))

# cropping
mac1 = master_data_4[:, 0:384, 1168:1168+192] 
mac2 = master_data_4[:, 0:384, 920:920+192] 
mac3 = master_data_4[:, 60:60+324, 718:718+153] 
mac4 = master_data_4[:, 58:58+324, 640:640+75] 
lengths = [mac1.shape[2], mac2.shape[2], mac3.shape[2], mac4.shape[2]] 
heights = [mac1.shape[1], mac2.shape[1], mac3.shape[1], mac4.shape[1]] 

#calculate dimensions of simulated images according to scaling factor
ozfuel_dims_mac1 = [int(heights[0]/ozfuel_scaling), int(lengths[0]/ozfuel_scaling)]
ozfuel_dims_mac2 = [int(heights[1]/ozfuel_scaling), int(lengths[1]/ozfuel_scaling)]
ozfuel_dims_mac3 = [int(heights[2]/ozfuel_scaling), int(lengths[2]/ozfuel_scaling)]
ozfuel_dims_mac4 = [int(heights[3]/ozfuel_scaling), int(lengths[3]/ozfuel_scaling)]

measured_fmc_file4 = [112.52, 81.535, 100.495, 85.185]      #fuel moisture content of cropped region (float)              

# OzFuel MAC 1
resized_imgs = []
for wave in ozfuel_hyper[2]:
    resized_imgs.append( ski.resize(mac1[wave, :, :], (ozfuel_dims_mac1[0], ozfuel_dims_mac1[1]), preserve_range=True))
cube = np.stack(resized_imgs, axis=2)
raw_img = np.median(cube, axis=2)
norm_img = (raw_img - np.min(raw_img)) / (np.max(raw_img) - np.min(raw_img))
ozfuel_band3_spectra.append(np.median(norm_img))
# OzFuel MAC 2
resized_imgs = []
for wave in ozfuel_hyper[2]:
    resized_imgs.append( ski.resize(mac2[wave, :, :], (ozfuel_dims_mac2[0], ozfuel_dims_mac2[1]), preserve_range=True))
cube = np.stack(resized_imgs, axis=2)
raw_img = np.median(cube, axis=2)
norm_img = (raw_img - np.min(raw_img)) / (np.max(raw_img) - np.min(raw_img))
ozfuel_band3_spectra.append(np.median(norm_img))   
# OzFuel MAC 3
resized_imgs = []
for wave in ozfuel_hyper[2]:
    resized_imgs.append( ski.resize(mac3[wave, :, :], (ozfuel_dims_mac3[0], ozfuel_dims_mac3[1]), preserve_range=True))
cube = np.stack(resized_imgs, axis=2)
raw_img = np.median(cube, axis=2)
norm_img = (raw_img - np.min(raw_img)) / (np.max(raw_img) - np.min(raw_img))
ozfuel_band3_spectra.append(np.median(norm_img))
# OzFuel MAC 4
resized_imgs = []
for wave in ozfuel_hyper[2]:
    resized_imgs.append( ski.resize(mac4[wave, :, :], (ozfuel_dims_mac4[0], ozfuel_dims_mac4[1]), preserve_range=True))
cube = np.stack(resized_imgs, axis=2)
raw_img = np.median(cube, axis=2)
norm_img = (raw_img - np.min(raw_img)) / (np.max(raw_img) - np.min(raw_img))
ozfuel_band3_spectra.append(np.median(norm_img))


#File 5
file_5 = "C:/Users/Sam/Google Drive/UNI/2022/ENGN4350/Python/FMC/20211102_Reorder_288_3109_384_U16.bin"       
with open(file_5) as f:
    dt = np.dtype('<u2')   
    rawdata = np.fromfile(f, dtype=dt)

length = 3109       #file specifc
height = 384
rawdata = np.reshape(rawdata, (len(wavelengths), length, height))  
master_data_5 = np.rot90(rawdata, 1, (1,2))


# cropping
mac1 = master_data_5[:, 0:384, 1080:1080+216] 
mac2 = master_data_5[:, 0:310, 834:834+216] 
mac3 = master_data_5[:, 0:373, 659:659+153] 
mac4 = master_data_5[:, :252, 622:622+61] 
lengths = [mac1.shape[2], mac2.shape[2], mac3.shape[2], mac4.shape[2]] 
heights = [mac1.shape[1], mac2.shape[1], mac3.shape[1], mac4.shape[1]] 

#calculate dimensions of simulated images according to scaling factor
ozfuel_dims_mac1 = [int(heights[0]/ozfuel_scaling), int(lengths[0]/ozfuel_scaling)]
ozfuel_dims_mac2 = [int(heights[1]/ozfuel_scaling), int(lengths[1]/ozfuel_scaling)]
ozfuel_dims_mac3 = [int(heights[2]/ozfuel_scaling), int(lengths[2]/ozfuel_scaling)]
ozfuel_dims_mac4 = [int(heights[3]/ozfuel_scaling), int(lengths[3]/ozfuel_scaling)]

measured_fmc_file5 = [74.36, 73.41, 76.56, 63.405]      #fuel moisture content of cropped region (float)              

# OzFuel MAC 1
resized_imgs = []
for wave in ozfuel_hyper[2]:
    resized_imgs.append( ski.resize(mac1[wave, :, :], (ozfuel_dims_mac1[0], ozfuel_dims_mac1[1]), preserve_range=True))
cube = np.stack(resized_imgs, axis=2)
raw_img = np.median(cube, axis=2)
norm_img = (raw_img - np.min(raw_img)) / (np.max(raw_img) - np.min(raw_img))
ozfuel_band3_spectra.append(np.median(norm_img))
# OzFuel MAC 2
resized_imgs = []
for wave in ozfuel_hyper[2]:
    resized_imgs.append( ski.resize(mac2[wave, :, :], (ozfuel_dims_mac2[0], ozfuel_dims_mac2[1]), preserve_range=True))
cube = np.stack(resized_imgs, axis=2)
raw_img = np.median(cube, axis=2)
norm_img = (raw_img - np.min(raw_img)) / (np.max(raw_img) - np.min(raw_img))
ozfuel_band3_spectra.append(np.median(norm_img))   
# OzFuel MAC 3
resized_imgs = []
for wave in ozfuel_hyper[2]:
    resized_imgs.append( ski.resize(mac3[wave, :, :], (ozfuel_dims_mac3[0], ozfuel_dims_mac3[1]), preserve_range=True))
cube = np.stack(resized_imgs, axis=2)
raw_img = np.median(cube, axis=2)
norm_img = (raw_img - np.min(raw_img)) / (np.max(raw_img) - np.min(raw_img))
ozfuel_band3_spectra.append(np.median(norm_img))
# OzFuel MAC 4
resized_imgs = []
for wave in ozfuel_hyper[2]:
    resized_imgs.append( ski.resize(mac4[wave, :, :], (ozfuel_dims_mac4[0], ozfuel_dims_mac4[1]), preserve_range=True))
cube = np.stack(resized_imgs, axis=2)
raw_img = np.median(cube, axis=2)
norm_img = (raw_img - np.min(raw_img)) / (np.max(raw_img) - np.min(raw_img))
ozfuel_band3_spectra.append(np.median(norm_img))

# In[]: OzFuel band 4

ozfuel_band4_spectra = []

#File 1
file_1 = "C:/Users/Sam/Google Drive/UNI/2022/ENGN4350/Python/FMC/20220228_Reorder_288_2349_384_U16.bin"       
with open(file_1) as f:
    dt = np.dtype('<u2')   
    rawdata = np.fromfile(f, dtype=dt)

length = 2349       #file specifc
height = 384
rawdata = np.reshape(rawdata, (len(wavelengths), length, height))  
master_data_1 = np.rot90(rawdata, 1, (1,2))

# cropping
mac1 = master_data_1[:, 49:49+333, 1293:1293+178] 
mac2 = master_data_1[:, 41:41+333, 1068:1068+178] 
mac3 = master_data_1[:, 49:49+334, 899:899+124] 
mac4 = master_data_1[:, 102:102+276, 791:791+85] 
lengths = [mac1.shape[2], mac2.shape[2], mac3.shape[2], mac4.shape[2]] 
heights = [mac1.shape[1], mac2.shape[1], mac3.shape[1], mac4.shape[1]] 

#calculate dimensions of simulated images according to scaling factor
ozfuel_dims_mac1 = [int(heights[0]/ozfuel_scaling), int(lengths[0]/ozfuel_scaling)]
ozfuel_dims_mac2 = [int(heights[1]/ozfuel_scaling), int(lengths[1]/ozfuel_scaling)]
ozfuel_dims_mac3 = [int(heights[2]/ozfuel_scaling), int(lengths[2]/ozfuel_scaling)]
ozfuel_dims_mac4 = [int(heights[3]/ozfuel_scaling), int(lengths[3]/ozfuel_scaling)]

#File 1           
# OzFuel MAC 1
resized_imgs = []
for wave in ozfuel_hyper[3]:
    resized_imgs.append( ski.resize(mac1[wave, :, :], (ozfuel_dims_mac1[0], ozfuel_dims_mac1[1]), preserve_range=True))
cube = np.stack(resized_imgs, axis=2)
raw_img = np.median(cube, axis=2)
norm_img = (raw_img - np.min(raw_img)) / (np.max(raw_img) - np.min(raw_img))
ozfuel_band4_spectra.append(np.median(norm_img))
# OzFuel MAC 2
resized_imgs = []
for wave in ozfuel_hyper[3]:
    resized_imgs.append( ski.resize(mac2[wave, :, :], (ozfuel_dims_mac2[0], ozfuel_dims_mac2[1]), preserve_range=True))
cube = np.stack(resized_imgs, axis=2)
raw_img = np.median(cube, axis=2)
norm_img = (raw_img - np.min(raw_img)) / (np.max(raw_img) - np.min(raw_img))
ozfuel_band4_spectra.append(np.median(norm_img))   
# OzFuel MAC 3
resized_imgs = []
for wave in ozfuel_hyper[3]:
    resized_imgs.append( ski.resize(mac3[wave, :, :], (ozfuel_dims_mac3[0], ozfuel_dims_mac3[1]), preserve_range=True))
cube = np.stack(resized_imgs, axis=2)
raw_img = np.median(cube, axis=2)
norm_img = (raw_img - np.min(raw_img)) / (np.max(raw_img) - np.min(raw_img))
ozfuel_band4_spectra.append(np.median(norm_img))
# OzFuel MAC 4
resized_imgs = []
for wave in ozfuel_hyper[3]:
    resized_imgs.append( ski.resize(mac4[wave, :, :], (ozfuel_dims_mac4[0], ozfuel_dims_mac4[1]), preserve_range=True))
cube = np.stack(resized_imgs, axis=2)
raw_img = np.median(cube, axis=2)
norm_img = (raw_img - np.min(raw_img)) / (np.max(raw_img) - np.min(raw_img))
ozfuel_band4_spectra.append(np.median(norm_img))
  


  
#File 2
file_2 = "C:/Users/Sam/Google Drive/UNI/2022/ENGN4350/Python/FMC/20211206_Reorder_288_2108_384_U16.bin"       
with open(file_2) as f:
    dt = np.dtype('<u2')   
    rawdata = np.fromfile(f, dtype=dt)

length = 2108       #file specifc
height = 384
rawdata = np.reshape(rawdata, (len(wavelengths), length, height))  
master_data_2 = np.rot90(rawdata, 1, (1,2))

# cropping
mac1 = master_data_2[:, 3:3+257, 1003:1003+207] 
mac2 = master_data_2[:, 0:257, 733:733+207] 
mac3 = master_data_2[:, 3:336, 523:523+207] 
lengths = [mac1.shape[2], mac2.shape[2], mac3.shape[2], mac4.shape[2]] 
heights = [mac1.shape[1], mac2.shape[1], mac3.shape[1], mac4.shape[1]] 

#calculate dimensions of simulated images according to scaling factor
ozfuel_dims_mac1 = [int(heights[0]/ozfuel_scaling), int(lengths[0]/ozfuel_scaling)]
ozfuel_dims_mac2 = [int(heights[1]/ozfuel_scaling), int(lengths[1]/ozfuel_scaling)]
ozfuel_dims_mac3 = [int(heights[2]/ozfuel_scaling), int(lengths[2]/ozfuel_scaling)]

measured_fmc_file2 = [91.005, 88.56, 103.16]      #fuel moisture content of cropped region (float)              

# OzFuel MAC 1
resized_imgs = []
for wave in ozfuel_hyper[3]:
    resized_imgs.append( ski.resize(mac1[wave, :, :], (ozfuel_dims_mac1[0], ozfuel_dims_mac1[1]), preserve_range=True))
cube = np.stack(resized_imgs, axis=2)
raw_img = np.median(cube, axis=2)
norm_img = (raw_img - np.min(raw_img)) / (np.max(raw_img) - np.min(raw_img))
ozfuel_band4_spectra.append(np.median(norm_img))
# OzFuel MAC 2
resized_imgs = []
for wave in ozfuel_hyper[3]:
    resized_imgs.append( ski.resize(mac2[wave, :, :], (ozfuel_dims_mac2[0], ozfuel_dims_mac2[1]), preserve_range=True))
cube = np.stack(resized_imgs, axis=2)
raw_img = np.median(cube, axis=2)
norm_img = (raw_img - np.min(raw_img)) / (np.max(raw_img) - np.min(raw_img))
ozfuel_band4_spectra.append(np.median(norm_img))   
# OzFuel MAC 3
resized_imgs = []
for wave in ozfuel_hyper[3]:
    resized_imgs.append( ski.resize(mac3[wave, :, :], (ozfuel_dims_mac3[0], ozfuel_dims_mac3[1]), preserve_range=True))
cube = np.stack(resized_imgs, axis=2)
raw_img = np.median(cube, axis=2)
norm_img = (raw_img - np.min(raw_img)) / (np.max(raw_img) - np.min(raw_img))
ozfuel_band4_spectra.append(np.median(norm_img))
    

#File 3
file_3 = "C:/Users/Sam/Google Drive/UNI/2022/ENGN4350/Python/FMC/20220113_Reorder_288_2431_384_U16.bin"       
with open(file_3) as f:
    dt = np.dtype('<u2')   
    rawdata = np.fromfile(f, dtype=dt)

length = 2431       #file specifc
height = 384
rawdata = np.reshape(rawdata, (len(wavelengths), length, height))  
master_data_3 = np.rot90(rawdata, 1, (1,2))

# cropping
mac1 = master_data_3[:, 0:384, 1126:1126+177] 
mac2 = master_data_3[:, 130:130+248, 830:830+219] 
mac3 = master_data_3[:, 148:148+235, 671:671+139] 
mac4 = master_data_3[:, 177:177+266, 577:577+85] 
lengths = [mac1.shape[2], mac2.shape[2], mac3.shape[2], mac4.shape[2]] 
heights = [mac1.shape[1], mac2.shape[1], mac3.shape[1], mac4.shape[1]] 

#calculate dimensions of simulated images according to scaling factor
ozfuel_dims_mac1 = [int(heights[0]/ozfuel_scaling), int(lengths[0]/ozfuel_scaling)]
ozfuel_dims_mac2 = [int(heights[1]/ozfuel_scaling), int(lengths[1]/ozfuel_scaling)]
ozfuel_dims_mac3 = [int(heights[2]/ozfuel_scaling), int(lengths[2]/ozfuel_scaling)]
ozfuel_dims_mac4 = [int(heights[3]/ozfuel_scaling), int(lengths[3]/ozfuel_scaling)]

measured_fmc_file3 = [102.84, 75.26, 137.735, 68.125]      #fuel moisture content of cropped region (float)              

# OzFuel MAC 1
resized_imgs = []
for wave in ozfuel_hyper[3]:
    resized_imgs.append( ski.resize(mac1[wave, :, :], (ozfuel_dims_mac1[0], ozfuel_dims_mac1[1]), preserve_range=True))
cube = np.stack(resized_imgs, axis=2)
raw_img = np.median(cube, axis=2)
norm_img = (raw_img - np.min(raw_img)) / (np.max(raw_img) - np.min(raw_img))
ozfuel_band4_spectra.append(np.median(norm_img))
# OzFuel MAC 2
resized_imgs = []
for wave in ozfuel_hyper[3]:
    resized_imgs.append( ski.resize(mac2[wave, :, :], (ozfuel_dims_mac2[0], ozfuel_dims_mac2[1]), preserve_range=True))
cube = np.stack(resized_imgs, axis=2)
raw_img = np.median(cube, axis=2)
norm_img = (raw_img - np.min(raw_img)) / (np.max(raw_img) - np.min(raw_img))
ozfuel_band4_spectra.append(np.median(norm_img))   
# OzFuel MAC 3
resized_imgs = []
for wave in ozfuel_hyper[3]:
    resized_imgs.append( ski.resize(mac3[wave, :, :], (ozfuel_dims_mac3[0], ozfuel_dims_mac3[1]), preserve_range=True))
cube = np.stack(resized_imgs, axis=2)
raw_img = np.median(cube, axis=2)
norm_img = (raw_img - np.min(raw_img)) / (np.max(raw_img) - np.min(raw_img))
ozfuel_band4_spectra.append(np.median(norm_img))
# OzFuel MAC 4
resized_imgs = []
for wave in ozfuel_hyper[3]:
    resized_imgs.append( ski.resize(mac4[wave, :, :], (ozfuel_dims_mac4[0], ozfuel_dims_mac4[1]), preserve_range=True))
cube = np.stack(resized_imgs, axis=2)
raw_img = np.median(cube, axis=2)
norm_img = (raw_img - np.min(raw_img)) / (np.max(raw_img) - np.min(raw_img))
ozfuel_band4_spectra.append(np.median(norm_img))


#File 4
file_4 = "C:/Users/Sam/Google Drive/UNI/2022/ENGN4350/Python/FMC/20220202_Reorder_288_2351_384_U16.bin"       
with open(file_4) as f:
    dt = np.dtype('<u2')   
    rawdata = np.fromfile(f, dtype=dt)

length = 2351       #file specifc
height = 384
rawdata = np.reshape(rawdata, (len(wavelengths), length, height))  
master_data_4 = np.rot90(rawdata, 1, (1,2))

# cropping
mac1 = master_data_4[:, 0:384, 1168:1168+192] 
mac2 = master_data_4[:, 0:384, 920:920+192] 
mac3 = master_data_4[:, 60:60+324, 718:718+153] 
mac4 = master_data_4[:, 58:58+324, 640:640+75] 
lengths = [mac1.shape[2], mac2.shape[2], mac3.shape[2], mac4.shape[2]] 
heights = [mac1.shape[1], mac2.shape[1], mac3.shape[1], mac4.shape[1]] 

#calculate dimensions of simulated images according to scaling factor
ozfuel_dims_mac1 = [int(heights[0]/ozfuel_scaling), int(lengths[0]/ozfuel_scaling)]
ozfuel_dims_mac2 = [int(heights[1]/ozfuel_scaling), int(lengths[1]/ozfuel_scaling)]
ozfuel_dims_mac3 = [int(heights[2]/ozfuel_scaling), int(lengths[2]/ozfuel_scaling)]
ozfuel_dims_mac4 = [int(heights[3]/ozfuel_scaling), int(lengths[3]/ozfuel_scaling)]

measured_fmc_file4 = [112.52, 81.535, 100.495, 85.185]      #fuel moisture content of cropped region (float)              

# OzFuel MAC 1
resized_imgs = []
for wave in ozfuel_hyper[3]:
    resized_imgs.append( ski.resize(mac1[wave, :, :], (ozfuel_dims_mac1[0], ozfuel_dims_mac1[1]), preserve_range=True))
cube = np.stack(resized_imgs, axis=2)
raw_img = np.median(cube, axis=2)
norm_img = (raw_img - np.min(raw_img)) / (np.max(raw_img) - np.min(raw_img))
ozfuel_band4_spectra.append(np.median(norm_img))
# OzFuel MAC 2
resized_imgs = []
for wave in ozfuel_hyper[3]:
    resized_imgs.append( ski.resize(mac2[wave, :, :], (ozfuel_dims_mac2[0], ozfuel_dims_mac2[1]), preserve_range=True))
cube = np.stack(resized_imgs, axis=2)
raw_img = np.median(cube, axis=2)
norm_img = (raw_img - np.min(raw_img)) / (np.max(raw_img) - np.min(raw_img))
ozfuel_band4_spectra.append(np.median(norm_img))   
# OzFuel MAC 3
resized_imgs = []
for wave in ozfuel_hyper[3]:
    resized_imgs.append( ski.resize(mac3[wave, :, :], (ozfuel_dims_mac3[0], ozfuel_dims_mac3[1]), preserve_range=True))
cube = np.stack(resized_imgs, axis=2)
raw_img = np.median(cube, axis=2)
norm_img = (raw_img - np.min(raw_img)) / (np.max(raw_img) - np.min(raw_img))
ozfuel_band4_spectra.append(np.median(norm_img))
# OzFuel MAC 4
resized_imgs = []
for wave in ozfuel_hyper[3]:
    resized_imgs.append( ski.resize(mac4[wave, :, :], (ozfuel_dims_mac4[0], ozfuel_dims_mac4[1]), preserve_range=True))
cube = np.stack(resized_imgs, axis=2)
raw_img = np.median(cube, axis=2)
norm_img = (raw_img - np.min(raw_img)) / (np.max(raw_img) - np.min(raw_img))
ozfuel_band4_spectra.append(np.median(norm_img))


#File 5
file_5 = "C:/Users/Sam/Google Drive/UNI/2022/ENGN4350/Python/FMC/20211102_Reorder_288_3109_384_U16.bin"       
with open(file_5) as f:
    dt = np.dtype('<u2')   
    rawdata = np.fromfile(f, dtype=dt)

length = 3109       #file specifc
height = 384
rawdata = np.reshape(rawdata, (len(wavelengths), length, height))  
master_data_5 = np.rot90(rawdata, 1, (1,2))


# cropping
mac1 = master_data_5[:, 0:384, 1080:1080+216] 
mac2 = master_data_5[:, 0:310, 834:834+216] 
mac3 = master_data_5[:, 0:373, 659:659+153] 
mac4 = master_data_5[:, :252, 622:622+61] 
lengths = [mac1.shape[2], mac2.shape[2], mac3.shape[2], mac4.shape[2]] 
heights = [mac1.shape[1], mac2.shape[1], mac3.shape[1], mac4.shape[1]] 

#calculate dimensions of simulated images according to scaling factor
ozfuel_dims_mac1 = [int(heights[0]/ozfuel_scaling), int(lengths[0]/ozfuel_scaling)]
ozfuel_dims_mac2 = [int(heights[1]/ozfuel_scaling), int(lengths[1]/ozfuel_scaling)]
ozfuel_dims_mac3 = [int(heights[2]/ozfuel_scaling), int(lengths[2]/ozfuel_scaling)]
ozfuel_dims_mac4 = [int(heights[3]/ozfuel_scaling), int(lengths[3]/ozfuel_scaling)]

measured_fmc_file5 = [74.36, 73.41, 76.56, 63.405]      #fuel moisture content of cropped region (float)              

# OzFuel MAC 1
resized_imgs = []
for wave in ozfuel_hyper[3]:
    resized_imgs.append( ski.resize(mac1[wave, :, :], (ozfuel_dims_mac1[0], ozfuel_dims_mac1[1]), preserve_range=True))
cube = np.stack(resized_imgs, axis=2)
raw_img = np.median(cube, axis=2)
norm_img = (raw_img - np.min(raw_img)) / (np.max(raw_img) - np.min(raw_img))
ozfuel_band4_spectra.append(np.median(norm_img))
# OzFuel MAC 2
resized_imgs = []
for wave in ozfuel_hyper[3]:
    resized_imgs.append( ski.resize(mac2[wave, :, :], (ozfuel_dims_mac2[0], ozfuel_dims_mac2[1]), preserve_range=True))
cube = np.stack(resized_imgs, axis=2)
raw_img = np.median(cube, axis=2)
norm_img = (raw_img - np.min(raw_img)) / (np.max(raw_img) - np.min(raw_img))
ozfuel_band4_spectra.append(np.median(norm_img))   
# OzFuel MAC 3
resized_imgs = []
for wave in ozfuel_hyper[3]:
    resized_imgs.append( ski.resize(mac3[wave, :, :], (ozfuel_dims_mac3[0], ozfuel_dims_mac3[1]), preserve_range=True))
cube = np.stack(resized_imgs, axis=2)
raw_img = np.median(cube, axis=2)
norm_img = (raw_img - np.min(raw_img)) / (np.max(raw_img) - np.min(raw_img))
ozfuel_band4_spectra.append(np.median(norm_img))
# OzFuel MAC 4
resized_imgs = []
for wave in ozfuel_hyper[3]:
    resized_imgs.append( ski.resize(mac4[wave, :, :], (ozfuel_dims_mac4[0], ozfuel_dims_mac4[1]), preserve_range=True))
cube = np.stack(resized_imgs, axis=2)
raw_img = np.median(cube, axis=2)
norm_img = (raw_img - np.min(raw_img)) / (np.max(raw_img) - np.min(raw_img))
ozfuel_band4_spectra.append(np.median(norm_img))

# In[]: TRI OzFuel band 1

ozfuel_tri_band1_spectra = []


#File 1
file_1 = "C:/Users/Sam/Google Drive/UNI/2022/ENGN4350/Python/FMC/20220228_Reorder_288_2349_384_U16.bin"       
with open(file_1) as f:
    dt = np.dtype('<u2')   
    rawdata = np.fromfile(f, dtype=dt)

length = 2349       #file specifc
height = 384
rawdata = np.reshape(rawdata, (len(wavelengths), length, height))  
master_data_1 = np.rot90(rawdata, 1, (1,2))

# cropping
tri1 = master_data_1[:, 29:29+276, 1619:1619+255] 
tri2 = master_data_1[:, 0:293, 1857:1857+115] 
tri3 = master_data_1[:, 0:191, 1965:1965+115] 
tri4 = master_data_1[:, 0:111, 2069:2069+114] 
lengths = [tri1.shape[2], tri2.shape[2], tri3.shape[2], tri4.shape[2]] 
heights = [tri1.shape[1], tri2.shape[1], tri3.shape[1], tri4.shape[1]] 

#calculate dimensions of simulated images according to scaling factor
ozfuel_dims_tri1 = [int(heights[0]/ozfuel_scaling), int(lengths[0]/ozfuel_scaling)]
ozfuel_dims_tri2 = [int(heights[1]/ozfuel_scaling), int(lengths[1]/ozfuel_scaling)]
ozfuel_dims_tri3 = [int(heights[2]/ozfuel_scaling), int(lengths[2]/ozfuel_scaling)]
ozfuel_dims_tri4 = [int(heights[3]/ozfuel_scaling), int(lengths[3]/ozfuel_scaling)]

measured_tri_fmc_file1 = [98.775, 103.905, 115.135, 104.915]      #fuel moisture content of cropped region (float)              

# OzFuel tri 1
resized_imgs = []
for wave in ozfuel_hyper[0]:
    resized_imgs.append( ski.resize(tri1[wave, :, :], (ozfuel_dims_tri1[0], ozfuel_dims_tri1[1]), preserve_range=True))
cube = np.stack(resized_imgs, axis=2)
raw_img = np.median(cube, axis=2)
norm_img = (raw_img - np.min(raw_img)) / (np.max(raw_img) - np.min(raw_img))
ozfuel_tri_band1_spectra.append(np.median(norm_img))
# OzFuel tri 2
resized_imgs = []
for wave in ozfuel_hyper[0]:
    resized_imgs.append( ski.resize(tri2[wave, :, :], (ozfuel_dims_tri2[0], ozfuel_dims_tri2[1]), preserve_range=True))
cube = np.stack(resized_imgs, axis=2)
raw_img = np.median(cube, axis=2)
norm_img = (raw_img - np.min(raw_img)) / (np.max(raw_img) - np.min(raw_img))
ozfuel_tri_band1_spectra.append(np.median(norm_img))   
# OzFuel tri 3
resized_imgs = []
for wave in ozfuel_hyper[0]:
    resized_imgs.append( ski.resize(tri3[wave, :, :], (ozfuel_dims_tri3[0], ozfuel_dims_tri3[1]), preserve_range=True))
cube = np.stack(resized_imgs, axis=2)
raw_img = np.median(cube, axis=2)
norm_img = (raw_img - np.min(raw_img)) / (np.max(raw_img) - np.min(raw_img))
ozfuel_tri_band1_spectra.append(np.median(norm_img))
# OzFuel tri 4
resized_imgs = []
for wave in ozfuel_hyper[0]:
    resized_imgs.append( ski.resize(tri4[wave, :, :], (ozfuel_dims_tri4[0], ozfuel_dims_tri4[1]), preserve_range=True))
cube = np.stack(resized_imgs, axis=2)
raw_img = np.median(cube, axis=2)
norm_img = (raw_img - np.min(raw_img)) / (np.max(raw_img) - np.min(raw_img))
ozfuel_tri_band1_spectra.append(np.median(norm_img))
  


  
#File 2
file_2 = "C:/Users/Sam/Google Drive/UNI/2022/ENGN4350/Python/FMC/20211206_Reorder_288_2108_384_U16.bin"       
with open(file_2) as f:
    dt = np.dtype('<u2')   
    rawdata = np.fromfile(f, dtype=dt)

length = 2108       #file specifc
height = 384
rawdata = np.reshape(rawdata, (len(wavelengths), length, height))  
master_data_2 = np.rot90(rawdata, 1, (1,2))

# cropping
tri1 = master_data_2[:, 3:3+213, 1310:1310+298] 
tri2 = master_data_2[:, 0:188, 1614:1614+84] 
tri3 = master_data_2[:, 0:169, 1696:1696+93]
tri4 = master_data_2[:, 0:169, 1784:1784+93] 
lengths = [tri1.shape[2], tri2.shape[2], tri3.shape[2], tri4.shape[2]] 
heights = [tri1.shape[1], tri2.shape[1], tri3.shape[1], tri4.shape[1]] 

#calculate dimensions of simulated images according to scaling factor
ozfuel_dims_tri1 = [int(heights[0]/ozfuel_scaling), int(lengths[0]/ozfuel_scaling)]
ozfuel_dims_tri2 = [int(heights[1]/ozfuel_scaling), int(lengths[1]/ozfuel_scaling)]
ozfuel_dims_tri3 = [int(heights[2]/ozfuel_scaling), int(lengths[2]/ozfuel_scaling)]
ozfuel_dims_tri4 = [int(heights[3]/ozfuel_scaling), int(lengths[3]/ozfuel_scaling)]

measured_tri_fmc_file2 = [114.135, 122.94, 148.54, 150.73]      #fuel moisture content of cropped region (float)              

# OzFuel tri 1
resized_imgs = []
for wave in ozfuel_hyper[0]:
    resized_imgs.append( ski.resize(tri1[wave, :, :], (ozfuel_dims_tri1[0], ozfuel_dims_tri1[1]), preserve_range=True))
cube = np.stack(resized_imgs, axis=2)
raw_img = np.median(cube, axis=2)
norm_img = (raw_img - np.min(raw_img)) / (np.max(raw_img) - np.min(raw_img))
ozfuel_tri_band1_spectra.append(np.median(norm_img))
# OzFuel tri 2
resized_imgs = []
for wave in ozfuel_hyper[0]:
    resized_imgs.append( ski.resize(tri2[wave, :, :], (ozfuel_dims_tri2[0], ozfuel_dims_tri2[1]), preserve_range=True))
cube = np.stack(resized_imgs, axis=2)
raw_img = np.median(cube, axis=2)
norm_img = (raw_img - np.min(raw_img)) / (np.max(raw_img) - np.min(raw_img))
ozfuel_tri_band1_spectra.append(np.median(norm_img))   
# OzFuel tri 3
resized_imgs = []
for wave in ozfuel_hyper[0]:
    resized_imgs.append( ski.resize(tri3[wave, :, :], (ozfuel_dims_tri3[0], ozfuel_dims_tri3[1]), preserve_range=True))
cube = np.stack(resized_imgs, axis=2)
raw_img = np.median(cube, axis=2)
norm_img = (raw_img - np.min(raw_img)) / (np.max(raw_img) - np.min(raw_img))
ozfuel_tri_band1_spectra.append(np.median(norm_img))
# OzFuel tri 4
resized_imgs = []
for wave in ozfuel_hyper[0]:
    resized_imgs.append( ski.resize(tri4[wave, :, :], (ozfuel_dims_tri4[0], ozfuel_dims_tri4[1]), preserve_range=True))
cube = np.stack(resized_imgs, axis=2)
raw_img = np.median(cube, axis=2)
norm_img = (raw_img - np.min(raw_img)) / (np.max(raw_img) - np.min(raw_img))
ozfuel_tri_band1_spectra.append(np.median(norm_img))
     

#File 3
file_3 = "C:/Users/Sam/Google Drive/UNI/2022/ENGN4350/Python/FMC/20220113_Reorder_288_2431_384_U16.bin"       
with open(file_3) as f:
    dt = np.dtype('<u2')   
    rawdata = np.fromfile(f, dtype=dt)

length = 2431       #file specifc
height = 384
rawdata = np.reshape(rawdata, (len(wavelengths), length, height))  
master_data_3 = np.rot90(rawdata, 1, (1,2))

# cropping
tri1 = master_data_3[:, 28:28+352, 1466:1466+291] 
tri2 = master_data_3[:, 8:271, 1727:1727+291] 
tri3 = master_data_3[:, 0:193, 1853:1853+106] 
tri4 = master_data_3[:, 0:139, 1958:1958+106] 
lengths = [tri1.shape[2], tri2.shape[2], tri3.shape[2], tri4.shape[2]] 
heights = [tri1.shape[1], tri2.shape[1], tri3.shape[1], tri4.shape[1]] 

#calculate dimensions of simulated images according to scaling factor
ozfuel_dims_tri1 = [int(heights[0]/ozfuel_scaling), int(lengths[0]/ozfuel_scaling)]
ozfuel_dims_tri2 = [int(heights[1]/ozfuel_scaling), int(lengths[1]/ozfuel_scaling)]
ozfuel_dims_tri3 = [int(heights[2]/ozfuel_scaling), int(lengths[2]/ozfuel_scaling)]
ozfuel_dims_tri4 = [int(heights[3]/ozfuel_scaling), int(lengths[3]/ozfuel_scaling)]

measured_tri_fmc_file3 = [98.975, 131.19, 151.045, 140.86]      #fuel moisture content of cropped region (float)              

# OzFuel tri 1
resized_imgs = []
for wave in ozfuel_hyper[0]:
    resized_imgs.append( ski.resize(tri1[wave, :, :], (ozfuel_dims_tri1[0], ozfuel_dims_tri1[1]), preserve_range=True))
cube = np.stack(resized_imgs, axis=2)
raw_img = np.median(cube, axis=2)
norm_img = (raw_img - np.min(raw_img)) / (np.max(raw_img) - np.min(raw_img))
ozfuel_tri_band1_spectra.append(np.median(norm_img))
# OzFuel tri 2
resized_imgs = []
for wave in ozfuel_hyper[0]:
    resized_imgs.append( ski.resize(tri2[wave, :, :], (ozfuel_dims_tri2[0], ozfuel_dims_tri2[1]), preserve_range=True))
cube = np.stack(resized_imgs, axis=2)
raw_img = np.median(cube, axis=2)
norm_img = (raw_img - np.min(raw_img)) / (np.max(raw_img) - np.min(raw_img))
ozfuel_tri_band1_spectra.append(np.median(norm_img))   
# OzFuel tri 3
resized_imgs = []
for wave in ozfuel_hyper[0]:
    resized_imgs.append( ski.resize(tri3[wave, :, :], (ozfuel_dims_tri3[0], ozfuel_dims_tri3[1]), preserve_range=True))
cube = np.stack(resized_imgs, axis=2)
raw_img = np.median(cube, axis=2)
norm_img = (raw_img - np.min(raw_img)) / (np.max(raw_img) - np.min(raw_img))
ozfuel_tri_band1_spectra.append(np.median(norm_img))
# OzFuel tri 4
resized_imgs = []
for wave in ozfuel_hyper[0]:
    resized_imgs.append( ski.resize(tri4[wave, :, :], (ozfuel_dims_tri4[0], ozfuel_dims_tri4[1]), preserve_range=True))
cube = np.stack(resized_imgs, axis=2)
raw_img = np.median(cube, axis=2)
norm_img = (raw_img - np.min(raw_img)) / (np.max(raw_img) - np.min(raw_img))
ozfuel_tri_band1_spectra.append(np.median(norm_img))





#File 4
file_4 = "C:/Users/Sam/Google Drive/UNI/2022/ENGN4350/Python/FMC/20220202_Reorder_288_2351_384_U16.bin"       
with open(file_4) as f:
    dt = np.dtype('<u2')   
    rawdata = np.fromfile(f, dtype=dt)

length = 2351       #file specifc
height = 384
rawdata = np.reshape(rawdata, (len(wavelengths), length, height))  
master_data_4 = np.rot90(rawdata, 1, (1,2))

# cropping
tri1 = master_data_4[:, 0:384, 1511:1511+245] 
tri2 = master_data_4[:, 0:248, 1773:1773+118] 
tri3 = master_data_4[:, 0:194, 1888:1888+112] 
tri4 = master_data_4[:, 0:109, 2000:2083] 
lengths = [tri1.shape[2], tri2.shape[2], tri3.shape[2], tri4.shape[2]] 
heights = [tri1.shape[1], tri2.shape[1], tri3.shape[1], tri4.shape[1]] 

#calculate dimensions of simulated images according to scaling factor
ozfuel_dims_tri1 = [int(heights[0]/ozfuel_scaling), int(lengths[0]/ozfuel_scaling)]
ozfuel_dims_tri2 = [int(heights[1]/ozfuel_scaling), int(lengths[1]/ozfuel_scaling)]
ozfuel_dims_tri3 = [int(heights[2]/ozfuel_scaling), int(lengths[2]/ozfuel_scaling)]
ozfuel_dims_tri4 = [int(heights[3]/ozfuel_scaling), int(lengths[3]/ozfuel_scaling)]

measured_tri_fmc_file4 = [113.735, 111.975, 107.355, 125.405]      #fuel moisture content of cropped region (float)              

# OzFuel tri 1
resized_imgs = []
for wave in ozfuel_hyper[0]:
    resized_imgs.append( ski.resize(tri1[wave, :, :], (ozfuel_dims_tri1[0], ozfuel_dims_tri1[1]), preserve_range=True))
cube = np.stack(resized_imgs, axis=2)
raw_img = np.median(cube, axis=2)
norm_img = (raw_img - np.min(raw_img)) / (np.max(raw_img) - np.min(raw_img))
ozfuel_tri_band1_spectra.append(np.median(norm_img))
# OzFuel tri 2
resized_imgs = []
for wave in ozfuel_hyper[0]:
    resized_imgs.append( ski.resize(tri2[wave, :, :], (ozfuel_dims_tri2[0], ozfuel_dims_tri2[1]), preserve_range=True))
cube = np.stack(resized_imgs, axis=2)
raw_img = np.median(cube, axis=2)
norm_img = (raw_img - np.min(raw_img)) / (np.max(raw_img) - np.min(raw_img))
ozfuel_tri_band1_spectra.append(np.median(norm_img))   
# OzFuel tri 3
resized_imgs = []
for wave in ozfuel_hyper[0]:
    resized_imgs.append( ski.resize(tri3[wave, :, :], (ozfuel_dims_tri3[0], ozfuel_dims_tri3[1]), preserve_range=True))
cube = np.stack(resized_imgs, axis=2)
raw_img = np.median(cube, axis=2)
norm_img = (raw_img - np.min(raw_img)) / (np.max(raw_img) - np.min(raw_img))
ozfuel_tri_band1_spectra.append(np.median(norm_img))
# OzFuel tri 4
resized_imgs = []
for wave in ozfuel_hyper[0]:
    resized_imgs.append( ski.resize(tri4[wave, :, :], (ozfuel_dims_tri4[0], ozfuel_dims_tri4[1]), preserve_range=True))
cube = np.stack(resized_imgs, axis=2)
raw_img = np.median(cube, axis=2)
norm_img = (raw_img - np.min(raw_img)) / (np.max(raw_img) - np.min(raw_img))
ozfuel_tri_band1_spectra.append(np.median(norm_img))



#File 5
file_5 = "C:/Users/Sam/Google Drive/UNI/2022/ENGN4350/Python/FMC/20211102_Reorder_288_3109_384_U16.bin"       
with open(file_5) as f:
    dt = np.dtype('<u2')   
    rawdata = np.fromfile(f, dtype=dt)

length = 3109       #file specifc
height = 384
rawdata = np.reshape(rawdata, (len(wavelengths), length, height))  
master_data_5 = np.rot90(rawdata, 1, (1,2))


# cropping
tri1 = master_data_5[:, 0:255, 1500:1500+280] 
tri2 = master_data_5[:, 0:212, 1786:1786+122] 
tri3 = master_data_5[:, 0:148, 1906:1906+131] 
tri4 = master_data_5[:, 0:56,  2042:2042+108] 
lengths = [tri1.shape[2], tri2.shape[2], tri3.shape[2], tri4.shape[2]] 
heights = [tri1.shape[1], tri2.shape[1], tri3.shape[1], tri4.shape[1]] 

#calculate dimensions of simulated images according to scaling factor
ozfuel_dims_tri1 = [int(heights[0]/ozfuel_scaling), int(lengths[0]/ozfuel_scaling)]
ozfuel_dims_tri2 = [int(heights[1]/ozfuel_scaling), int(lengths[1]/ozfuel_scaling)]
ozfuel_dims_tri3 = [int(heights[2]/ozfuel_scaling), int(lengths[2]/ozfuel_scaling)]
ozfuel_dims_tri4 = [int(heights[3]/ozfuel_scaling), int(lengths[3]/ozfuel_scaling)]

measured_tri_fmc_file5 = [87.79, 90.13, 86.57, 102.78]      #fuel moisture content of cropped region (float)              

# OzFuel tri 1
resized_imgs = []
for wave in ozfuel_hyper[0]:
    resized_imgs.append( ski.resize(tri1[wave, :, :], (ozfuel_dims_tri1[0], ozfuel_dims_tri1[1]), preserve_range=True))
cube = np.stack(resized_imgs, axis=2)
raw_img = np.median(cube, axis=2)
norm_img = (raw_img - np.min(raw_img)) / (np.max(raw_img) - np.min(raw_img))
ozfuel_tri_band1_spectra.append(np.median(norm_img))
# OzFuel tri 2
resized_imgs = []
for wave in ozfuel_hyper[0]:
    resized_imgs.append( ski.resize(tri2[wave, :, :], (ozfuel_dims_tri2[0], ozfuel_dims_tri2[1]), preserve_range=True))
cube = np.stack(resized_imgs, axis=2)
raw_img = np.median(cube, axis=2)
norm_img = (raw_img - np.min(raw_img)) / (np.max(raw_img) - np.min(raw_img))
ozfuel_tri_band1_spectra.append(np.median(norm_img))   
# OzFuel tri 3
resized_imgs = []
for wave in ozfuel_hyper[0]:
    resized_imgs.append( ski.resize(tri3[wave, :, :], (ozfuel_dims_tri3[0], ozfuel_dims_tri3[1]), preserve_range=True))
cube = np.stack(resized_imgs, axis=2)
raw_img = np.median(cube, axis=2)
norm_img = (raw_img - np.min(raw_img)) / (np.max(raw_img) - np.min(raw_img))
ozfuel_tri_band1_spectra.append(np.median(norm_img))
# OzFuel tri 4
resized_imgs = []
for wave in ozfuel_hyper[0]:
    resized_imgs.append( ski.resize(tri4[wave, :, :], (ozfuel_dims_tri4[0], ozfuel_dims_tri4[1]), preserve_range=True))
cube = np.stack(resized_imgs, axis=2)
raw_img = np.median(cube, axis=2)
norm_img = (raw_img - np.min(raw_img)) / (np.max(raw_img) - np.min(raw_img))
ozfuel_tri_band1_spectra.append(np.median(norm_img))

# In[]: TRI OzFuel band 2

ozfuel_tri_band2_spectra = []


#File 1
file_1 = "C:/Users/Sam/Google Drive/UNI/2022/ENGN4350/Python/FMC/20220228_Reorder_288_2349_384_U16.bin"       
with open(file_1) as f:
    dt = np.dtype('<u2')   
    rawdata = np.fromfile(f, dtype=dt)

length = 2349       #file specifc
height = 384
rawdata = np.reshape(rawdata, (len(wavelengths), length, height))  
master_data_1 = np.rot90(rawdata, 1, (1,2))

# cropping
tri1 = master_data_1[:, 29:29+276, 1619:1619+255] 
tri2 = master_data_1[:, 0:293, 1857:1857+115] 
tri3 = master_data_1[:, 0:191, 1965:1965+115] 
tri4 = master_data_1[:, 0:111, 2069:2069+114] 
lengths = [tri1.shape[2], tri2.shape[2], tri3.shape[2], tri4.shape[2]] 
heights = [tri1.shape[1], tri2.shape[1], tri3.shape[1], tri4.shape[1]] 

#calculate dimensions of simulated images according to scaling factor
ozfuel_dims_tri1 = [int(heights[0]/ozfuel_scaling), int(lengths[0]/ozfuel_scaling)]
ozfuel_dims_tri2 = [int(heights[1]/ozfuel_scaling), int(lengths[1]/ozfuel_scaling)]
ozfuel_dims_tri3 = [int(heights[2]/ozfuel_scaling), int(lengths[2]/ozfuel_scaling)]
ozfuel_dims_tri4 = [int(heights[3]/ozfuel_scaling), int(lengths[3]/ozfuel_scaling)]

measured_tri_fmc_file1 = [98.775, 103.905, 115.135, 104.915]      #fuel moisture content of cropped region (float)              

# OzFuel tri 1
resized_imgs = []
for wave in ozfuel_hyper[1]:
    resized_imgs.append( ski.resize(tri1[wave, :, :], (ozfuel_dims_tri1[0], ozfuel_dims_tri1[1]), preserve_range=True))
cube = np.stack(resized_imgs, axis=2)
raw_img = np.median(cube, axis=2)
norm_img = (raw_img - np.min(raw_img)) / (np.max(raw_img) - np.min(raw_img))
ozfuel_tri_band2_spectra.append(np.median(norm_img))
# OzFuel tri 2
resized_imgs = []
for wave in ozfuel_hyper[1]:
    resized_imgs.append( ski.resize(tri2[wave, :, :], (ozfuel_dims_tri2[0], ozfuel_dims_tri2[1]), preserve_range=True))
cube = np.stack(resized_imgs, axis=2)
raw_img = np.median(cube, axis=2)
norm_img = (raw_img - np.min(raw_img)) / (np.max(raw_img) - np.min(raw_img))
ozfuel_tri_band2_spectra.append(np.median(norm_img))   
# OzFuel tri 3
resized_imgs = []
for wave in ozfuel_hyper[1]:
    resized_imgs.append( ski.resize(tri3[wave, :, :], (ozfuel_dims_tri3[0], ozfuel_dims_tri3[1]), preserve_range=True))
cube = np.stack(resized_imgs, axis=2)
raw_img = np.median(cube, axis=2)
norm_img = (raw_img - np.min(raw_img)) / (np.max(raw_img) - np.min(raw_img))
ozfuel_tri_band2_spectra.append(np.median(norm_img))
# OzFuel tri 4
resized_imgs = []
for wave in ozfuel_hyper[1]:
    resized_imgs.append( ski.resize(tri4[wave, :, :], (ozfuel_dims_tri4[0], ozfuel_dims_tri4[1]), preserve_range=True))
cube = np.stack(resized_imgs, axis=2)
raw_img = np.median(cube, axis=2)
norm_img = (raw_img - np.min(raw_img)) / (np.max(raw_img) - np.min(raw_img))
ozfuel_tri_band2_spectra.append(np.median(norm_img))
  


  
#File 2
file_2 = "C:/Users/Sam/Google Drive/UNI/2022/ENGN4350/Python/FMC/20211206_Reorder_288_2108_384_U16.bin"       
with open(file_2) as f:
    dt = np.dtype('<u2')   
    rawdata = np.fromfile(f, dtype=dt)

length = 2108       #file specifc
height = 384
rawdata = np.reshape(rawdata, (len(wavelengths), length, height))  
master_data_2 = np.rot90(rawdata, 1, (1,2))

# cropping
tri1 = master_data_2[:, 3:3+213, 1310:1310+298] 
tri2 = master_data_2[:, 0:188, 1614:1614+84] 
tri3 = master_data_2[:, 0:169, 1696:1696+93]
tri4 = master_data_2[:, 0:169, 1784:1784+93] 
lengths = [tri1.shape[2], tri2.shape[2], tri3.shape[2], tri4.shape[2]] 
heights = [tri1.shape[1], tri2.shape[1], tri3.shape[1], tri4.shape[1]] 

#calculate dimensions of simulated images according to scaling factor
ozfuel_dims_tri1 = [int(heights[0]/ozfuel_scaling), int(lengths[0]/ozfuel_scaling)]
ozfuel_dims_tri2 = [int(heights[1]/ozfuel_scaling), int(lengths[1]/ozfuel_scaling)]
ozfuel_dims_tri3 = [int(heights[2]/ozfuel_scaling), int(lengths[2]/ozfuel_scaling)]
ozfuel_dims_tri4 = [int(heights[3]/ozfuel_scaling), int(lengths[3]/ozfuel_scaling)]

measured_tri_fmc_file2 = [114.135, 122.94, 148.54, 150.73]      #fuel moisture content of cropped region (float)              

# OzFuel tri 1
resized_imgs = []
for wave in ozfuel_hyper[1]:
    resized_imgs.append( ski.resize(tri1[wave, :, :], (ozfuel_dims_tri1[0], ozfuel_dims_tri1[1]), preserve_range=True))
cube = np.stack(resized_imgs, axis=2)
raw_img = np.median(cube, axis=2)
norm_img = (raw_img - np.min(raw_img)) / (np.max(raw_img) - np.min(raw_img))
ozfuel_tri_band2_spectra.append(np.median(norm_img))
# OzFuel tri 2
resized_imgs = []
for wave in ozfuel_hyper[1]:
    resized_imgs.append( ski.resize(tri2[wave, :, :], (ozfuel_dims_tri2[0], ozfuel_dims_tri2[1]), preserve_range=True))
cube = np.stack(resized_imgs, axis=2)
raw_img = np.median(cube, axis=2)
norm_img = (raw_img - np.min(raw_img)) / (np.max(raw_img) - np.min(raw_img))
ozfuel_tri_band2_spectra.append(np.median(norm_img))   
# OzFuel tri 3
resized_imgs = []
for wave in ozfuel_hyper[1]:
    resized_imgs.append( ski.resize(tri3[wave, :, :], (ozfuel_dims_tri3[0], ozfuel_dims_tri3[1]), preserve_range=True))
cube = np.stack(resized_imgs, axis=2)
raw_img = np.median(cube, axis=2)
norm_img = (raw_img - np.min(raw_img)) / (np.max(raw_img) - np.min(raw_img))
ozfuel_tri_band2_spectra.append(np.median(norm_img))
# OzFuel tri 4
resized_imgs = []
for wave in ozfuel_hyper[1]:
    resized_imgs.append( ski.resize(tri4[wave, :, :], (ozfuel_dims_tri4[0], ozfuel_dims_tri4[1]), preserve_range=True))
cube = np.stack(resized_imgs, axis=2)
raw_img = np.median(cube, axis=2)
norm_img = (raw_img - np.min(raw_img)) / (np.max(raw_img) - np.min(raw_img))
ozfuel_tri_band2_spectra.append(np.median(norm_img))
     

#File 3
file_3 = "C:/Users/Sam/Google Drive/UNI/2022/ENGN4350/Python/FMC/20220113_Reorder_288_2431_384_U16.bin"       
with open(file_3) as f:
    dt = np.dtype('<u2')   
    rawdata = np.fromfile(f, dtype=dt)

length = 2431       #file specifc
height = 384
rawdata = np.reshape(rawdata, (len(wavelengths), length, height))  
master_data_3 = np.rot90(rawdata, 1, (1,2))

# cropping
tri1 = master_data_3[:, 28:28+352, 1466:1466+291] 
tri2 = master_data_3[:, 8:271, 1727:1727+291] 
tri3 = master_data_3[:, 0:193, 1853:1853+106] 
tri4 = master_data_3[:, 0:139, 1958:1958+106] 
lengths = [tri1.shape[2], tri2.shape[2], tri3.shape[2], tri4.shape[2]] 
heights = [tri1.shape[1], tri2.shape[1], tri3.shape[1], tri4.shape[1]] 

#calculate dimensions of simulated images according to scaling factor
ozfuel_dims_tri1 = [int(heights[0]/ozfuel_scaling), int(lengths[0]/ozfuel_scaling)]
ozfuel_dims_tri2 = [int(heights[1]/ozfuel_scaling), int(lengths[1]/ozfuel_scaling)]
ozfuel_dims_tri3 = [int(heights[2]/ozfuel_scaling), int(lengths[2]/ozfuel_scaling)]
ozfuel_dims_tri4 = [int(heights[3]/ozfuel_scaling), int(lengths[3]/ozfuel_scaling)]

measured_tri_fmc_file3 = [98.975, 131.19, 151.045, 140.86]      #fuel moisture content of cropped region (float)              

# OzFuel tri 1
resized_imgs = []
for wave in ozfuel_hyper[1]:
    resized_imgs.append( ski.resize(tri1[wave, :, :], (ozfuel_dims_tri1[0], ozfuel_dims_tri1[1]), preserve_range=True))
cube = np.stack(resized_imgs, axis=2)
raw_img = np.median(cube, axis=2)
norm_img = (raw_img - np.min(raw_img)) / (np.max(raw_img) - np.min(raw_img))
ozfuel_tri_band2_spectra.append(np.median(norm_img))
# OzFuel tri 2
resized_imgs = []
for wave in ozfuel_hyper[1]:
    resized_imgs.append( ski.resize(tri2[wave, :, :], (ozfuel_dims_tri2[0], ozfuel_dims_tri2[1]), preserve_range=True))
cube = np.stack(resized_imgs, axis=2)
raw_img = np.median(cube, axis=2)
norm_img = (raw_img - np.min(raw_img)) / (np.max(raw_img) - np.min(raw_img))
ozfuel_tri_band2_spectra.append(np.median(norm_img))   
# OzFuel tri 3
resized_imgs = []
for wave in ozfuel_hyper[1]:
    resized_imgs.append( ski.resize(tri3[wave, :, :], (ozfuel_dims_tri3[0], ozfuel_dims_tri3[1]), preserve_range=True))
cube = np.stack(resized_imgs, axis=2)
raw_img = np.median(cube, axis=2)
norm_img = (raw_img - np.min(raw_img)) / (np.max(raw_img) - np.min(raw_img))
ozfuel_tri_band2_spectra.append(np.median(norm_img))
# OzFuel tri 4
resized_imgs = []
for wave in ozfuel_hyper[1]:
    resized_imgs.append( ski.resize(tri4[wave, :, :], (ozfuel_dims_tri4[0], ozfuel_dims_tri4[1]), preserve_range=True))
cube = np.stack(resized_imgs, axis=2)
raw_img = np.median(cube, axis=2)
norm_img = (raw_img - np.min(raw_img)) / (np.max(raw_img) - np.min(raw_img))
ozfuel_tri_band2_spectra.append(np.median(norm_img))



#File 4
file_4 = "C:/Users/Sam/Google Drive/UNI/2022/ENGN4350/Python/FMC/20220202_Reorder_288_2351_384_U16.bin"       
with open(file_4) as f:
    dt = np.dtype('<u2')   
    rawdata = np.fromfile(f, dtype=dt)

length = 2351       #file specifc
height = 384
rawdata = np.reshape(rawdata, (len(wavelengths), length, height))  
master_data_4 = np.rot90(rawdata, 1, (1,2))

# cropping
tri1 = master_data_4[:, 0:384, 1511:1511+245] 
tri2 = master_data_4[:, 0:248, 1773:1773+118] 
tri3 = master_data_4[:, 0:194, 1888:1888+112] 
tri4 = master_data_4[:, 0:109, 2000:2083] 
lengths = [tri1.shape[2], tri2.shape[2], tri3.shape[2], tri4.shape[2]] 
heights = [tri1.shape[1], tri2.shape[1], tri3.shape[1], tri4.shape[1]] 

#calculate dimensions of simulated images according to scaling factor
ozfuel_dims_tri1 = [int(heights[0]/ozfuel_scaling), int(lengths[0]/ozfuel_scaling)]
ozfuel_dims_tri2 = [int(heights[1]/ozfuel_scaling), int(lengths[1]/ozfuel_scaling)]
ozfuel_dims_tri3 = [int(heights[2]/ozfuel_scaling), int(lengths[2]/ozfuel_scaling)]
ozfuel_dims_tri4 = [int(heights[3]/ozfuel_scaling), int(lengths[3]/ozfuel_scaling)]

measured_tri_fmc_file4 = [113.735, 111.975, 107.355, 125.405]      #fuel moisture content of cropped region (float)              

# OzFuel tri 1
resized_imgs = []
for wave in ozfuel_hyper[1]:
    resized_imgs.append( ski.resize(tri1[wave, :, :], (ozfuel_dims_tri1[0], ozfuel_dims_tri1[1]), preserve_range=True))
cube = np.stack(resized_imgs, axis=2)
raw_img = np.median(cube, axis=2)
norm_img = (raw_img - np.min(raw_img)) / (np.max(raw_img) - np.min(raw_img))
ozfuel_tri_band2_spectra.append(np.median(norm_img))
# OzFuel tri 2
resized_imgs = []
for wave in ozfuel_hyper[1]:
    resized_imgs.append( ski.resize(tri2[wave, :, :], (ozfuel_dims_tri2[0], ozfuel_dims_tri2[1]), preserve_range=True))
cube = np.stack(resized_imgs, axis=2)
raw_img = np.median(cube, axis=2)
norm_img = (raw_img - np.min(raw_img)) / (np.max(raw_img) - np.min(raw_img))
ozfuel_tri_band2_spectra.append(np.median(norm_img))   
# OzFuel tri 3
resized_imgs = []
for wave in ozfuel_hyper[1]:
    resized_imgs.append( ski.resize(tri3[wave, :, :], (ozfuel_dims_tri3[0], ozfuel_dims_tri3[1]), preserve_range=True))
cube = np.stack(resized_imgs, axis=2)
raw_img = np.median(cube, axis=2)
norm_img = (raw_img - np.min(raw_img)) / (np.max(raw_img) - np.min(raw_img))
ozfuel_tri_band2_spectra.append(np.median(norm_img))
# OzFuel tri 4
resized_imgs = []
for wave in ozfuel_hyper[1]:
    resized_imgs.append( ski.resize(tri4[wave, :, :], (ozfuel_dims_tri4[0], ozfuel_dims_tri4[1]), preserve_range=True))
cube = np.stack(resized_imgs, axis=2)
raw_img = np.median(cube, axis=2)
norm_img = (raw_img - np.min(raw_img)) / (np.max(raw_img) - np.min(raw_img))
ozfuel_tri_band2_spectra.append(np.median(norm_img))




#File 5
file_5 = "C:/Users/Sam/Google Drive/UNI/2022/ENGN4350/Python/FMC/20211102_Reorder_288_3109_384_U16.bin"       
with open(file_5) as f:
    dt = np.dtype('<u2')   
    rawdata = np.fromfile(f, dtype=dt)

length = 3109       #file specifc
height = 384
rawdata = np.reshape(rawdata, (len(wavelengths), length, height))  
master_data_5 = np.rot90(rawdata, 1, (1,2))


# cropping
tri1 = master_data_5[:, 0:255, 1500:1500+280] 
tri2 = master_data_5[:, 0:212, 1786:1786+122] 
tri3 = master_data_5[:, 0:148, 1906:1906+131] 
tri4 = master_data_5[:, 0:56,  2042:2042+108] 
lengths = [tri1.shape[2], tri2.shape[2], tri3.shape[2], tri4.shape[2]] 
heights = [tri1.shape[1], tri2.shape[1], tri3.shape[1], tri4.shape[1]] 

#calculate dimensions of simulated images according to scaling factor
ozfuel_dims_tri1 = [int(heights[0]/ozfuel_scaling), int(lengths[0]/ozfuel_scaling)]
ozfuel_dims_tri2 = [int(heights[1]/ozfuel_scaling), int(lengths[1]/ozfuel_scaling)]
ozfuel_dims_tri3 = [int(heights[2]/ozfuel_scaling), int(lengths[2]/ozfuel_scaling)]
ozfuel_dims_tri4 = [int(heights[3]/ozfuel_scaling), int(lengths[3]/ozfuel_scaling)]

measured_tri_fmc_file5 = [87.79, 90.13, 86.57, 102.78]      #fuel moisture content of cropped region (float)              

# OzFuel tri 1
resized_imgs = []
for wave in ozfuel_hyper[1]:
    resized_imgs.append( ski.resize(tri1[wave, :, :], (ozfuel_dims_tri1[0], ozfuel_dims_tri1[1]), preserve_range=True))
cube = np.stack(resized_imgs, axis=2)
raw_img = np.median(cube, axis=2)
norm_img = (raw_img - np.min(raw_img)) / (np.max(raw_img) - np.min(raw_img))
ozfuel_tri_band2_spectra.append(np.median(norm_img))
# OzFuel tri 2
resized_imgs = []
for wave in ozfuel_hyper[1]:
    resized_imgs.append( ski.resize(tri2[wave, :, :], (ozfuel_dims_tri2[0], ozfuel_dims_tri2[1]), preserve_range=True))
cube = np.stack(resized_imgs, axis=2)
raw_img = np.median(cube, axis=2)
norm_img = (raw_img - np.min(raw_img)) / (np.max(raw_img) - np.min(raw_img))
ozfuel_tri_band2_spectra.append(np.median(norm_img))   
# OzFuel tri 3
resized_imgs = []
for wave in ozfuel_hyper[1]:
    resized_imgs.append( ski.resize(tri3[wave, :, :], (ozfuel_dims_tri3[0], ozfuel_dims_tri3[1]), preserve_range=True))
cube = np.stack(resized_imgs, axis=2)
raw_img = np.median(cube, axis=2)
norm_img = (raw_img - np.min(raw_img)) / (np.max(raw_img) - np.min(raw_img))
ozfuel_tri_band2_spectra.append(np.median(norm_img))
# OzFuel tri 4
resized_imgs = []
for wave in ozfuel_hyper[1]:
    resized_imgs.append( ski.resize(tri4[wave, :, :], (ozfuel_dims_tri4[0], ozfuel_dims_tri4[1]), preserve_range=True))
cube = np.stack(resized_imgs, axis=2)
raw_img = np.median(cube, axis=2)
norm_img = (raw_img - np.min(raw_img)) / (np.max(raw_img) - np.min(raw_img))
ozfuel_tri_band2_spectra.append(np.median(norm_img))


# In[]: TRI OzFuel band 3

ozfuel_tri_band3_spectra = []


#File 1
file_1 = "C:/Users/Sam/Google Drive/UNI/2022/ENGN4350/Python/FMC/20220228_Reorder_288_2349_384_U16.bin"       
with open(file_1) as f:
    dt = np.dtype('<u2')   
    rawdata = np.fromfile(f, dtype=dt)

length = 2349       #file specifc
height = 384
rawdata = np.reshape(rawdata, (len(wavelengths), length, height))  
master_data_1 = np.rot90(rawdata, 1, (1,2))

# cropping
tri1 = master_data_1[:, 29:29+276, 1619:1619+255] 
tri2 = master_data_1[:, 0:293, 1857:1857+115] 
tri3 = master_data_1[:, 0:191, 1965:1965+115] 
tri4 = master_data_1[:, 0:111, 2069:2069+114] 
lengths = [tri1.shape[2], tri2.shape[2], tri3.shape[2], tri4.shape[2]] 
heights = [tri1.shape[1], tri2.shape[1], tri3.shape[1], tri4.shape[1]] 

#calculate dimensions of simulated images according to scaling factor
ozfuel_dims_tri1 = [int(heights[0]/ozfuel_scaling), int(lengths[0]/ozfuel_scaling)]
ozfuel_dims_tri2 = [int(heights[1]/ozfuel_scaling), int(lengths[1]/ozfuel_scaling)]
ozfuel_dims_tri3 = [int(heights[2]/ozfuel_scaling), int(lengths[2]/ozfuel_scaling)]
ozfuel_dims_tri4 = [int(heights[3]/ozfuel_scaling), int(lengths[3]/ozfuel_scaling)]

measured_tri_fmc_file1 = [98.775, 103.905, 115.135, 104.915]      #fuel moisture content of cropped region (float)              

# OzFuel tri 1
resized_imgs = []
for wave in ozfuel_hyper[2]:
    resized_imgs.append( ski.resize(tri1[wave, :, :], (ozfuel_dims_tri1[0], ozfuel_dims_tri1[1]), preserve_range=True))
cube = np.stack(resized_imgs, axis=2)
raw_img = np.median(cube, axis=2)
norm_img = (raw_img - np.min(raw_img)) / (np.max(raw_img) - np.min(raw_img))
ozfuel_tri_band3_spectra.append(np.median(norm_img))
# OzFuel tri 2
resized_imgs = []
for wave in ozfuel_hyper[2]:
    resized_imgs.append( ski.resize(tri2[wave, :, :], (ozfuel_dims_tri2[0], ozfuel_dims_tri2[1]), preserve_range=True))
cube = np.stack(resized_imgs, axis=2)
raw_img = np.median(cube, axis=2)
norm_img = (raw_img - np.min(raw_img)) / (np.max(raw_img) - np.min(raw_img))
ozfuel_tri_band3_spectra.append(np.median(norm_img))   
# OzFuel tri 3
resized_imgs = []
for wave in ozfuel_hyper[2]:
    resized_imgs.append( ski.resize(tri3[wave, :, :], (ozfuel_dims_tri3[0], ozfuel_dims_tri3[1]), preserve_range=True))
cube = np.stack(resized_imgs, axis=2)
raw_img = np.median(cube, axis=2)
norm_img = (raw_img - np.min(raw_img)) / (np.max(raw_img) - np.min(raw_img))
ozfuel_tri_band3_spectra.append(np.median(norm_img))
# OzFuel tri 4
resized_imgs = []
for wave in ozfuel_hyper[2]:
    resized_imgs.append( ski.resize(tri4[wave, :, :], (ozfuel_dims_tri4[0], ozfuel_dims_tri4[1]), preserve_range=True))
cube = np.stack(resized_imgs, axis=2)
raw_img = np.median(cube, axis=2)
norm_img = (raw_img - np.min(raw_img)) / (np.max(raw_img) - np.min(raw_img))
ozfuel_tri_band3_spectra.append(np.median(norm_img))
  


  
#File 2
file_2 = "C:/Users/Sam/Google Drive/UNI/2022/ENGN4350/Python/FMC/20211206_Reorder_288_2108_384_U16.bin"       
with open(file_2) as f:
    dt = np.dtype('<u2')   
    rawdata = np.fromfile(f, dtype=dt)

length = 2108       #file specifc
height = 384
rawdata = np.reshape(rawdata, (len(wavelengths), length, height))  
master_data_2 = np.rot90(rawdata, 1, (1,2))

# cropping
tri1 = master_data_2[:, 3:3+213, 1310:1310+298] 
tri2 = master_data_2[:, 0:188, 1614:1614+84] 
tri3 = master_data_2[:, 0:169, 1696:1696+93]
tri4 = master_data_2[:, 0:169, 1784:1784+93] 
lengths = [tri1.shape[2], tri2.shape[2], tri3.shape[2], tri4.shape[2]] 
heights = [tri1.shape[1], tri2.shape[1], tri3.shape[1], tri4.shape[1]] 

#calculate dimensions of simulated images according to scaling factor
ozfuel_dims_tri1 = [int(heights[0]/ozfuel_scaling), int(lengths[0]/ozfuel_scaling)]
ozfuel_dims_tri2 = [int(heights[1]/ozfuel_scaling), int(lengths[1]/ozfuel_scaling)]
ozfuel_dims_tri3 = [int(heights[2]/ozfuel_scaling), int(lengths[2]/ozfuel_scaling)]
ozfuel_dims_tri4 = [int(heights[3]/ozfuel_scaling), int(lengths[3]/ozfuel_scaling)]

measured_tri_fmc_file2 = [114.135, 122.94, 148.54, 150.73]      #fuel moisture content of cropped region (float)              

# OzFuel tri 1
resized_imgs = []
for wave in ozfuel_hyper[2]:
    resized_imgs.append( ski.resize(tri1[wave, :, :], (ozfuel_dims_tri1[0], ozfuel_dims_tri1[1]), preserve_range=True))
cube = np.stack(resized_imgs, axis=2)
raw_img = np.median(cube, axis=2)
norm_img = (raw_img - np.min(raw_img)) / (np.max(raw_img) - np.min(raw_img))
ozfuel_tri_band3_spectra.append(np.median(norm_img))
# OzFuel tri 2
resized_imgs = []
for wave in ozfuel_hyper[2]:
    resized_imgs.append( ski.resize(tri2[wave, :, :], (ozfuel_dims_tri2[0], ozfuel_dims_tri2[1]), preserve_range=True))
cube = np.stack(resized_imgs, axis=2)
raw_img = np.median(cube, axis=2)
norm_img = (raw_img - np.min(raw_img)) / (np.max(raw_img) - np.min(raw_img))
ozfuel_tri_band3_spectra.append(np.median(norm_img))   
# OzFuel tri 3
resized_imgs = []
for wave in ozfuel_hyper[2]:
    resized_imgs.append( ski.resize(tri3[wave, :, :], (ozfuel_dims_tri3[0], ozfuel_dims_tri3[1]), preserve_range=True))
cube = np.stack(resized_imgs, axis=2)
raw_img = np.median(cube, axis=2)
norm_img = (raw_img - np.min(raw_img)) / (np.max(raw_img) - np.min(raw_img))
ozfuel_tri_band3_spectra.append(np.median(norm_img))
# OzFuel tri 4
resized_imgs = []
for wave in ozfuel_hyper[2]:
    resized_imgs.append( ski.resize(tri4[wave, :, :], (ozfuel_dims_tri4[0], ozfuel_dims_tri4[1]), preserve_range=True))
cube = np.stack(resized_imgs, axis=2)
raw_img = np.median(cube, axis=2)
norm_img = (raw_img - np.min(raw_img)) / (np.max(raw_img) - np.min(raw_img))
ozfuel_tri_band3_spectra.append(np.median(norm_img))
     

#File 3
file_3 = "C:/Users/Sam/Google Drive/UNI/2022/ENGN4350/Python/FMC/20220113_Reorder_288_2431_384_U16.bin"       
with open(file_3) as f:
    dt = np.dtype('<u2')   
    rawdata = np.fromfile(f, dtype=dt)

length = 2431       #file specifc
height = 384
rawdata = np.reshape(rawdata, (len(wavelengths), length, height))  
master_data_3 = np.rot90(rawdata, 1, (1,2))

# cropping
tri1 = master_data_3[:, 28:28+352, 1466:1466+291] 
tri2 = master_data_3[:, 8:271, 1727:1727+291] 
tri3 = master_data_3[:, 0:193, 1853:1853+106] 
tri4 = master_data_3[:, 0:139, 1958:1958+106] 
lengths = [tri1.shape[2], tri2.shape[2], tri3.shape[2], tri4.shape[2]] 
heights = [tri1.shape[1], tri2.shape[1], tri3.shape[1], tri4.shape[1]] 

#calculate dimensions of simulated images according to scaling factor
ozfuel_dims_tri1 = [int(heights[0]/ozfuel_scaling), int(lengths[0]/ozfuel_scaling)]
ozfuel_dims_tri2 = [int(heights[1]/ozfuel_scaling), int(lengths[1]/ozfuel_scaling)]
ozfuel_dims_tri3 = [int(heights[2]/ozfuel_scaling), int(lengths[2]/ozfuel_scaling)]
ozfuel_dims_tri4 = [int(heights[3]/ozfuel_scaling), int(lengths[3]/ozfuel_scaling)]

measured_tri_fmc_file3 = [98.975, 131.19, 151.045, 140.86]      #fuel moisture content of cropped region (float)              

# OzFuel tri 1
resized_imgs = []
for wave in ozfuel_hyper[2]:
    resized_imgs.append( ski.resize(tri1[wave, :, :], (ozfuel_dims_tri1[0], ozfuel_dims_tri1[1]), preserve_range=True))
cube = np.stack(resized_imgs, axis=2)
raw_img = np.median(cube, axis=2)
norm_img = (raw_img - np.min(raw_img)) / (np.max(raw_img) - np.min(raw_img))
ozfuel_tri_band3_spectra.append(np.median(norm_img))
# OzFuel tri 2
resized_imgs = []
for wave in ozfuel_hyper[2]:
    resized_imgs.append( ski.resize(tri2[wave, :, :], (ozfuel_dims_tri2[0], ozfuel_dims_tri2[1]), preserve_range=True))
cube = np.stack(resized_imgs, axis=2)
raw_img = np.median(cube, axis=2)
norm_img = (raw_img - np.min(raw_img)) / (np.max(raw_img) - np.min(raw_img))
ozfuel_tri_band3_spectra.append(np.median(norm_img))   
# OzFuel tri 3
resized_imgs = []
for wave in ozfuel_hyper[2]:
    resized_imgs.append( ski.resize(tri3[wave, :, :], (ozfuel_dims_tri3[0], ozfuel_dims_tri3[1]), preserve_range=True))
cube = np.stack(resized_imgs, axis=2)
raw_img = np.median(cube, axis=2)
norm_img = (raw_img - np.min(raw_img)) / (np.max(raw_img) - np.min(raw_img))
ozfuel_tri_band3_spectra.append(np.median(norm_img))
# OzFuel tri 4
resized_imgs = []
for wave in ozfuel_hyper[2]:
    resized_imgs.append( ski.resize(tri4[wave, :, :], (ozfuel_dims_tri4[0], ozfuel_dims_tri4[1]), preserve_range=True))
cube = np.stack(resized_imgs, axis=2)
raw_img = np.median(cube, axis=2)
norm_img = (raw_img - np.min(raw_img)) / (np.max(raw_img) - np.min(raw_img))
ozfuel_tri_band3_spectra.append(np.median(norm_img))



#File 4
file_4 = "C:/Users/Sam/Google Drive/UNI/2022/ENGN4350/Python/FMC/20220202_Reorder_288_2351_384_U16.bin"       
with open(file_4) as f:
    dt = np.dtype('<u2')   
    rawdata = np.fromfile(f, dtype=dt)

length = 2351       #file specifc
height = 384
rawdata = np.reshape(rawdata, (len(wavelengths), length, height))  
master_data_4 = np.rot90(rawdata, 1, (1,2))

# cropping
tri1 = master_data_4[:, 0:384, 1511:1511+245] 
tri2 = master_data_4[:, 0:248, 1773:1773+118] 
tri3 = master_data_4[:, 0:194, 1888:1888+112] 
tri4 = master_data_4[:, 0:109, 2000:2083] 
lengths = [tri1.shape[2], tri2.shape[2], tri3.shape[2], tri4.shape[2]] 
heights = [tri1.shape[1], tri2.shape[1], tri3.shape[1], tri4.shape[1]] 

#calculate dimensions of simulated images according to scaling factor
ozfuel_dims_tri1 = [int(heights[0]/ozfuel_scaling), int(lengths[0]/ozfuel_scaling)]
ozfuel_dims_tri2 = [int(heights[1]/ozfuel_scaling), int(lengths[1]/ozfuel_scaling)]
ozfuel_dims_tri3 = [int(heights[2]/ozfuel_scaling), int(lengths[2]/ozfuel_scaling)]
ozfuel_dims_tri4 = [int(heights[3]/ozfuel_scaling), int(lengths[3]/ozfuel_scaling)]

measured_tri_fmc_file4 = [113.735, 111.975, 107.355, 125.405]      #fuel moisture content of cropped region (float)              

# OzFuel tri 1
resized_imgs = []
for wave in ozfuel_hyper[2]:
    resized_imgs.append( ski.resize(tri1[wave, :, :], (ozfuel_dims_tri1[0], ozfuel_dims_tri1[1]), preserve_range=True))
cube = np.stack(resized_imgs, axis=2)
raw_img = np.median(cube, axis=2)
norm_img = (raw_img - np.min(raw_img)) / (np.max(raw_img) - np.min(raw_img))
ozfuel_tri_band3_spectra.append(np.median(norm_img))
# OzFuel tri 2
resized_imgs = []
for wave in ozfuel_hyper[2]:
    resized_imgs.append( ski.resize(tri2[wave, :, :], (ozfuel_dims_tri2[0], ozfuel_dims_tri2[1]), preserve_range=True))
cube = np.stack(resized_imgs, axis=2)
raw_img = np.median(cube, axis=2)
norm_img = (raw_img - np.min(raw_img)) / (np.max(raw_img) - np.min(raw_img))
ozfuel_tri_band3_spectra.append(np.median(norm_img))   
# OzFuel tri 3
resized_imgs = []
for wave in ozfuel_hyper[2]:
    resized_imgs.append( ski.resize(tri3[wave, :, :], (ozfuel_dims_tri3[0], ozfuel_dims_tri3[1]), preserve_range=True))
cube = np.stack(resized_imgs, axis=2)
raw_img = np.median(cube, axis=2)
norm_img = (raw_img - np.min(raw_img)) / (np.max(raw_img) - np.min(raw_img))
ozfuel_tri_band3_spectra.append(np.median(norm_img))
# OzFuel tri 4
resized_imgs = []
for wave in ozfuel_hyper[2]:
    resized_imgs.append( ski.resize(tri4[wave, :, :], (ozfuel_dims_tri4[0], ozfuel_dims_tri4[1]), preserve_range=True))
cube = np.stack(resized_imgs, axis=2)
raw_img = np.median(cube, axis=2)
norm_img = (raw_img - np.min(raw_img)) / (np.max(raw_img) - np.min(raw_img))
ozfuel_tri_band3_spectra.append(np.median(norm_img))





#File 5
file_5 = "C:/Users/Sam/Google Drive/UNI/2022/ENGN4350/Python/FMC/20211102_Reorder_288_3109_384_U16.bin"       
with open(file_5) as f:
    dt = np.dtype('<u2')   
    rawdata = np.fromfile(f, dtype=dt)

length = 3109       #file specifc
height = 384
rawdata = np.reshape(rawdata, (len(wavelengths), length, height))  
master_data_5 = np.rot90(rawdata, 1, (1,2))


# cropping
tri1 = master_data_5[:, 0:255, 1500:1500+280] 
tri2 = master_data_5[:, 0:212, 1786:1786+122] 
tri3 = master_data_5[:, 0:148, 1906:1906+131] 
tri4 = master_data_5[:, 0:56,  2042:2042+108] 
lengths = [tri1.shape[2], tri2.shape[2], tri3.shape[2], tri4.shape[2]] 
heights = [tri1.shape[1], tri2.shape[1], tri3.shape[1], tri4.shape[1]] 

#calculate dimensions of simulated images according to scaling factor
ozfuel_dims_tri1 = [int(heights[0]/ozfuel_scaling), int(lengths[0]/ozfuel_scaling)]
ozfuel_dims_tri2 = [int(heights[1]/ozfuel_scaling), int(lengths[1]/ozfuel_scaling)]
ozfuel_dims_tri3 = [int(heights[2]/ozfuel_scaling), int(lengths[2]/ozfuel_scaling)]
ozfuel_dims_tri4 = [int(heights[3]/ozfuel_scaling), int(lengths[3]/ozfuel_scaling)]

measured_tri_fmc_file5 = [87.79, 90.13, 86.57, 102.78]      #fuel moisture content of cropped region (float)              

# OzFuel tri 1
resized_imgs = []
for wave in ozfuel_hyper[2]:
    resized_imgs.append( ski.resize(tri1[wave, :, :], (ozfuel_dims_tri1[0], ozfuel_dims_tri1[1]), preserve_range=True))
cube = np.stack(resized_imgs, axis=2)
raw_img = np.median(cube, axis=2)
norm_img = (raw_img - np.min(raw_img)) / (np.max(raw_img) - np.min(raw_img))
ozfuel_tri_band3_spectra.append(np.median(norm_img))
# OzFuel tri 2
resized_imgs = []
for wave in ozfuel_hyper[2]:
    resized_imgs.append( ski.resize(tri2[wave, :, :], (ozfuel_dims_tri2[0], ozfuel_dims_tri2[1]), preserve_range=True))
cube = np.stack(resized_imgs, axis=2)
raw_img = np.median(cube, axis=2)
norm_img = (raw_img - np.min(raw_img)) / (np.max(raw_img) - np.min(raw_img))
ozfuel_tri_band3_spectra.append(np.median(norm_img))   
# OzFuel tri 3
resized_imgs = []
for wave in ozfuel_hyper[2]:
    resized_imgs.append( ski.resize(tri3[wave, :, :], (ozfuel_dims_tri3[0], ozfuel_dims_tri3[1]), preserve_range=True))
cube = np.stack(resized_imgs, axis=2)
raw_img = np.median(cube, axis=2)
norm_img = (raw_img - np.min(raw_img)) / (np.max(raw_img) - np.min(raw_img))
ozfuel_tri_band3_spectra.append(np.median(norm_img))
# OzFuel tri 4
resized_imgs = []
for wave in ozfuel_hyper[2]:
    resized_imgs.append( ski.resize(tri4[wave, :, :], (ozfuel_dims_tri4[0], ozfuel_dims_tri4[1]), preserve_range=True))
cube = np.stack(resized_imgs, axis=2)
raw_img = np.median(cube, axis=2)
norm_img = (raw_img - np.min(raw_img)) / (np.max(raw_img) - np.min(raw_img))
ozfuel_tri_band3_spectra.append(np.median(norm_img))

# In[]: TRI OzFuel band 4

ozfuel_tri_band4_spectra = []


#File 1
file_1 = "C:/Users/Sam/Google Drive/UNI/2022/ENGN4350/Python/FMC/20220228_Reorder_288_2349_384_U16.bin"       
with open(file_1) as f:
    dt = np.dtype('<u2')   
    rawdata = np.fromfile(f, dtype=dt)

length = 2349       #file specifc
height = 384
rawdata = np.reshape(rawdata, (len(wavelengths), length, height))  
master_data_1 = np.rot90(rawdata, 1, (1,2))

# cropping
tri1 = master_data_1[:, 29:29+276, 1619:1619+255] 
tri2 = master_data_1[:, 0:293, 1857:1857+115] 
tri3 = master_data_1[:, 0:191, 1965:1965+115] 
tri4 = master_data_1[:, 0:111, 2069:2069+114] 
lengths = [tri1.shape[2], tri2.shape[2], tri3.shape[2], tri4.shape[2]] 
heights = [tri1.shape[1], tri2.shape[1], tri3.shape[1], tri4.shape[1]] 

#calculate dimensions of simulated images according to scaling factor
ozfuel_dims_tri1 = [int(heights[0]/ozfuel_scaling), int(lengths[0]/ozfuel_scaling)]
ozfuel_dims_tri2 = [int(heights[1]/ozfuel_scaling), int(lengths[1]/ozfuel_scaling)]
ozfuel_dims_tri3 = [int(heights[2]/ozfuel_scaling), int(lengths[2]/ozfuel_scaling)]
ozfuel_dims_tri4 = [int(heights[3]/ozfuel_scaling), int(lengths[3]/ozfuel_scaling)]

measured_tri_fmc_file1 = [98.775, 103.905, 115.135, 104.915]      #fuel moisture content of cropped region (float)              

# OzFuel tri 1
resized_imgs = []
for wave in ozfuel_hyper[3]:
    resized_imgs.append( ski.resize(tri1[wave, :, :], (ozfuel_dims_tri1[0], ozfuel_dims_tri1[1]), preserve_range=True))
cube = np.stack(resized_imgs, axis=2)
raw_img = np.median(cube, axis=2)
norm_img = (raw_img - np.min(raw_img)) / (np.max(raw_img) - np.min(raw_img))
ozfuel_tri_band4_spectra.append(np.median(norm_img))
# OzFuel tri 2
resized_imgs = []
for wave in ozfuel_hyper[3]:
    resized_imgs.append( ski.resize(tri2[wave, :, :], (ozfuel_dims_tri2[0], ozfuel_dims_tri2[1]), preserve_range=True))
cube = np.stack(resized_imgs, axis=2)
raw_img = np.median(cube, axis=2)
norm_img = (raw_img - np.min(raw_img)) / (np.max(raw_img) - np.min(raw_img))
ozfuel_tri_band4_spectra.append(np.median(norm_img))   
# OzFuel tri 3
resized_imgs = []
for wave in ozfuel_hyper[3]:
    resized_imgs.append( ski.resize(tri3[wave, :, :], (ozfuel_dims_tri3[0], ozfuel_dims_tri3[1]), preserve_range=True))
cube = np.stack(resized_imgs, axis=2)
raw_img = np.median(cube, axis=2)
norm_img = (raw_img - np.min(raw_img)) / (np.max(raw_img) - np.min(raw_img))
ozfuel_tri_band4_spectra.append(np.median(norm_img))
# OzFuel tri 4
resized_imgs = []
for wave in ozfuel_hyper[3]:
    resized_imgs.append( ski.resize(tri4[wave, :, :], (ozfuel_dims_tri4[0], ozfuel_dims_tri4[1]), preserve_range=True))
cube = np.stack(resized_imgs, axis=2)
raw_img = np.median(cube, axis=2)
norm_img = (raw_img - np.min(raw_img)) / (np.max(raw_img) - np.min(raw_img))
ozfuel_tri_band4_spectra.append(np.median(norm_img))
  


  
#File 2
file_2 = "C:/Users/Sam/Google Drive/UNI/2022/ENGN4350/Python/FMC/20211206_Reorder_288_2108_384_U16.bin"       
with open(file_2) as f:
    dt = np.dtype('<u2')   
    rawdata = np.fromfile(f, dtype=dt)

length = 2108       #file specifc
height = 384
rawdata = np.reshape(rawdata, (len(wavelengths), length, height))  
master_data_2 = np.rot90(rawdata, 1, (1,2))

# cropping
tri1 = master_data_2[:, 3:3+213, 1310:1310+298] 
tri2 = master_data_2[:, 0:188, 1614:1614+84] 
tri3 = master_data_2[:, 0:169, 1696:1696+93]
tri4 = master_data_2[:, 0:169, 1784:1784+93] 
lengths = [tri1.shape[2], tri2.shape[2], tri3.shape[2], tri4.shape[2]] 
heights = [tri1.shape[1], tri2.shape[1], tri3.shape[1], tri4.shape[1]] 

#calculate dimensions of simulated images according to scaling factor
ozfuel_dims_tri1 = [int(heights[0]/ozfuel_scaling), int(lengths[0]/ozfuel_scaling)]
ozfuel_dims_tri2 = [int(heights[1]/ozfuel_scaling), int(lengths[1]/ozfuel_scaling)]
ozfuel_dims_tri3 = [int(heights[2]/ozfuel_scaling), int(lengths[2]/ozfuel_scaling)]
ozfuel_dims_tri4 = [int(heights[3]/ozfuel_scaling), int(lengths[3]/ozfuel_scaling)]

measured_tri_fmc_file2 = [114.135, 122.94, 148.54, 150.73]      #fuel moisture content of cropped region (float)              

# OzFuel tri 1
resized_imgs = []
for wave in ozfuel_hyper[3]:
    resized_imgs.append( ski.resize(tri1[wave, :, :], (ozfuel_dims_tri1[0], ozfuel_dims_tri1[1]), preserve_range=True))
cube = np.stack(resized_imgs, axis=2)
raw_img = np.median(cube, axis=2)
norm_img = (raw_img - np.min(raw_img)) / (np.max(raw_img) - np.min(raw_img))
ozfuel_tri_band4_spectra.append(np.median(norm_img))
# OzFuel tri 2
resized_imgs = []
for wave in ozfuel_hyper[3]:
    resized_imgs.append( ski.resize(tri2[wave, :, :], (ozfuel_dims_tri2[0], ozfuel_dims_tri2[1]), preserve_range=True))
cube = np.stack(resized_imgs, axis=2)
raw_img = np.median(cube, axis=2)
norm_img = (raw_img - np.min(raw_img)) / (np.max(raw_img) - np.min(raw_img))
ozfuel_tri_band4_spectra.append(np.median(norm_img))   
# OzFuel tri 3
resized_imgs = []
for wave in ozfuel_hyper[3]:
    resized_imgs.append( ski.resize(tri3[wave, :, :], (ozfuel_dims_tri3[0], ozfuel_dims_tri3[1]), preserve_range=True))
cube = np.stack(resized_imgs, axis=2)
raw_img = np.median(cube, axis=2)
norm_img = (raw_img - np.min(raw_img)) / (np.max(raw_img) - np.min(raw_img))
ozfuel_tri_band4_spectra.append(np.median(norm_img))
# OzFuel tri 4
resized_imgs = []
for wave in ozfuel_hyper[3]:
    resized_imgs.append( ski.resize(tri4[wave, :, :], (ozfuel_dims_tri4[0], ozfuel_dims_tri4[1]), preserve_range=True))
cube = np.stack(resized_imgs, axis=2)
raw_img = np.median(cube, axis=2)
norm_img = (raw_img - np.min(raw_img)) / (np.max(raw_img) - np.min(raw_img))
ozfuel_tri_band4_spectra.append(np.median(norm_img))
     

#File 3
file_3 = "C:/Users/Sam/Google Drive/UNI/2022/ENGN4350/Python/FMC/20220113_Reorder_288_2431_384_U16.bin"       
with open(file_3) as f:
    dt = np.dtype('<u2')   
    rawdata = np.fromfile(f, dtype=dt)

length = 2431       #file specifc
height = 384
rawdata = np.reshape(rawdata, (len(wavelengths), length, height))  
master_data_3 = np.rot90(rawdata, 1, (1,2))

# cropping
tri1 = master_data_3[:, 28:28+352, 1466:1466+291] 
tri2 = master_data_3[:, 8:271, 1727:1727+291] 
tri3 = master_data_3[:, 0:193, 1853:1853+106] 
tri4 = master_data_3[:, 0:139, 1958:1958+106] 
lengths = [tri1.shape[2], tri2.shape[2], tri3.shape[2], tri4.shape[2]] 
heights = [tri1.shape[1], tri2.shape[1], tri3.shape[1], tri4.shape[1]] 

#calculate dimensions of simulated images according to scaling factor
ozfuel_dims_tri1 = [int(heights[0]/ozfuel_scaling), int(lengths[0]/ozfuel_scaling)]
ozfuel_dims_tri2 = [int(heights[1]/ozfuel_scaling), int(lengths[1]/ozfuel_scaling)]
ozfuel_dims_tri3 = [int(heights[2]/ozfuel_scaling), int(lengths[2]/ozfuel_scaling)]
ozfuel_dims_tri4 = [int(heights[3]/ozfuel_scaling), int(lengths[3]/ozfuel_scaling)]

measured_tri_fmc_file3 = [98.975, 131.19, 151.045, 140.86]      #fuel moisture content of cropped region (float)              

# OzFuel tri 1
resized_imgs = []
for wave in ozfuel_hyper[3]:
    resized_imgs.append( ski.resize(tri1[wave, :, :], (ozfuel_dims_tri1[0], ozfuel_dims_tri1[1]), preserve_range=True))
cube = np.stack(resized_imgs, axis=2)
raw_img = np.median(cube, axis=2)
norm_img = (raw_img - np.min(raw_img)) / (np.max(raw_img) - np.min(raw_img))
ozfuel_tri_band4_spectra.append(np.median(norm_img))
# OzFuel tri 2
resized_imgs = []
for wave in ozfuel_hyper[3]:
    resized_imgs.append( ski.resize(tri2[wave, :, :], (ozfuel_dims_tri2[0], ozfuel_dims_tri2[1]), preserve_range=True))
cube = np.stack(resized_imgs, axis=2)
raw_img = np.median(cube, axis=2)
norm_img = (raw_img - np.min(raw_img)) / (np.max(raw_img) - np.min(raw_img))
ozfuel_tri_band4_spectra.append(np.median(norm_img))   
# OzFuel tri 3
resized_imgs = []
for wave in ozfuel_hyper[3]:
    resized_imgs.append( ski.resize(tri3[wave, :, :], (ozfuel_dims_tri3[0], ozfuel_dims_tri3[1]), preserve_range=True))
cube = np.stack(resized_imgs, axis=2)
raw_img = np.median(cube, axis=2)
norm_img = (raw_img - np.min(raw_img)) / (np.max(raw_img) - np.min(raw_img))
ozfuel_tri_band4_spectra.append(np.median(norm_img))
# OzFuel tri 4
resized_imgs = []
for wave in ozfuel_hyper[3]:
    resized_imgs.append( ski.resize(tri4[wave, :, :], (ozfuel_dims_tri4[0], ozfuel_dims_tri4[1]), preserve_range=True))
cube = np.stack(resized_imgs, axis=2)
raw_img = np.median(cube, axis=2)
norm_img = (raw_img - np.min(raw_img)) / (np.max(raw_img) - np.min(raw_img))
ozfuel_tri_band4_spectra.append(np.median(norm_img))



#File 4
file_4 = "C:/Users/Sam/Google Drive/UNI/2022/ENGN4350/Python/FMC/20220202_Reorder_288_2351_384_U16.bin"       
with open(file_4) as f:
    dt = np.dtype('<u2')   
    rawdata = np.fromfile(f, dtype=dt)

length = 2351       #file specifc
height = 384
rawdata = np.reshape(rawdata, (len(wavelengths), length, height))  
master_data_4 = np.rot90(rawdata, 1, (1,2))

# cropping
tri1 = master_data_4[:, 0:384, 1511:1511+245] 
tri2 = master_data_4[:, 0:248, 1773:1773+118] 
tri3 = master_data_4[:, 0:194, 1888:1888+112] 
tri4 = master_data_4[:, 0:109, 2000:2083] 
lengths = [tri1.shape[2], tri2.shape[2], tri3.shape[2], tri4.shape[2]] 
heights = [tri1.shape[1], tri2.shape[1], tri3.shape[1], tri4.shape[1]] 

#calculate dimensions of simulated images according to scaling factor
ozfuel_dims_tri1 = [int(heights[0]/ozfuel_scaling), int(lengths[0]/ozfuel_scaling)]
ozfuel_dims_tri2 = [int(heights[1]/ozfuel_scaling), int(lengths[1]/ozfuel_scaling)]
ozfuel_dims_tri3 = [int(heights[2]/ozfuel_scaling), int(lengths[2]/ozfuel_scaling)]
ozfuel_dims_tri4 = [int(heights[3]/ozfuel_scaling), int(lengths[3]/ozfuel_scaling)]

measured_tri_fmc_file4 = [113.735, 111.975, 107.355, 125.405]      #fuel moisture content of cropped region (float)              

# OzFuel tri 1
resized_imgs = []
for wave in ozfuel_hyper[3]:
    resized_imgs.append( ski.resize(tri1[wave, :, :], (ozfuel_dims_tri1[0], ozfuel_dims_tri1[1]), preserve_range=True))
cube = np.stack(resized_imgs, axis=2)
raw_img = np.median(cube, axis=2)
norm_img = (raw_img - np.min(raw_img)) / (np.max(raw_img) - np.min(raw_img))
ozfuel_tri_band4_spectra.append(np.median(norm_img))
# OzFuel tri 2
resized_imgs = []
for wave in ozfuel_hyper[3]:
    resized_imgs.append( ski.resize(tri2[wave, :, :], (ozfuel_dims_tri2[0], ozfuel_dims_tri2[1]), preserve_range=True))
cube = np.stack(resized_imgs, axis=2)
raw_img = np.median(cube, axis=2)
norm_img = (raw_img - np.min(raw_img)) / (np.max(raw_img) - np.min(raw_img))
ozfuel_tri_band4_spectra.append(np.median(norm_img))   
# OzFuel tri 3
resized_imgs = []
for wave in ozfuel_hyper[3]:
    resized_imgs.append( ski.resize(tri3[wave, :, :], (ozfuel_dims_tri3[0], ozfuel_dims_tri3[1]), preserve_range=True))
cube = np.stack(resized_imgs, axis=2)
raw_img = np.median(cube, axis=2)
norm_img = (raw_img - np.min(raw_img)) / (np.max(raw_img) - np.min(raw_img))
ozfuel_tri_band4_spectra.append(np.median(norm_img))
# OzFuel tri 4
resized_imgs = []
for wave in ozfuel_hyper[3]:
    resized_imgs.append( ski.resize(tri4[wave, :, :], (ozfuel_dims_tri4[0], ozfuel_dims_tri4[1]), preserve_range=True))
cube = np.stack(resized_imgs, axis=2)
raw_img = np.median(cube, axis=2)
norm_img = (raw_img - np.min(raw_img)) / (np.max(raw_img) - np.min(raw_img))
ozfuel_tri_band4_spectra.append(np.median(norm_img))





#File 5
file_5 = "C:/Users/Sam/Google Drive/UNI/2022/ENGN4350/Python/FMC/20211102_Reorder_288_3109_384_U16.bin"       
with open(file_5) as f:
    dt = np.dtype('<u2')   
    rawdata = np.fromfile(f, dtype=dt)

length = 3109       #file specifc
height = 384
rawdata = np.reshape(rawdata, (len(wavelengths), length, height))  
master_data_5 = np.rot90(rawdata, 1, (1,2))


# cropping
tri1 = master_data_5[:, 0:255, 1500:1500+280] 
tri2 = master_data_5[:, 0:212, 1786:1786+122] 
tri3 = master_data_5[:, 0:148, 1906:1906+131] 
tri4 = master_data_5[:, 0:56,  2042:2042+108] 
lengths = [tri1.shape[2], tri2.shape[2], tri3.shape[2], tri4.shape[2]] 
heights = [tri1.shape[1], tri2.shape[1], tri3.shape[1], tri4.shape[1]] 

#calculate dimensions of simulated images according to scaling factor
ozfuel_dims_tri1 = [int(heights[0]/ozfuel_scaling), int(lengths[0]/ozfuel_scaling)]
ozfuel_dims_tri2 = [int(heights[1]/ozfuel_scaling), int(lengths[1]/ozfuel_scaling)]
ozfuel_dims_tri3 = [int(heights[2]/ozfuel_scaling), int(lengths[2]/ozfuel_scaling)]
ozfuel_dims_tri4 = [int(heights[3]/ozfuel_scaling), int(lengths[3]/ozfuel_scaling)]

measured_tri_fmc_file5 = [87.79, 90.13, 86.57, 102.78]      #fuel moisture content of cropped region (float)              

# OzFuel tri 1
resized_imgs = []
for wave in ozfuel_hyper[3]:
    resized_imgs.append( ski.resize(tri1[wave, :, :], (ozfuel_dims_tri1[0], ozfuel_dims_tri1[1]), preserve_range=True))
cube = np.stack(resized_imgs, axis=2)
raw_img = np.median(cube, axis=2)
norm_img = (raw_img - np.min(raw_img)) / (np.max(raw_img) - np.min(raw_img))
ozfuel_tri_band4_spectra.append(np.median(norm_img))
# OzFuel tri 2
resized_imgs = []
for wave in ozfuel_hyper[3]:
    resized_imgs.append( ski.resize(tri2[wave, :, :], (ozfuel_dims_tri2[0], ozfuel_dims_tri2[1]), preserve_range=True))
cube = np.stack(resized_imgs, axis=2)
raw_img = np.median(cube, axis=2)
norm_img = (raw_img - np.min(raw_img)) / (np.max(raw_img) - np.min(raw_img))
ozfuel_tri_band4_spectra.append(np.median(norm_img))   
# OzFuel tri 3
resized_imgs = []
for wave in ozfuel_hyper[3]:
    resized_imgs.append( ski.resize(tri3[wave, :, :], (ozfuel_dims_tri3[0], ozfuel_dims_tri3[1]), preserve_range=True))
cube = np.stack(resized_imgs, axis=2)
raw_img = np.median(cube, axis=2)
norm_img = (raw_img - np.min(raw_img)) / (np.max(raw_img) - np.min(raw_img))
ozfuel_tri_band4_spectra.append(np.median(norm_img))
# OzFuel tri 4
resized_imgs = []
for wave in ozfuel_hyper[3]:
    resized_imgs.append( ski.resize(tri4[wave, :, :], (ozfuel_dims_tri4[0], ozfuel_dims_tri4[1]), preserve_range=True))
cube = np.stack(resized_imgs, axis=2)
raw_img = np.median(cube, axis=2)
norm_img = (raw_img - np.min(raw_img)) / (np.max(raw_img) - np.min(raw_img))
ozfuel_tri_band4_spectra.append(np.median(norm_img))

# In[]: Plot

measured_fmc = measured_fmc_file1 + measured_fmc_file2 + measured_fmc_file3 + measured_fmc_file4 + measured_fmc_file5 + measured_tri_fmc_file1 + measured_tri_fmc_file2 + measured_tri_fmc_file3 + measured_tri_fmc_file4 + measured_tri_fmc_file5
all_points = ozfuel_band1_spectra + ozfuel_band2_spectra + ozfuel_band3_spectra + ozfuel_band4_spectra + ozfuel_tri_band1_spectra + ozfuel_tri_band2_spectra + ozfuel_tri_band3_spectra + ozfuel_tri_band4_spectra
num_points = len(all_points)

fig = plt.figure(dpi=600, figsize=(12,8))
plt.scatter(measured_fmc*4, all_points) 

slope, intercept, r_value, p_value, std_err = stats.linregress(measured_fmc*4, all_points)
xvals = range(int(min(measured_fmc)), int(max(measured_fmc)))

plt.plot(xvals, slope*xvals + intercept, 'r')
plt.title('Simulated spectra from imagery vs measured fuel moisture content')
plt.xlabel('Measured fuel moisture %')
plt.ylabel('Median pixel value')

plt.legend([f'OzFuel spectra: n = {num_points}', f'Linear Regression: y = {slope:.2}x + {intercept:.2f}, R\N{SUPERSCRIPT TWO} = {r_value**2:.2f}'])
# save figure for external processing
plt.savefig('fmc/allpoints_fmc.png', bbox_inches='tight')

# In[]:
# data_file.close()
end = time.time()
print(end-start)