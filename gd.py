import gdal
dataset = gdal.Open(r'E:\PROGRAM\S1+L8\Data\test\images\5_1_C11\1_325.tif')
cols=dataset.RasterXSize#图像长度
rows=(dataset.RasterYSize)#图像宽度
dataset=dataset.ReadAsArray(0,0,cols,rows)
print(dataset)