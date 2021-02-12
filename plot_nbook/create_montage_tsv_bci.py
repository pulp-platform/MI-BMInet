DATA = """

Fz	46	90
FC3	-50	-28
FC1	-32	-45
FCz	23	90
FC2	32	45
FC4	50	28
C5	-69	0
C3	-46	0
C1	-23	0
Cz	0	0
C2	23	0
C4	46	0
C6	69	0
CP3	-50	28
CP1	-32	45
CPz	23	-90
CP2	32	-45
CP4	50	-28
P1	-50	68
Pz	46	-90
P2	50	-68
POz	69	-90

Nz	 115	 90
LPA	-115	  0
RPA	 115	  0
"""
# Iz	115	-90

#P9    -115    36
#P10    115 -36
#MyNas   115     90
#MyCheek     -112    -72

fname = 'physionet_mmmi22chs.tsv'
with open(fname, 'w') as fout:
    fout.write(DATA)


# import scipy.io as sio

# path = "/usr/scratch/bismantova/xiaywang/Projects/BCI/datasets/BCI-CompIV-2a/QuantLab/BCI-CompIV-2a/data/"
# subject = 1

# a = sio.loadmat(path+'A0'+str(subject)+'T.mat')

# #print(a)
# a_data = a['data']
# for ii in range(0,a_data.size):
#     a_data1 = a_data[0,ii]
#     a_data2=[a_data1[0,0]]
#     a_data3=a_data2[0]
#     a_X 		= a_data3[0]
#     a_trial 	= a_data3[1]
#     a_y 		= a_data3[2]
#     print(a_y)
#     a_fs 		= a_data3[3]
#     a_classes 	= a_data3[4]
#     print(a_classes)
#     a_artifacts = a_data3[5]
#     a_gender 	= a_data3[6]
#     a_age 		= a_data3[7]
