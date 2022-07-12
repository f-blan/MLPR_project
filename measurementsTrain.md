# MVG

## Raw

| model        | 𝝅 = 0.5 | 𝝅 = 0.9 | 𝝅 = 0.1 |
| ------------ | ------- | -------  | ------- |
| MVG_FC       | 0.049   | 0.1206   | 0.1266  |
|MVG_Tied      | 0.047   | 0.1263   | 0.1210  |
|MVG_Diag      | 0.5654  | 0.8616   | 0.8170  |

## Gaussianization

| model        | 𝝅 = 0.5 | 𝝅 = 0.9 | 𝝅 = 0.1 |
| ------------ | ------- | ------- | ------- |
| MVG_FC       | 0.0613  | 0.1773  | 0.1856  | 
|MVG_Tied      | 0.0593  | 0.1646  | 0.1790  |
|MVG_Diag      | 0.5409  | 0.8356  | 0.8029  |

## Z-norm

| model        | 𝝅 = 0.5 | 𝝅 = 0.9 | 𝝅 = 0.1 |
| ------------ | ------- | ------- | ------- |
| MVG_FC       | 0.0490  | 0.1206  | 0.1266  |
|MVG_Tied      | 0.0470  | 0.1263  | 0.1210  |
|MVG_Diag      | 0.5653  | 0.8616  | 0.8170  |

## PCA 10

| model        | 𝝅 = 0.5 | 𝝅 = 0.9 | 𝝅 = 0.1 |
| ------------ | ------- | ------- | ------- |
| MVG_FC       | 0.0473  | 0.1183  | 0.1393  | 
|MVG_Tied      | 0.0470  | 0.1243  | 0.1303  |
|MVG_Diag      | 0.0670  | 0.1646  | 0.1663  |

## PCA 8
| model        | 𝝅 = 0.5 | 𝝅 = 0.9 | 𝝅 = 0.1 |
| ------------ | ------- | ------- | ------- |
| MVG_FC       | 0.0446  | 0.1226  | 0.1403  | 
|MVG_Tied      | 0.0446  | 0.1256  | 0.1320  |
|MVG_Diag      | 0.0673  | 0.1623  | 0.1703  |

## Z + PCA 8

| model        | 𝝅 = 0.5 | 𝝅 = 0.9 | 𝝅 = 0.1 |
| ------------ | ------- | ------- | ------- |
| MVG_FC       | 0.1770  | 0.4210  | 0.4493  |
|MVG_Tied      | 0.1740  | 0.4090  | 0.4489  |
|MVG_Diag      | 0.01843 | 0.4343  | 0.4553  | 

## Gauss + PCA 8

| model        | 𝝅 = 0.5 | 𝝅 = 0.9 | 𝝅 = 0.1 |
| ------------ | ------- | ------- | ------- |
| MVG_FC       | 0.1663  | 0.4233  | 0.4380  |
|MVG_Tied      | 0.1643  | 0.4076  | 0.4323  |
|MVG_Diag      | 0.1720  | 0.4240  | 0.4486  | 


# Logistic regression

## raw

| model                 | 𝝅 = 0.5 | 𝝅 = 0.9 | 𝝅 = 0.1 |
| ------------          | ------- | ------- | ------- |
| linear(l = 0.001)     | 0.0466  | 0.1233  | 0.1253  | 
| qudratic(l = +- 10e4) | 0.0569  | 0.1413  | 0.1593  | 

* for quadratic min for pi = 0.9 was 0.1230 at l = 0.001 and for pi = 0.1 it was 0.1256 at l= 0.001

## pca 8

| model                 | 𝝅 = 0.5 | 𝝅 = 0.9 | 𝝅 = 0.1 |
| ------------          | ------- | ------- | ------- |
| linear(l = 0.001)     | 0.0453  | 0.1283  | 0.1310  | 
| qudratic(l = +- 10e4) | 0.0463  | 0.1293  | 0.1333  |


## Gaussianization
| model                 | 𝝅 = 0.5 | 𝝅 = 0.9 | 𝝅 = 0.1 |
| ------------          | ------- | ------- | ------- |
| linear(l = 0.001)     | 0.0580  | 0.1686  | 0.1730  | 
| qudratic(l = +- 10-2) | 0.0553  | 0.1536  | 0.1560  |

# SVM

## Raw

| model                 | 𝝅 = 0.5 | 𝝅 = 0.9 | 𝝅 = 0.1 |
| ------------          | ------- | ------- | ------- |
| linear(C = 1)         | 0.0556  | 0.1516  | 0.1486  |
| qudratic(C =  10e-3)  | 0.5376  | -       | -       |
| RBF (gamma= 0.01, C=20)| 0.0520  | 0.1783  | 0.1583  |

## pca 8

| model                 | 𝝅 = 0.5 | 𝝅 = 0.9 | 𝝅 = 0.1 |
| ------------          | ------- | ------- | ------- |
| linear(C = 1)         | 0.0683  | 0.1886  | 0.1786  |        
| qudratic(C =  10e-3)  | 0.9786  | -       | -       |
| RBF (gamma= 0.01,C=20)| 0.0516  | 0.1783  | 0.1583  |

## Gaussianize


| model                 | 𝝅 = 0.5 | 𝝅 = 0.9 | 𝝅 = 0.1 |
| ------------          | ------- | ------- | ------- |
| linear(C = 10e-3)     | 0.0550  | 0.1586  | 0.1566  |       
| qudratic(C =  20)     | 0.0546  | 0.1503  | 0.1676  | 
| RBF (gamma=0.1,C=10e3)| 0.0673  | 0.3996  | 0.3896  |


# GMM

## Raw

| model                 | 𝝅 = 0.5 | 𝝅 = 0.9 | 𝝅 = 0.1 |
| ------------          | ------- | ------- | ------- |
| FC (g =  8)           | 0.0333  | 0.0873  | 0.0989  |   
| T (g= 16)             | 0.0320  | 0.0916  | 0.1063  | 
| N (g= 32)             | 0.0793  | 0.2116  | 0.0193  |
| NT (g= 16)            | 0.0810  | 0.2180  | 0.1996  |

## PCA 8

| model                 | 𝝅 = 0.5 | 𝝅 = 0.9 | 𝝅 = 0.1 |
| ------------          | ------- | ------- | ------- |
| FC (g =  8)           | 0.0376  | 0.1066  | 0.1053  |   
| T (g=  4)             | 0.0670  | 0.1590  | 0.1353  | 
| N (g=  4)             | 0.0670  | 0.1590  | 0.1353  |
| NT (g=  8)            | 0.0530  | 0.1483  | 0.1460  |

## Gauss

| model                 | 𝝅 = 0.5 | 𝝅 = 0.9 | 𝝅 = 0.1 |
| ------------          | ------- | ------- | ------- |
| FC (g =  8)           | 0.0456  | 0.1246  | 0.1283  |   
| T (g=  8)             | 0.0579  | 0.1406  | 0.1669  | 
| N (g= 16)             | 0.1020  | 0.2740  | 0.2593  |
| NT (g= 8)             | 0.1473  | 0.3433  | 0.426  |

# Calibration

## no calibration


| model                 |      𝝅 = 0.5     |      𝝅 = 0.9     |      𝝅 = 0.1     |
| ----------------      | ---------------- | ---------------- | ---------------- |
| primary               | 0.0320 - 0.0356  | 0.0916 - 0.1520  | 0.1063 - 0.1530  |   
| secondary             | 0.0333 - 0.0340  | 0.0873 - 0.0890  | 0.0989 - 0.1043  |

## best th estimation

### Primary

|  𝝅                    |      minDCF      |      actDCF      |      optDCF      |   th   |   
| ----------------      | ---------------- | ---------------- | ---------------- | ------ |
| 0.5                   |      0.0265      |      0.0300      |      0.0297      | 4.4267 |
| 0.9                   |      0.0837      |      0.1475      |      0.0920      | -35.98 |
| 0.1                   |      0.1028      |      0.1267      |      0.1250      | 37.614 |       

### Secondary

|  𝝅                    |      minDCF      |      actDCF      |      optDCF      |   th   |   
| ----------------      | ---------------- | ---------------- | ---------------- | ------ |
| 0.5                   |      0.0265      |      0.0297      |      0.0297      | -0.366 |
| 0.9                   |      0.0812      |      0.1027      |      0.1009      | -2.16 |
| 0.1                   |      0.0585      |      0.0870      |      0.0918      | 2.57 |

## LR calibration primary

| calibration mode      |      minDCF (𝝅=0.5)      |      minDCF (𝝅=0.9)      |      minDCF (𝝅=0.1)      | 
| ----------------      | ------------------------ | ------------------------ | ------------------------ |
| all modes             |          0.0265          |           0.0837         |           0.1028         |
| fusion model          |          0.0283          |           0.0899         |           0.0823         |

| calibration mode      |      actDCF (𝝅=0.5)      |      actDCF (𝝅=0.9)      |      actDCF (𝝅=0.1)      |
| ----------------      | ------------------------ | ------------------------ | ------------------------ |
| uncalibrated          |          0.0300          |           0.1475         |           0.1267         |
| logReg (pi=0.5)       |          0.0300          |           0.0955         |           0.1266         |
| logReg (pi=0.9)       |          0.0283          |           0.0955         |           0.1298         |
| logReg (pi=0.1)       |          0.0336          |           0.0883         |           0.1298         |
| opt_threshold         |          0.0297          |           0.0920         |           0.1250         |
        

## Fusion

| model                 | min 0.5 | act 0.5  | min 0.9 | act 0.9 | min 0.1 | act 0.1 |
| ----------------      | ------- | -------- | ------- | ------- | ------- | ------- |
| primary+secondary cal | 0.0253  |  0.0281  | 0.0688  | 0.0955  | 0.0807  | 0.0886  |
| primary uncal         | 0.0265  |  0.0300  | 0.0836  | 0.1475  | 0.1028  | 0.1267  |
| secondary undal       | 0.0265  |  0.0297  | 0.8120  | 0.1027  | 0.0585  | 0.0870  | 


