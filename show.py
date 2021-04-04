import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

name = 'Level4'

lev1_lab0 = pd.read_csv(r'data_set_1000train\\'+name+'_test.csv').values
df0 = pd.DataFrame(lev1_lab0)

label = lev1_lab0[:,0]
data = lev1_lab0[:,1:]
# orginal data
tsne = TSNE(n_components=3, init='pca', random_state=1)
result = tsne.fit_transform(data)
x_min, x_max = np.min(result), np.max(result)# 这一步似乎让结果都变为0-1的数字
result = (result - x_min)/(x_max-x_min)
fig = plt.figure()
ax = plt.subplot(111)
ax = Axes3D(fig)


ax.scatter(result[:100,0], result[:100,1],result[:100,2],c = '#00CED1',s=8,marker='o',label = "normal")
ax.scatter(result[100:200,0], result[100:200,1],result[100:200,2],c = 'Chocolate',s=8,marker='o',label = "CF")
ax.scatter(result[200:300,0], result[200:300,1],result[200:300,2],c = '#DC143C',s=8,marker='o',label = "EO")
ax.scatter(result[300:400,0], result[300:400,1],result[300:400,2],c = '#A9A9A9',s=8,marker='o',label = "NCR")
ax.scatter(result[400:500,0], result[400:500,1],result[400:500,2],c = '#556B2F',s=8,marker='o',label = "RCW")
ax.scatter(result[500:600,0], result[500:600,1],result[500:600,2],c = '#9932CC',s=8,marker='o',label = "REW")
ax.scatter(result[600:700,0], result[600:700,1],result[600:700,2],c = 'Gold',s=8,marker='o',label = "RL")
ax.scatter(result[700:800,0], result[700:800,1],result[700:800,2],c = 'Indigo',s=8,marker='o',label = "RO")
ax.legend(loc=2)


plt.title('Classification data visualization of '+ name)
plt.show(fig)