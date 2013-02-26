import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()

plt.plot( np.arange(0,10),[9,4,5,2,3,5,7,12,2,3],'.-',label='sample1' )
plt.plot ( np.arange(0,10),[12,5,33,2,4,5,3,3,22,10],'o-',label='sample2' )
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.title('my sample graphs')
plt.legend(('sample1','sample2'))
plt.savefig("sampleg.png",dpi=(640/8))
