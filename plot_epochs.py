import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np

without_momentum = [(0.5, 99491), (0.45, 39374), (0.4, 91128), (0.3500000000000000, 40964),
                    (0.3000000000000000, 32120), (0.2500000000000000, 78036), (0.2000000000000000, 254201),
                    (0.1500000000000000, 100496), (0.1000000000000000, 896103), (0.05, 1643911)]
with_momentum = [(0.5, 10372), (0.45, 8965), (0.4, 10090), (0.3500000000000000, 135816), (0.3000000000000000, 21606),
                 (0.2500000000000000, 18024), (0.2000000000000000, 17940), (0.1500000000000000, 233119),
                 (0.1000000000000000, 46799), (0.050000000000000, 418600)]

# set width of bar
barWidth = 0.25

matplotlib.use('TkAgg')
value_with_out_m=[]
value_with_m=[]
for i in range(len(with_momentum)):
    value_with_out_m.append(without_momentum[i][1])
    value_with_m.append(with_momentum[i][1])
# print(value_with_out)
range_ = np.arange(len(value_with_out_m))
range_w_m = [x+barWidth for x in range_]
print(range_)
print(range_w_m)
fig,ax=plt.subplots()
plt.bar(range_,value_with_out_m, color='#000000',width=barWidth, edgecolor='black', label='without momentum')
plt.bar(range_w_m,value_with_m, color='#A9A9A9',width=barWidth, edgecolor='white', label='with momentum')

plt.xlabel('learning rates', fontweight='bold')
plt.ylabel('epochs')
print([r + barWidth*2 for r in np.arange(len(range_))])
strs = []
for i in range(len(with_momentum)):
    strs.append("{:.2f}".format(with_momentum[i][0]))
# ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
plt.xticks([r + barWidth for r in np.arange(len(range_))],strs)

# Create legend & Show graphic
plt.legend()
plt.show()








# # set height of bar
# bars1 = [12, 30, 1, 8, 22]
# bars2 = [28, 6, 16, 5, 10]
# bars3 = [29, 3, 24, 25, 17]
#
# # Set position of bar on X axis
# r1 = np.arange(len(bars1))
# r2 = [x + barWidth for x in r1]
# r3 = [x + barWidth for x in r2]
#
# # Make the plot
# plt.bar(r1, bars1, color='#000000', width=barWidth, edgecolor='white', label='var1')
# plt.bar(r2, bars2, color='#A9A9A9', width=barWidth, edgecolor='white', label='var2')
# plt.bar(r3, bars3, color='#808080', width=barWidth, edgecolor='white', label='var3')
#
# # Add xticks on the middle of the group bars
# plt.xlabel('group', fontweight='bold')
# plt.xticks([r + barWidth for r in range(len(bars1))], ['A', 'B', 'C', 'D', 'E'])
#
# # Create legend & Show graphic
# plt.legend()
# plt.show()
