import numpy as np

baseline_last = [84.53, 84.60, 84.56, 84.64, 84.38]
baseline_best = [84.86, 84.63, 84.83, 84.73,84.54]

op_last = [84.56, 84.73, 85.20, 85.10, 84.58]
op_best = [84.75, 84.73, 85.22, 85.13, 84.66]

comp_last = [84.89, 84.50, 84.56, 85.15, 84.51]
comp_best = [85.04, 84.83, 84.67, 85.17, 84.74]

baseline_best = np.array(baseline_best)
baseline_last = np.array(baseline_last)
op_best = np.array(op_best)
op_last = np.array(op_last)
comp_best = np.array(comp_best)
comp_last = np.array(comp_last)

print('baseline_best: ', baseline_best.mean(), baseline_best.std())
print('baseline_last: ', baseline_last.mean(), baseline_last.std())
print('op_best: ', op_best.mean(), op_best.std())
print('op_last: ', op_last.mean(), op_last.std())
print('comp_best: ', comp_best.mean(), comp_best.std())
print('comp_last: ', comp_last.mean(), comp_last.std())



### baseline_best:  84.718 0.12023310692151154
### baseline_last:  84.542 0.08908422980528145
### op_best:  84.898 0.22990432792794374
### op_last:  84.834 0.266503283281838
### comp_best:  84.89000000000001 0.18729655629509254
### comp_last:  84.72200000000001 0.2576354012941553