import pandas as pd
import numpy as np

ls = [1,2,3,4,5]
ls = pd.DataFrame(ls)
features = ['GDP_2018', 'gini', 'Infant Mortality Rate', 'Population Density', 'Urban Population', 'Median Age']

x = [[1, 2, 3], [4, 5, 6]]
print(np.sum(x, axis=0))

print(np.log(5))