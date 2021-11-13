import pandas as pd

ls = [1,2,3,4,5]
ls = pd.DataFrame(ls)
features = ['GDP_2018', 'gini', 'Infant Mortality Rate', 'Population Density', 'Urban Population', 'Median Age']


def partition(lst, n):
    division = len(lst) / n
    return [lst[round(division * i):round(division * (i + 1))] for i in range(n)]


print(partition([1,2,3,4,5],5))
