import pandas as pd
import numpy as np 
data+pd.DataFrame([[9,4,8,9],[8,10,7,6],[7,6,8,5]],columns=['Maths','English','Science','History'])
print(data.agg(['sum','min','max']))
m=lambda x:x+10
print(m(5))
print(data)
list(map(lambda x:x*x,data['Maths']))
a=list(filter(lambda x:x%2,data['Maths']))
from functools import reduce
b=reduce(lambda x,y:x+y,data['Science'])
print(b)
