from annoy import AnnoyIndex
import random
import time
import numpy as np

f = 40
t = AnnoyIndex(f, 'angular')  # Length of item vector that will be indexed
for i in range(100):
    v = np.random.randn(f)
    t.add_item(i, v)
startTime = time.time()
t.build(1000) # 10 trees
t.save('test.ann')

# ...
u = AnnoyIndex(f, 'angular')
u.load('test.ann') # super fast, will just mmap the file
a = u.get_nns_by_item(0, 10)
a.remove(0)
print(a)
 # will find the 1000 nearest neighbors
endTime = time.time()
print(endTime-startTime)