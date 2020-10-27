# arrowmatrix
A matrix file format

Inspired in part by
https://gist.github.com/thomwolf/13ca2b2b172b2d17ac66685aa2eeba62

This repo contains a "straw man" proposal for the next generation openmatrix format.
You can review a speed demo in the notebooks folder.


## Open Matrix (via ActivitySim)


```python
with resource_usage:
    asim_skims = skim_dict(settings) # loads all skim data into memory
```

    45.8 s: Net 7.11 GB, Total 7.45 GB


It takes some time to load, and a fair bit of RAM.  Now we can load values from one of the skim tables,
which is quick and easy, and only uses enough extra memory 
to store the values we have collected.


```python
with resource_usage:
    asim_data1 = asim_skims.get('DISTBIKE').get(otaz,dtaz)
```

    668 ms: Net 277 MB, Total 7.73 GB


# Parquet Matrix

Contrast that with the first of two formats of arrow matrix, ParquetMatrix.  
As we did above using the `skims_dict`, let's open the matrix reference itself first.


```python
with resource_usage:
    pqmx = amx.ParquetMatrix('data/mtc_full_skims.pmx')
```

    26.9 ms: Net 1e+03 KB, Total 7.73 GB


The matrix object can be created almost instantly because it
doesn't load all the data into RAM, just the schema and metadata.
The actual data remains on disk, waiting patiently for us to read
it later.  So let's do that!


```python
with resource_usage:
    pqmx_data1 = pqmx.get_rc('DISTBIKE', otaz-1, dtaz-1, attach_index=False).to_numpy().reshape(-1).astype('float32')
```

    264 ms: Net 39.8 MB, Total 7.77 GB


Loading this data from the arrow matrix requires barely more memory
footprint than the loaded data itself (the array of 10 million double-precision
floats uses 76.3 MB). 

Well, you may say, that's nothing special.  The whole point of the 
ActivitySim skims module is to be fast, by having the necessary skims
preloaded into RAM so they can be read from as fast as possible.


```python
%timeit asim_skims.get('DISTWALK').get(otaz,dtaz)
```

    566 ms ± 23.4 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)


_pqmx_: "Hold my beer"


```python
%timeit pqmx.get_rc('DISTWALK', otaz-1, dtaz-1, attach_index=False).to_numpy().reshape(-1)
```

    218 ms ± 10.1 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)


Of course, there's no free lunch. Arrow is super fast, but 
reading data from disk has a high fixed cost. In particular, 
for Parquet (as configured in this demo, at least) we need
to read and decompress the entire source matrix data, to 
extract what we need. We can beat the 
pre-loaded ActivitySim skims when the chunk size is very large, 
but for very small chunk sizes the RAM solution is much faster. 


```python
otaz2, dtaz2 = otaz[:50], dtaz[:50]
```


```python
%timeit asim_skims.get('DISTWALK').get(otaz2,dtaz2)
```

    9.47 µs ± 191 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)



```python
%timeit pqmx.get_rc('DISTWALK', otaz2-1, dtaz2-1, attach_index=False).to_numpy().reshape(-1)
```

    19.6 ms ± 858 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)


But what if I want the speed of in-memory data, but without actually needing to allocate all that memory?

# Feather Matrix


```python
with resource_usage:
    fmx = amx.FeatherMatrix('data/mtc_full_skims_uncompressed.fmx')
```

    1.08 s: Net 620 KB, Total 7.8 GB


Feather is able to point to space on disk and use it like RAM.  It's not quite as fast as
actual RAM, but these days solid state drives can get kind of close.  So, like ParquetMatrix above, 
we create the object reference almost instantly and with no overhead.

We can contrast now the performance with loading this big chunks...


```python
%timeit fmx.get_rc('DISTBIKE', otaz-1, dtaz-1, attach_index=False).to_numpy().reshape(-1)
```

    200 ms ± 7.69 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)


... and the small chunks.


```python
%timeit fmx.get_rc('DISTWALK', otaz2-1, dtaz2-1, attach_index=False).to_numpy().reshape(-1)
```

    140 µs ± 3.25 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)


Let's plot the relative speed across a variety of chunk sizes.

![png](speed.png)
    
Assuming a large enough chunk size, either format performs better than
the current ActivitySim implementation.  Even with a quite small chunk size,
the feather format performs reasonably well and with no RAM footprint.

