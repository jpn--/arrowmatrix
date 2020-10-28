import os
import gc as _gc
import timeit
import resource
import time
import numpy as np
import pyarrow as pa
import pandas as pd
import pyarrow.feather as pf
try:
	import psutil
except:
	psutil = None

#       nano micro milli    kilo mega giga tera peta exa  zeta yotta
tiers = ['n', 'µ', 'm', '', 'K', 'M', 'G', 'T', 'P', 'E', 'Z', 'Y']


def si_units(x, kind='B', f="{:.3g} {}{}"):
	tier = 3
	shift = 1024 if kind=='B' else 1000
	if x > 0:
		while x > 1024 and tier < len(tiers):
			x /= shift
			tier += 1
		while x < 1 and tier >= 0:
			x *= shift
			tier -= 1
	return f.format(x,tiers[tier],kind)


class MemoryUsage:

	def __init__(self):
		self.memory_history = [0,]
		self.max_memory_history = [0,]
		self.pid = os.getpid()  # the current process identifier, to track memory usage
		if psutil is None:
			raise ModuleNotFoundError("pstil")
		self.check()

	def check(self, silent=False, gc=False, time_checkpoint=None):
		if gc:
			_gc.collect()
		now_memory = psutil.Process(self.pid).memory_info().rss
		max_memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
		marginal_usage = now_memory - self.memory_history[-1]
		marginal_max = max_memory - self.max_memory_history[-1]
		if time_checkpoint:
			time_note = si_units(time.time()-time_checkpoint, kind='s') + ": "
		else:
			time_note = ""
		if not silent:
			print(f"{time_note}Net {si_units(marginal_usage)}, Total {si_units(now_memory)}")
		self.memory_history.append(now_memory)
		self.max_memory_history.append(max_memory)

	def __enter__(self):
		_gc.collect()
		_gc.disable()
		self._context_start_time = time.time()
		self.check(silent=True, gc=False)

	def __exit__(self, exc_type, exc_val, exc_tb):
		self.check(time_checkpoint=self._context_start_time)
		_gc.enable()
		_gc.collect()


resource_usage = MemoryUsage()


def timing(stmt, setup='pass', repeat=10, globals=None, quiet=False):
	t = timeit.Timer(
		stmt,
		setup=setup,
		globals=globals,
	)
	n, duration = t.autorange()
	timings = np.asarray([duration]+t.repeat(repeat-1, n)) / n
	_mean = si_units(np.mean(timings), 's')
	_std = si_units(np.std(timings), 's')
	_min = si_units(min(timings), 's')
	_max = si_units(max(timings), 's')
	if not quiet:
		if n > 1:
			print(f"{_mean} ± {_std} per loop (mean ± std of {repeat} runs, {n} loops each), {_min} to {_max}")
		else:
			print(f"{_mean} ± {_std} per run (mean ± std of {repeat} runs), {_min} to {_max}")
	return timings


def to_simple_buffer(table, compression="zstd"):
	"""
	Write a pandas DataFrame to a pyarrow buffer, stripping pandas metadata.

	Parameters
	----------
	table : pyarrow.Table or pandas.DataFrame
	compression : {'zstd', 'lz4', 'uncompressed'}

	Returns
	-------
	pyarrow.buffer
	"""
	if isinstance(table, pd.DataFrame):
		table = pa.Table.from_pandas(
			table, preserve_index=False, nthreads=None, columns=None,
		)
		# strip pandas metadata
		table = pa.Table.from_batches(
			table.to_batches(),
			schema=table.schema.with_metadata(None),
		)
	sink = pa.BufferOutputStream()
	pf.write_feather(
		table,
		sink,
		compression=compression,
	)
	return sink.getvalue()
