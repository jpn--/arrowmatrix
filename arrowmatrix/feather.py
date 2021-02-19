import os
import ast
import json
import numpy as np
import pandas as pd
import pyarrow.feather as pf
import pyarrow.ipc as ipc
from .common import AbstractArrowMatrix


class FeatherMatrix(AbstractArrowMatrix):

	def __init__(self, filename, memory_map=True):
		self.filename = filename
		schema = ipc.open_file(filename).schema
		self._column_defs = schema.names
		omx_version = schema.metadata[b'OMX_VERSION'].decode()
		shape = ast.literal_eval(schema.metadata[b'SHAPE'].decode())
		super().__init__(filename=filename, shape=shape, omx_version=omx_version)
		self._table = pf.read_table(self.filename, memory_map=memory_map)

	def list_matrices(self):
		return self._column_defs

	def _get_arrow_table(self, names=None):
		"""
		Get a pyarrow.Table for named columns.

		Parameters
		----------
		names : Collection[str], optional
			Column names to load.  If not given, load all columns.

		Returns
		-------
		pyarrow.Table
		"""
		if names is None:
			return self._table
		else:
			return self._table.select(names)

	@staticmethod
	def _write_arrow_table(filename, table, shape, **kwargs):
		"""
		Write to Feather format.

		Parameters
		----------
		filename : str
			Local destination path.
		table : pandas.DataFrame or pyarrow.Table
			Data to write out as Feather format.
		shape : list-like
			Dimensions of the matrix being written.
		compression : string, default None
			Can be one of {"zstd", "lz4", "uncompressed"}. The default of None uses
			LZ4 for V2 files if it is available, otherwise uncompressed.
		compression_level : int, default None
			Use a compression level particular to the chosen compressor. If None
			use the default compression level
		chunksize : int, default None
			For V2 files, the internal maximum size of Arrow RecordBatch chunks
			when writing the Arrow IPC file format. None means use the default,
			which is currently 64K
		version : int, default 2
			Feather file version. Version 2 is the current. Version 1 is the more
			limited legacy format
		"""
		if 'chunksize' not in kwargs:
			kwargs['chunksize'] = np.prod(shape)
		pf.write_feather(table, filename, **kwargs)
