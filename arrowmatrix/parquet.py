import os
import ast
import warnings
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from .common import AbstractArrowMatrix


class ParquetMatrix(AbstractArrowMatrix):

	def __init__(
			self,
			filename,
			omx_version=None,
			shape=None,
			buffer=False,
	):
		if buffer:
			filename = pa.py_buffer(pa.input_stream(filename).read())
		self.parquet_file = pq.ParquetFile(filename)
		schema = pq.read_schema(filename)
		shape = ast.literal_eval(schema.metadata[b'SHAPE'].decode())
		omx_version = schema.metadata[b'OMX_VERSION'].decode()
		num_rows = self.parquet_file.metadata.num_rows
		if np.prod(shape) != num_rows:
			warnings.warn(f"{self.__class__.__name__} shape {shape} not consistent with {num_rows} rows")
		super().__init__(filename=filename, shape=shape, omx_version=omx_version)

	def _get_arrow_table(self, names=None):
		return self.parquet_file.read(columns=names)

	@staticmethod
	def _write_arrow_table(filename, table, shape, **kwargs):
		pq.write_table(table=table, where=filename, **kwargs)

	def list_matrices(self):
		return self.parquet_file.schema.names

