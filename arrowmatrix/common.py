
import os
import ast
import json
import pathlib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.feather as pf
from abc import ABC, abstractmethod

from .exceptions import MissingShapeError

OMX_VERSION = b'0.3.0a'

def omx_hdf5_2_to_arrow(
		omx_file,
		*,
		shape=None,
):
	"""
	Convert the 2-d part of an HDF5 OMX file to a tall arrow table.

	Parameters
	----------
	omx_file : path-like or omx.File or pd.DataFrame
		The source data, as the path to an openmatrix file,
		or an open file handle, or as pre-loaded tall-format
		pandas DataFrame.
	shape : tuple, optional
		Only needed if `omx_file` is a pre-loaded DataFrame.

	Returns
	-------
	pyarrow.Table
	"""
	import openmatrix as omx # import here, optional dependency
	if isinstance(omx_file, (str, pathlib.Path)):
		omx_file = omx.open_file(omx_file)
	if isinstance(omx_file, pd.DataFrame):
		df = omx_file
		if shape is None:
			raise MissingShapeError('when using preloaded dataframe, must give shape')
	else:
		assert isinstance(omx_file, omx.File)
		shape = omx_file.shape()
		matrix_names = [
			node.name
			for node in omx_file.list_nodes(where="/data")
		]
		df = pd.DataFrame({
			k: omx_file.get_node(f"/data/{k}")[:].reshape(-1)
			for k in matrix_names
		})
	table = pa.table(df)
	metadata = table.schema.metadata
	metadata[b'OMX_VERSION'] = OMX_VERSION
	metadata[b'SHAPE'] = str(shape).encode()
	new_schema = table.schema.with_metadata(metadata)
	table = table.cast(new_schema)
	return table

def omx_hdf5_1_to_arrow(
		omx_file,
		*,
		shape=None,
):
	if isinstance(shape, tuple):
		if len(shape) == 1:
			shape = shape[0]
		else:
			raise ValueError("must give one dimensional shape")
	import openmatrix as omx # import here, optional dependency
	if isinstance(omx_file, str):
		omx_file = omx.open_file(omx_file)
	if isinstance(omx_file, pd.DataFrame):
		df = omx_file
		if shape is None:
			shape = len(df)
		else:
			assert np.prod(shape) == len(df)
	else:
		assert isinstance(omx_file, omx.File)
		shape_2 = omx_file.shape()
		if shape_2[0] == shape_2[1]:
			shape = shape_2[0]
		elif shape is None:
			raise MissingShapeError("must give shape for non-square file")
		else:
			assert shape == shape_2[0] or shape == shape_2[1]
		lookup_names = [
			node.name
			for node in omx_file.list_nodes(where="/lookup")
		]
		df_content = {}
		for k in lookup_names:
			v = omx_file.get_node(f"/lookup/{k}")[:].reshape(-1)
			if v.size == shape:
				df_content[k] = v
		df = pd.DataFrame(df_content)
	table = pa.table(df)
	metadata = table.schema.metadata
	metadata[b'OMX_VERSION'] = OMX_VERSION
	metadata[b'SHAPE'] = str(shape).encode()
	new_schema = table.schema.with_metadata(metadata)
	table = table.cast(new_schema)
	return table


def check_write_file(filename, overwrite=False):
	assert isinstance(filename, (str, pathlib.Path))
	if os.path.exists(filename) and not overwrite:
		raise FileExistsError(filename)


class AbstractArrowMatrix(ABC):

	@classmethod
	def from_hdf5(
			cls,
			omx_file,
			to_filename,
			*,
			overwrite=False,
			shape=None,
	):
		check_write_file(to_filename, overwrite=overwrite)
		table = omx_hdf5_2_to_arrow(omx_file, shape=shape)
		cls._write_arrow_table(to_filename, table, shape)
		return cls(to_filename)

	@classmethod
	def from_dataframe(
			cls,
			dataframe,
			to_filename,
			*,
			overwrite=False,
			shape=None,
	):
		check_write_file(to_filename, overwrite=overwrite)
		if shape is None:
			if isinstance(dataframe.index, pd.MultiIndex):
				shape = [i.size for i in dataframe.index.levels]
		table = omx_hdf5_1_to_arrow(dataframe, shape=shape)
		cls._write_arrow_table(to_filename, table, shape)
		return cls(to_filename)

	@classmethod
	def from_arrow(cls, source, to_filename, names=None, overwrite=False, **kwargs):
		check_write_file(to_filename, overwrite=overwrite)
		table = source._get_arrow_table(names=names)
		cls._write_arrow_table(to_filename, table, source.shape, **kwargs)
		return cls(to_filename)

	@classmethod
	def as_buffer(cls, source, names=None, **kwargs):
		table = source._get_arrow_table(names=names)
		buffer_stream = pa.BufferOutputStream()
		cls._write_arrow_table(buffer_stream, table, source.shape, **kwargs)
		buffer = buffer_stream.getvalue()
		return buffer

	@classmethod
	def buffered(cls, source, names=None, **kwargs):
		table = source._get_arrow_table(names=names)
		buffer_stream = pa.BufferOutputStream()
		cls._write_arrow_table(buffer_stream, table, source.shape, **kwargs)
		buffer = buffer_stream.getvalue()
		return cls(buffer)

	def __init__(
			self,
			filename,
			omx_version=None,
			shape=None,
	):
		self.filename = filename
		self._omx_version = omx_version
		if isinstance(shape, int):
			shape = (shape,)
		self._shape = shape

	@property
	def shape(self):
		if self._shape is None:
			raise MissingShapeError()
		return self._shape

	@property
	def ndims(self):
		return len(self.shape)

	@property
	def omx_version(self):
		return self._omx_version

	def _get_rc_preprocess(self, indexes, attach_index=('i','j')):
		index_names = []
		for index, letter in zip(indexes, 'ijklmnopqrstuvwxyz'):
			index_names.append(getattr(index, 'name', letter))
		indexes = [
			np.asarray(index).reshape(-1)
			for index in indexes
		]
		if attach_index:
			idx = pd.MultiIndex.from_arrays(
				indexes,
				names=index_names,
			)
		else:
			idx = None
		return indexes, idx

	@abstractmethod
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

	@staticmethod
	@abstractmethod
	def _write_arrow_table(filename, table, shape, **kwargs):
		"""
		Write a pyarrow.Table to a file.

		Parameters
		----------
		filename : path-like
			The location to write the data file.
		table : pyarrow.Table
			The data to write.  Any metadata required should
			already attached to the schema.
		shape : tuple
			The shape of the table being written.  Can be used
			to optimize the data structure on write.
		"""

	def get_matrix(self, name):
		"""
		Load a matrix into memory.

		Note that it is not always necessary to load an
		entire matrix at once.  To access parts of a matrix,
		try `get_rc`.

		Parameters
		----------
		name : str
			The name of the matrix to load.

		Returns
		-------
		numpy.ndarray
		"""
		t = self._get_arrow_table(names=[name])
		return t.to_pandas().to_numpy().reshape(self.shape)


	def _get_rc_by_takers(self, names, takers, method=4, dtype='float64'):
		if isinstance(names, str):
			# extracting a single column is faster than multiple columns
			# as we avoid the overhead of interpreting how the different
			# columns should be joined together in a dataframe.
			return self._get_arrow_table(names=[names]).take(takers).column(0).to_pandas()
		else:
			# Marginally slower for small data loads

			# method 4 is generally fastest and with smallest memory footprint
			# but other methods are left here for future performance validation & optimization
			if method==1:
				table = self._get_arrow_table(names=names)
				result = table.to_pandas().iloc[takers]
			elif method == 2:
				raw_result = np.zeros(shape=[len(takers), len(names)], dtype=dtype)
				for n,name in enumerate(names):
					table = self._get_arrow_table(names=[name]).to_pandas()
					raw_result[:,n] = table.iloc[takers,0]
				result = pd.DataFrame(
					raw_result,
					columns=names,
				)
			elif method==3:
				raw_result = np.zeros(shape=[len(takers), len(names)], dtype=dtype)
				for n,name in enumerate(names):
					table = self._get_arrow_table(names=[name]).take(takers).to_pandas()
					raw_result[:,n] = table.iloc[:,0]
				result = pd.DataFrame(
					raw_result,
					columns=names,
				)
			elif method==4:
				result = self._get_arrow_table(names=names).take(takers).to_pandas()
			else:
				raise ValueError(f"undefined method {method}")

			return result

	def _takers(self, *indexes, attach_index=False):
		if len(indexes) != self.ndims:
			raise ValueError(f'number of indexes ({len(indexes)}) does not match ndims ({self.ndims})')
		indexes, idx = self._get_rc_preprocess(indexes, attach_index)
		n = 1
		takers = indexes[0] * np.prod(self.shape[n:])
		for index in indexes[1:]:
			n += 1
			if n >= self.ndims:
				takers += index
			else:
				takers += index * np.prod(self.shape[n:])
		return takers, idx

	def get_raw(self, names=None):
		return self._get_arrow_table(names=names).to_pandas()

	def get_rc_table(self, names, *indexes):
		"""
		Extract values by index.

		Parameters
		----------
		names : str or Collection[str]
			The names of one or more matrix tables to load.
		*indexes : array-like or int
			The various index positions to load.  The number
			of tuple values must match the number of dimensions.

		Returns
		-------
		pyarrow.Table
		"""
		takers, _ = self._takers(*indexes, attach_index=False)
		if isinstance(names, str):
			return self._get_arrow_table(names=[names]).take(takers)
		else:
			return self._get_arrow_table(names=names).take(takers)

	def get_rc(self, names, *indexes, method=4, attach_index=True, dtype='float64'):
		"""
		Extract values by index.

		Parameters
		----------
		names : str or Collection[str]
			The names of one or more matrix tables to load.
		*indexes : array-like or int
			The various index positions to load.  The number
			of tuple values must match the number of dimensions.
		attach_index : bool or tuple
			Whether to attach a meaningful index to the
			output dataframe.  Set to a tuple of strings
			of length equal to the number of dimensions of
			the matrix to use these values as the names of
			the levels of the reulting MultiIndex.

		Returns
		-------
		pandas.DataFrame
		"""
		takers, idx = self._takers(*indexes, attach_index=attach_index)
		result = self._get_rc_by_takers(names, takers, method=method, dtype=dtype)
		if idx is None:
			result.reset_index(inplace=True, drop=True)
		else:
			result.index = idx
		return result

	@abstractmethod
	def list_matrices(self):
		"""list : Get a list of matrices in this file."""

	def __getitem__(self, item):
		"""Experimental, 2 dim only"""
		if isinstance(item, tuple):
			dims = len(self.shape)
			while len(item) < dims+1:
				item = item + (slice(None),)
		names = item[0]
		if isinstance(names, (str, bytes)):
			names = [names]
		add_multiindex = len(names) > 1
		if isinstance(names, slice):
			if names.start == None:
				names = self.list_matrices()
		slice1 = item[1]
		if isinstance(slice1, int):
			slice1 = slice(slice1, slice1+1)
		slice2 = item[2]
		if isinstance(slice2, int):
			slice2 = slice(slice2, slice2+1)
		takers, takers_shape = _take_array(slice1, slice2, self.shape)
		result = self._get_rc_by_takers(names, takers)
		if add_multiindex:
			idx1 = np.broadcast_to(_slice_to_range(slice1, self.shape[0]).reshape(-1,1), takers_shape).reshape(-1)
			idx2 = np.broadcast_to(_slice_to_range(slice2, self.shape[1]), takers_shape).reshape(-1)
			result.index = pd.MultiIndex.from_arrays([idx1,idx2])
			return result
		else:
			return result.to_numpy().reshape(takers_shape)

def _cap(a,b):
	if a is None:
		return b
	if b is None:
		return a
	return min(a,b)

def _slice_to_range(s, size):
	if isinstance(s, slice):
		return np.arange(s.start or 0, _cap(s.stop,size), s.step or 1)
	return np.asarray(s)

def _slice_size(s, size):
	return (_cap(s.stop,size) - (s.start or 0)) / (s.step or 1)

def _take_array(slice1, slice2, shape):
	# stride = shape[1]
	combined = np.add(
		_slice_to_range(slice1, shape[0]).reshape(-1,1)*shape[1],
		_slice_to_range(slice2, shape[1]),
	)
	return combined.reshape(-1), combined.shape

