import pytest
import os
import pandas as pd
import numpy as np
import openmatrix as omx
import arrowmatrix as amx

@pytest.fixture(scope="module", params=["feather","parquet"])
def arrow_matrix(request):
	if request.param == 'feather':
		filename = "temp_skims.feathermatrix"
		cls = amx.FeatherMatrix
	elif request.param == 'parquet':
		filename = "temp_skims.pqmx"
		cls = amx.ParquetMatrix
	else:
		raise ValueError
	if os.path.exists(filename):
		os.remove(filename)
	return cls.from_hdf5("data/tiny-skims.omx", filename)


def test_single_table(arrow_matrix):
	ref_matrix = omx.open_file("data/tiny-skims.omx")
	np.testing.assert_array_equal(
		ref_matrix.get_node("/data/SOV_TIME__AM")[:],
		arrow_matrix.get_matrix('SOV_TIME__AM')
	)


def test_rc(arrow_matrix):
	sov_time_names = [
		'SOV_TIME__EA',
		'SOV_TIME__AM',
		'SOV_TIME__MD',
		'SOV_TIME__PM',
		'SOV_TIME__EV',
	]
	o = [1, 2, 3, 4, 8, 6]
	d = [9, 7, 5, 6, 3, 0]
	refdata = {}
	ref_matrix = omx.open_file("data/tiny-skims.omx")
	for j in sov_time_names:
		refdata[j] = ref_matrix.get_node(f"/data/{j}")[:]
	ref_rc = pd.DataFrame({
		j: refdata[j][o, d]
		for j in sov_time_names
	})
	pd.testing.assert_frame_equal(
		ref_rc,
		arrow_matrix.get_rc(sov_time_names, o, d, attach_index=False),
	)
	np.testing.assert_array_equal(
		arrow_matrix.get_rc(sov_time_names, o, d, attach_index=True).index.get_level_values(0),
		o
	)
	np.testing.assert_array_equal(
		arrow_matrix.get_rc(sov_time_names, o, d, attach_index=True).index.get_level_values(1),
		d
	)
	# test single name
	pd.testing.assert_series_equal(
		ref_rc['SOV_TIME__AM'],
		arrow_matrix.get_rc('SOV_TIME__AM', o, d, attach_index=False),
	)


def test_basic_metadata(arrow_matrix):
		assert arrow_matrix.shape == (25, 25)
		assert len(arrow_matrix.list_matrices()) == 826
		np.testing.assert_array_equal(
			arrow_matrix.list_matrices()[:5],
			[
				'DIST',
				'DISTBIKE',
				'DISTWALK',
				'DRV_COM_WLK_BOARDS__AM',
				'DRV_COM_WLK_BOARDS__EA',
			]
		)
