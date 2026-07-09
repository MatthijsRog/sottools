import pickle

import numpy as np


def test_pickle_stiffness_matrix_solver(bar_mesh):
    pickled_mesh = pickle.dumps(bar_mesh)
    unpickled_mesh = pickle.loads(pickled_mesh)

    # Check if the stiffness matrix solver is properly reconstructed
    sms_A = bar_mesh.stiffness_matrix_solver
    sms_B = unpickled_mesh.stiffness_matrix_solver
    assert np.array_equal(sms_A.perm_r, sms_B.perm_r)
    assert np.array_equal(sms_A.perm_c, sms_B.perm_c)
    assert (sms_A.L - sms_B.L).nnz == 0
    assert (sms_A.U - sms_B.U).nnz == 0
