import numpy as np
import numpy.linalg as ln
import sys

def incrementalSVD(mat_a1, mat_a2, k, only_uk=False):
  """Menerapkan SVD untuk matriks dengan kolom baru

  Parameters
  ----------
  mat_a1 : [type]
       matriks asli (m x n1)
  mat_a2 : [type]
      kolom baru (m x n2)
  k : [type]
      rank-k untuk hasil yang diperkirakan
  only_uk : bool, optional
      [description], by default False

  Returns
  -------
      peringkat-k mendekati U, S, V ^ T sebagai hasil dari svd ([mat_a1, mat_a2])

  """

  if mat_a1.shape[0] != mat_a2.shape[0]:
    raise ValueError('Error: the number of rows both in mat_a1 and mat_a2 should be the same')

  # get the number of rows and columns
  m = mat_a1.shape[0]
  n1 = mat_a1.shape[1]
  n2 = mat_a2.shape[1]

  if k < 1:
    raise ValueError('Error: rank k must be greater than or equal to 1')
  if k > min(m, n1 + n2):
    raise ValueError('Error: rank k must be less than or equal to min(m, n1 + n2)')

  # apply SVD for the original matrix
  mat_u1, vec_s1, mat_v1t = ln.svd(mat_a1, full_matrices=False)
  mat_s1 = np.diag(vec_s1)

  # define mat_f as [S, U^T A_1], and decompose it by SVD
  mat_f = np.hstack((mat_s1, np.dot(mat_u1.T, mat_a2)))
  mat_uf, vec_sf, mat_vft = ln.svd(mat_f, full_matrices=False)

  # keep rank-k approximation
  mat_uf = mat_uf[:, :k]
  if only_uk: return  np.dot(mat_u1, mat_uf)
  vec_sf = vec_sf[:k]
  mat_vft = mat_vft[:k, :]

  # create a temporary matrix to compute V_k
  V = mat_v1t.T
  Z1 = np.zeros((n1, n2))
  Z2 = np.zeros((n2, V.shape[1]))
  I = np.eye(n2)
  mat_tmp = np.vstack((
      np.hstack((V, Z1)),
      np.hstack((Z2, I))
    ))
  mat_vk = np.dot(mat_tmp, mat_vft.T)

  # compute U_k and S_k
  mat_uk = np.dot(mat_u1, mat_uf)
  mat_sk = np.diag(vec_sf)

  return mat_uk, mat_sk, mat_vk.T
