from .incremental_svd import *
import numpy as np 
def incrementalApproSVD(mat_b1, mat_b2, c1, c2, k, p1, p2):
  """ Menerapkan Incremental ApproSVD untuk matriks dengan kolom baru

   Parameter
   ----------
   mat_b1: [tipe]
       matriks asli (m x n1)
   mat_b2: [tipe]
       kolom baru (m x n2)
   c1: [tipe]
       jumlah kolom sampel dari B1
   c2: [tipe]
       jumlah kolom sampel dari B2
   k: [tipe]
        peringkat-k untuk hasil yang diperkirakan
   p1: [jenis]
       probabilitas pengambilan sampel untuk setiap kolom di B1
   p2: [jenis]
       probabilitas pengambilan sampel untuk setiap kolom di B2
 
   Kembali
   -------
   tupel
       H_k sebagai output dari Incremental ApproSVD (H_k H_k ^ T = I)
  """
 
  if mat_b1.shape[0] != mat_b2.shape[0]:
    raise ValueError('Kesalahan: jumlah baris di mat_a1 dan mat_a2 harus sama')

  if len(p1[p1<0]) != 0:
    raise ValueError('Kesalahan: probabilitas negatif di p1 tidak diperbolehkan')
  if len(p2[p2<0]) != 0:
    raise ValueError('Kesalahan: probabilitas negatif dalam p2 tidak diperbolehkan')

  if not np.isclose(sum(p1), 1.):
    raise ValueError('Kesalahan: jumlah probabilitas harus 1 untuk p1')
  if not np.isclose(sum(p2), 1.):
    raise ValueError('Kesalahan: jumlah probabilitas harus 1 untuk p2')

  # get the number of rows and columns
  m = mat_b1.shape[0]
  n1 = mat_b1.shape[1]
  n2 = mat_b2.shape[1]

  if c1 >= n1:
    raise ValueError('Kesalahan: c1 harus lebih kecil dari n1')
  if c2 >= n2:
    raise ValueError('Kesalahan: c2 harus lebih kecil dari n2')

  if k < 1:
    raise ValueError('Kesalahan: peringkat k harus lebih besar dari atau sama dengan 1')
  if k > min(m, c1 + c2):
    raise ValueError('Kesalahan: harus kurang dari atau sama dengan min (m, n1 + n2) ')

  # sample kolom c1 dari B1, dan gabungkan sebagai matriks C1 
  mat_c1 = np.zeros((m, c1))
  samples = np.random.choice(list(range(n1)), c1, replace=False, p=p1)
  for t in range(c1):
    mat_c1[:, t] = mat_b1[:, samples[t]] / np.sqrt(c1 * p1[samples[t]])

  # sample kolom c2 dari B2, dan gabungkan sebagai matriks C2
  mat_c2 = np.zeros((m, c2))
  samples = np.random.choice(list(range(n2)), c2, replace=False, p=p2)
  for t in range(c2):
    mat_c2[:, t] = mat_b2[:, samples[t]] / np.sqrt(c2 * p2[samples[t]])

  # terapkan IncSVD  untuk matriks C1, C2 yang lebih kecil, dan dapatkan hanya U_k sebagai H_k 
  return incrementalSVD(mat_c1, mat_c2, k, True)
