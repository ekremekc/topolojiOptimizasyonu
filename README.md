# topolojiOptimizasyonu
denemeler

# Sanal python düzlemini etkinleştirmek için
```bash
source .venv/bin/actıvate
```

# Kullanılan kütüphaneler

```bash
pip3 install scipy matplotlib
```

# scipy kütüophanelerini hızlandırmak için (swig kısmında hata var)

```bash
sudo apt-get install libsuitesparse-dev
pip3 install swig
pip3 install scikit-umfpack
```

# PETSc ile çok daha hızlı matris çözümlemek için

PETSc kurulum linki [burada](https://petsc.org/release/petsc4py/install.html)

```bash
sudo apt install libopenmpi-dev # openmpi yukler
python -m pip install mpi4py petsc petsc4py
```
