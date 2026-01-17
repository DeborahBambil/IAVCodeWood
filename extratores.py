#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
from skimage import feature, measure, color, exposure
from skimage.feature import graycomatrix, graycoprops
from skimage.filters import prewitt_h, prewitt_v
import cv2
import os
import tempfile

# Tipo de atributos utilizados pelo Weka
numerico = 'numeric'
nominal = 'nominal'

# Diversos parâmetros utilizados no pré-processamento
cannyMin = 100
cannyMax = 200
glcmNiveis = 256
lbpRaio = 2
nBins = 18

# Controla algumas opções de apresentação de resultados
salvaImagens = False

class Extratores(object):
    def __init__(self):
        self.imagem = None
        self.imagemTonsDeCinza = None
        self.imagemBinaria = None
        self.imagemBorda = None
        self.imagemTamanhoFixo = None
        self.sequenciaImagens = 1
        self.tmp_dir = tempfile.gettempdir()

    def estatisticas_cores(self):
        imagemHSV = cv2.cvtColor(self.imagem, cv2.COLOR_BGR2HSV)
        imagemCIELab = cv2.cvtColor(self.imagem, cv2.COLOR_BGR2LAB)

        b, g, r = cv2.split(self.imagem)
        h, s, v = cv2.split(imagemHSV)
        ciel, ciea, cieb = cv2.split(imagemCIELab)

        nomes = [
            'cor_rmin', 'cor_rmax', 'cor_rmedia', 'cor_rdesvio',
            'cor_gmin', 'cor_gmax', 'cor_gmedia', 'cor_gdesvio',
            'cor_bmin', 'cor_bmax', 'cor_bmedia', 'cor_bdesvio',
            'cor_hmin', 'cor_hmax', 'cor_hmedia', 'cor_hdesvio',
            'cor_smin', 'cor_smax', 'cor_smedia', 'cor_sdesvio',
            'cor_vmin', 'cor_vmax', 'cor_vmedia', 'cor_vdesvio',
            'cor_cielmin', 'cor_cielmax', 'cor_cielmedia', 'cor_cieldesvio',
            'cor_cieamin', 'cor_cieamax', 'cor_cieamedia', 'cor_cieadesvio',
            'cor_ciebmin', 'cor_ciebmax', 'cor_ciebmedia', 'cor_ciebdesvio'
        ]

        tipos = [numerico] * len(nomes)
        canais = [r, g, b, h, s, v, ciel, ciea, cieb]
        valores = []
        for c in canais:
            valores.extend([float(np.min(c)), float(np.max(c)), float(np.mean(c)), float(np.std(c))])

        return nomes, tipos, valores

    def momentos_hu(self):
        M = measure.moments(self.imagemTonsDeCinza)
        if M[0, 0] == 0: return [f"hu_{i}" for i in range(7)], [numerico]*7, [0.0]*7
        row = int(M[1, 0] / M[0, 0])
        col = int(M[0, 1] / M[0, 0])
        row = max(0, min(row, self.imagemTonsDeCinza.shape[0] - 1))
        col = max(0, min(col, self.imagemTonsDeCinza.shape[1] - 1))
        mu = measure.moments_central(self.imagemTonsDeCinza, (row, col))
        nu = measure.moments_normalized(mu)
        hu = measure.moments_hu(nu)
        valores = [float(x) for x in hu]
        nomes = [f"hu_{i}" for i in range(len(valores))]
        return nomes, [numerico]*len(nomes), valores

    def matriz_coocorrencia(self):
        g = feature.graycomatrix(self.imagemTonsDeCinza, [1, 2], [0, np.pi/4, np.pi/2], glcmNiveis, normed=True, symmetric=True)
        props = ['contrast', 'dissimilarity', 'homogeneity', 'ASM', 'energy', 'correlation']
        todos_valores = []
        for p in props:
            todos_valores.extend(feature.graycoprops(g, p).flatten().tolist())
        
        nomes = [f'glcm_{p}_{i}' for p in props for i in range(6)]
        return nomes, [numerico]*len(nomes), [float(x) for x in todos_valores]

    def hog(self):
        valores = feature.hog(self.imagemTamanhoFixo, orientations=8, pixels_per_cell=(32, 32),
                             cells_per_block=(1, 1), block_norm='L1', visualize=False)
        valores = [float(x) for x in valores]
        nomes = [f"hog_{i}" for i in range(len(valores))]
        return nomes, [numerico]*len(nomes), valores

    def lbp(self):
        lbp_map = feature.local_binary_pattern(self.imagemTonsDeCinza, 8 * lbpRaio, lbpRaio, 'uniform')
        hist, _ = np.histogram(lbp_map, density=True, bins=nBins, range=(0, nBins))
        valores = [float(x) for x in hist]
        nomes = [f"lbp_{i}" for i in range(len(valores))]
        return nomes, [numerico]*len(nomes), valores

    def filtro_gabor(self):
        gabor_features = cv2.getGaborKernel((10, 10), 5.0, np.pi / 4, 10.0, 0.5, 0, ktype=cv2.CV_32F)
        filtered_img = cv2.filter2D(self.imagemTonsDeCinza, cv2.CV_8UC3, gabor_features)
        valores = [float(np.mean(filtered_img)), float(np.std(filtered_img))]
        return ['gabor_mean', 'gabor_std'], [numerico, numerico], valores

    def transformada_fourier(self):
        fft = np.fft.fft2(self.imagemTonsDeCinza)
        fft_shifted = np.fft.fftshift(fft)
        mag = 20 * np.log(np.abs(fft_shifted) + 1e-9)
        valores = [float(np.mean(mag)), float(np.std(mag)), float(np.max(mag)), float(np.min(mag))]
        nomes = ['fft_media', 'fft_std', 'fft_max', 'fft_min']
        return nomes, [numerico]*4, valores

    def bordas_canny(self):
        hist, _ = np.histogram(self.imagemBorda, bins=2, range=(0, 256))
        valores = [float(x) for x in hist]
        nomes = [f"canny_{i}" for i in range(len(valores))]
        return nomes, [numerico]*len(nomes), valores

    def scharr(self):
        sx = cv2.Scharr(self.imagemTonsDeCinza, cv2.CV_64F, 1, 0)
        sy = cv2.Scharr(self.imagemTonsDeCinza, cv2.CV_64F, 0, 1)
        mag = np.sqrt(sx**2 + sy**2)
        mag = cv2.normalize(mag, None, 0, 1, cv2.NORM_MINMAX)
        hist, _ = np.histogram(mag, density=True, bins=10, range=(0, 1))
        return [f"scharr_{i}" for i in range(10)], [numerico]*10, [float(x) for x in hist]

    def laplacian_descriptor(self):
        lap = cv2.Laplacian(self.imagemTonsDeCinza, cv2.CV_64F)
        mag = cv2.normalize(np.absolute(lap), None, 0, 1, cv2.NORM_MINMAX)
        hist, _ = np.histogram(mag, bins=10, range=(0, 1), density=True)
        return [f"laplacian_{i}" for i in range(10)], [numerico]*10, [float(x) for x in hist]

    def sobel_descriptor(self):
        sx = cv2.Sobel(self.imagemTonsDeCinza, cv2.CV_64F, 1, 0, ksize=3)
        sy = cv2.Sobel(self.imagemTonsDeCinza, cv2.CV_64F, 0, 1, ksize=3)
        mag = cv2.normalize(np.sqrt(sx**2 + sy**2), None, 0, 1, cv2.NORM_MINMAX)
        hist, _ = np.histogram(mag, bins=10, range=(0, 1), density=True)
        return [f"sobel_{i}" for i in range(10)], [numerico]*10, [float(x) for x in hist]

    def prewitt(self):
        px = prewitt_h(self.imagemTonsDeCinza)
        py = prewitt_v(self.imagemTonsDeCinza)
        mag = cv2.normalize(np.sqrt(px**2 + py**2), None, 0, 1, cv2.NORM_MINMAX)
        hist, _ = np.histogram(mag, density=True, bins=10, range=(0, 1))
        return [f"prewitt_{i}" for i in range(10)], [numerico]*10, [float(x) for x in hist]

    def extrai_todos(self, imagem):
        if imagem is None or imagem.size == 0: return None, None, None
        
        self.imagem = imagem
        self.imagemTonsDeCinza = cv2.cvtColor(self.imagem, cv2.COLOR_BGR2GRAY)
        self.imagemBorda = cv2.Canny(self.imagemTonsDeCinza, cannyMin, cannyMax)
        _, self.imagemBinaria = cv2.threshold(self.imagemTonsDeCinza, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        self.imagemTamanhoFixo = cv2.resize(self.imagemTonsDeCinza, (128, 128))

        if salvaImagens:
            prefix = os.path.join(self.tmp_dir, f"vcode_{self.sequenciaImagens}")
            cv2.imwrite(f"{prefix}_orig.jpg", self.imagem)
            cv2.imwrite(f"{prefix}_edge.jpg", self.imagemBorda)
            self.sequenciaImagens += 1

        extratores_lista = [
            self.estatisticas_cores, self.momentos_hu, self.matriz_coocorrencia,
            self.hog, self.lbp, self.filtro_gabor, self.transformada_fourier,
            self.bordas_canny, self.scharr, self.laplacian_descriptor,
            self.sobel_descriptor, self.prewitt
        ]

        t_nomes, t_tipos, t_valores = [], [], []
        for e in extratores_lista:
            n, t, v = e()
            t_nomes.extend(n)
            t_tipos.extend(t)
            t_valores.extend(v)

        return t_nomes, t_tipos, t_valores