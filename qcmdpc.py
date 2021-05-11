import csv
import math
from sympy import *
import numpy as np
from scipy import special
from scipy.stats import binom, hypergeom
import networkx as nx
from networkx.algorithms import bipartite
import matplotlib.pyplot as plt
import random

def genFirstRow(r, wi):

    firstRow = np.zeros(r, dtype=np.int32)
    # choisir les nombres wi de la plage (wi)
    randomArray = np.random.choice(r, wi, replace=False)

    for i in randomArray:
        firstRow[i] = 1

    return firstRow

def genCirculant(firstRow):
   
    rows = firstRow.size
    firstRow = np.array(firstRow, dtype = np.int32)
    M = np.zeros((rows, rows), dtype = np.int32)
    M[0,:] = firstRow

    for i in range(1,rows):
        for j in range(rows):
            M[i,j] = M[i-1, (j-1)%rows]
    return M

def genTransposePoly(firstRow):
  
    transposePoly = np.zeros(len(firstRow), dtype = np.int32)
    transposePoly[0] = firstRow[0]
    transposePoly[1:] = np.flip(firstRow[1:])
    return transposePoly

def genProdPoly(firstRowA, firstRowB):

    r = len(firstRowA)
    prodPoly = np.zeros(r, dtype = np.int32)
    convolvePoly = np.convolve(firstRowA, firstRowB)

    if len(convolvePoly) > r:
        prodPoly[0:r] = convolvePoly[0:r]
        prodPoly[0: len(convolvePoly) - r] += convolvePoly[r:]
    else:
        prodPoly[0: len(convolvePoly)] = convolvePoly[0: len(convolvePoly)]
        
    return prodPoly

def genInvPoly(firstRow):
  
    r = len(firstRow)
    invPoly = np.zeros(r, dtype = np.int32)

    # convertir un tableau numpy en polynôme
    inv = convertNumpyToSympy(firstRow)

    #définir un anneau polynomial, F_2 / (x**r - 1)
    FField = "x**" + str(r) + "-" + str(1)
    FField = poly(FField, domain=FF(2))

    #recherche de l'inverse du polynôme dans l'anneau polynomial spécifié
    inv = invert(poly(inv, domain=FF(2)), poly(FField, domain=FF(2)))
    temp = convertSympyToNumpy(inv)

    invPoly[0 : len(temp)] = temp
    
    return invPoly

def convertBinary(v):
   
    for i in range(len(v)):
        v[i] = v[i] % 2

    return v

def convertNumpyToSympy(f):
   
    polynomial = ""
    polynomial = polynomial + str(int(f[0]))
    
    for i in range(1, f.size):
        if f[i] != 0:
            polynomial += " + " + str(int(f[i])) + "*x**" + str(i)
    return polynomial

def convertSympyToNumpy(f):
    v = np.array(f.all_coeffs())

    #Sympy stocke les coefficients du polynôme f (x) en puissances décroissantes de x
    #ainsi l'ordre est inversé pour obtenir un tableau numpy de puissances croissantes de x
    v = np.flip(v)
    
    return v

def genQCMDPC(n, r, w):
    n0 = int(n / r)
    wi = int(w / n0)
    
    H = np.zeros((r, r * n0), dtype = np.int32)
    H_i = np.zeros((r, r), dtype = np.int32)

    for i in range(n0):
        firstRow = genFirstRow(r, wi)
        H_i = genCirculant(firstRow)

        if i == 0:
            H = H_i
        else :
            H = np.concatenate((H, H_i), axis=1)

    filename = str(n) + "_" + str(r) + "_" + str(w) + "_" + "parityCheckMatrix.csv"

    np.savetxt(filename, H, delimiter = ",", fmt = "%d")

    return H

def genGenQCMDPC(H):
   
    r, n = H.shape
    n0 = int(n / r)
    H_i = np.zeros((r, r), dtype = np.int32)
    G = np.eye(n - r, dtype = np.int32)
    block0 = np.zeros((r, r), dtype = np.int32)

    #extraire les polynômes générateurs pour h_ {n_0-1} de H
    lastPoly = H[0, n - r : n]

    #calculer le polynôme générateur de l'inverse de h_ {n_0-1}
    invLastPoly = genInvPoly(lastPoly)

    #calculer le premier bloc circulant pour G
    temp = genProdPoly(invLastPoly, H[0, 0:r])
    temp1 = convertBinary(temp)
    temp2 = genTransposePoly(temp1)
    block0 = genCirculant(temp2)

    #calculer le bloc circulant suivant pour G et les concaténer
    for i in range(1, n0 - 1):
        temp = genProdPoly(invLastPoly, H[0, i*r : (i+1)*r])
        temp1 = convertBinary(temp)
        temp2 = genTransposePoly(temp1)
        block = genCirculant(temp2)
        block0 = np.concatenate((block0, block), axis = 0)

    if (n0 == 1):
        return G
    else :
     #concaténer la matrice d'identité d'ordre (n - r) et l'empilement de matrices circulantes pour former G
        G = np.concatenate((G, block0), axis = 1)

    w = sum(H[0, :])
    
    filename = str(n) + "_" + str(r) + "_" + str(w) + "_" + "GeneratorMatrix.csv"

    np.savetxt(filename, G, delimiter = ",", fmt = "%d")
    
    return G