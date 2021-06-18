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

def genRandomVector(k, t):
   
    randomVector = np.zeros(k, dtype = np.int32)

    # choisir t positions aléatoires parmi k positions
    randomPositions = np.random.choice(k, t, replace=False)

    # attribuer aux positions aléatoires la valeur 1
    for j in randomPositions:
        randomVector[j] = 1
    
    return randomVector

def encryptMcEliece(G, m, e):
    rows, cols = G.shape
    n = cols
    r = cols - rows
    
    #le chiffrement suit le document de recherche, texte chiffré = m * G + e
    ciphertext = np.copy(np.add(np.matmul(m, G), e))
    
    ciphertext = convertBinary(ciphertext)

    #Décommentez ceci pour vérifier si le cryptage est correct
    #print("Plaintext : ", convertBinary(np.matmul(m, G)))
    #print("Error     : ", randomError)
    #print("Ciphertext: ", ciphertext)
    
    return ciphertext

def bitFlipping(H, c, N):

    print("\nStarting Bit-Flipping Algorithm...")
    rows, cols = H.shape
    
    #Le graphe de Tanner de la matrice de contrôle de parité est représenté à l'aide de listes adjancency
    bitsAdjList = [[] for i in range(cols)]
    checksAdjList = [[] for i in range(rows)]
    
    #nœuds de bits et nœuds de contrôle, et initialisation d'autres paramètres
    bits = np.copy(c)
    checks = np.zeros(rows)
    t = 0           #non. de tours
    numOnes = 0     #compte non. de uns dans les nœuds de contrôle, si numOnes = 0, le mot de code est décodé
    flipOrNot = 0   #utilisera le vote à la majorité pour décider si un morceau doit être retourné
    checkSum = 0    #juste une variable intermédiaire pour le calcul
    
    for i in range(rows):
        for j in range(cols):
            if H[i, j] == 1:
                bitsAdjList[j].append(i)
                checksAdjList[i].append(j)
    
    while t < N:
        #print("Bit-Flipping Decoding Round", t, ":")
        for i in range(rows):
            for j in checksAdjList[i]:
                checkSum += bits[j]
            checks[i] = checkSum % 2
            checkSum = 0
            if checks[i] == 1:
                numOnes += 1

        if numOnes == 0:
            break
        else:
            numOnes = 0
        
        for i in range(cols):
            for j in bitsAdjList[i]:
                if checks[j] == 1:
                    flipOrNot += 1
            if 2*flipOrNot > len(bitsAdjList[i]):
                bits[i] = (bits[i] + 1) % 2
            flipOrNot = 0

        t += 1
                
    if t < N:
        print("Decoded in", t, "step(s).")
        
        return bits
    else:
        print("Cannot decode")
    
    return 0

def decryptMcEliece(H, y, method, N):
  
    r, n = H.shape
    #Décryptage
    decryptedText = bitFlipping(H, y, N)
        
    if type(decryptedText) == int:
        print("Cannot decode by Bit-Flipping algorithm")
    else :
        decryptedText = decryptedText[0: n - r]
            
    return decryptedText

def decryptSuccess(plaintext, decryptedText):

    status = np.array_equal(plaintext, decryptedText)
    if (status == True):
        print("Succès du décryptage!")
    else:
        print("Échec du décryptage!")
        
    return status

def demo(H, y):
    print("H:\n", H)
    r, n = H.shape
    w = sum(H[0,:] == 1)
    d = w // 2
    iteration = 1
    flipped = 1

    s = convertBinary(np.matmul(y, np.transpose(H)))
    print("\ntext chiffre:", y)
    print("seuil T:", d)
    print("\n######### Démarrage de l'algorithme de Bit-Flipping... #########\n")

    print("s = yH^T:", s)
    while (np.count_nonzero(s) > 0 and flipped == 1):
        flipped = 0
        # syndrome weight
        T = 1

        for j in range(n):
            if (sum((s + H[:,j]) == 2)) >= T * d:
                print("FLIPPED position %d" % j)
                print("y:", y)
                y[j] = y[j] ^ 1
                
                s = convertBinary(np.matmul(y, np.transpose(H)))
                print("s = yH^T:", s)
                flipped = 1 
        iteration += 1
        # syndrome
        s = np.matmul(y, np.transpose(H))
        s = convertBinary(s)

    print("test Dechiffre:\n", y)
    if (sum(s == 1) == 0):
        return y[0: n-r]
    else:
        print("Cannot decode")
        return 0
    
    
##################################### Setting parameters ###################################

# code parameters
n0 = 2
r = 5
wi = 3

N = 20
k = r // 2
#decryption parameters
t = 1

################################## Processing parameters ###################################

n = n0 * r
w = wi * n0

##################################### Testing functions ####################################

# generate a random (n,r,w)-QC-MDPC matrix H
H = genQCMDPC(n, r, w)

# Generate the corresponding generator matrix G
G = genGenQCMDPC(H)
print("G:\n", G)
# generate a random message m of weight k
m = genRandomVector(r, k)

print("\Texte en clair m:", m)

# generate a random error vector e of weight t
e = genRandomVector(n, t)

# encrypt the message m
y = encryptMcEliece(G, m, e)

# decrypt the ciphertext
decryptedText = demo(H, y)

# check if decryption is correct
decryptSuccess(m, decryptedText)