# !/usr/local/lib/python2.7 python 
# -*- coding=utf-8 -*-  

# bridging the python 2 and python 3 gap
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import pymatgen as mg
from pymatgen.io.vasp.inputs import Poscar
import tensorflow as tf
import os # saving files





# data processing
def reduced_distribution(li):

    def distribution(u, d={}):
        if u not in d:
            d[u] = None
            return True

    return [ e for e in li if distribution(tuple(e)) ]



# Compute first neighbors for each crystallographic structure from its POSCAR file

def neighbors(struct):
    radius = 4.

    p = Poscar.from_file(struct)
    Natoms = len(p.structure)
    s = mg.Structure.from_file(struct)
    allneig = s.get_all_neighbors(radius)
    atom1 = []; neigh = []; atoms = [];
    for i in s:
        atom1.append(i.specie.symbol)
    for neig in allneig:
        neig.sort(key=lambda x: x[1])
        X = [[iat.specie.symbol, jat] for iat,jat in neig]
        neigh.append(X)
    data = [] 
    for i in range(len(neigh)):
        liste = neigh[i]
        nli = reduced_distribution(liste)
        data.append(nli)
    s= []
    for i in range(len(atom1)):
        if atom1[i]=='H':
            for j in range(len(data[i])):
                if data[i][j][0] == 'H':
                    s.append(data[i][j][1])
                    break
                else:
                    continue
        if atom1[i] == 'Pd':
            for k in range(len(data[i])):
                if data[i][k][0] == 'Pd' or data[i][k][0] == 'Ni':
                    s.append(data[i][k][1])
                    break
                else:
                    continue
        if atom1[i] == 'Ni':
            for h in range(len(data[i])):
                if data[i][h][0] == 'Pd' or data[i][h][0] == 'Ni':
                    s.append(data[i][h][1])
                    break
                else:
                    continue
    return(s)


def all_neighbors(POSCAR_folder,radius):
    s_i = []
    for element in sorted(os.listdir(POSCAR_folder)):
        struct = os.path.join(POSCAR_folder, element)  
        p = Poscar.from_file(struct)
        Natoms = len(p.structure)
        s = mg.Structure.from_file(struct)
        allneig = s.get_all_neighbors(radius)
        atom1 = []; neigh = []; atoms = [];
        for i in s:
            atom1.append(i.specie.symbol)
        for neig in allneig:
            neig.sort(key=lambda x: x[1])
            X = [[iat.specie.symbol, jat] for iat,jat in neig]
            neigh.append(X)
        data = [] 
        for i in range(len(neigh)):
            liste = neigh[i]
            nli = reduced_distribution(liste)
            data.append(nli)
        s= []
        for i in range(len(atom1)):
            if atom1[i]=='H':
                for j in range(len(data[i])):
                    if data[i][j][0] == 'H':
                        s.append(data[i][j][1])
                        break
                    else:
                        continue
            if atom1[i] == 'Pd':
                for k in range(len(data[i])):
                    if data[i][k][0] == 'Pd' or data[i][k][0] == 'Ni':
                        s.append(data[i][k][1])
                        break
                    else:
                        continue
            if atom1[i] == 'Ni':
                for h in range(len(data[i])):
                    if data[i][h][0] == 'Pd' or data[i][h][0] == 'Ni':
                        s.append(data[i][h][1])
                        break
                    else:
                        continue
        
        s_i.append(s)
    return(s_i)


def geo_constraints(s_i):

    distances_1 = []; distances_2 = [];

    for i in range(len(s_i)):
        dist_1=[]; dist_2=[];
        dist_1 = [j-1.8 for j in s_i[i]]
        dist_2 = [3-j for j in s_i[i]]

        distances_1.append(dist_1)
        distances_2.append(dist_2)

    return(distances_1,distances_2)

def list_of_norms(X):
    '''
    X is a list of vectors X = [x_1, ..., x_n], we return
        [d(x_1, x_1), d(x_2, x_2), ... , d(x_n, x_n)], where the distance
    function is the squared euclidean distance.
    '''
    return tf.reduce_sum(tf.pow(X, 2), axis=0)

def geo_loss(distances_1,distances_2):

	"""geo_1 = tf.reduce_mean(tf.pow(distances_1, 2))
	geo_2 = tf.reduce_mean(tf.pow(distances_1, 2))"""
	geometric_loss = list_of_norms(distances_1) + list_of_norms(distances_2)
	opt = tf.train.AdamOptimizer(1e-4, beta1=0.5)
	geo_opt = opt.minimize(geometric_loss)

	return(geometric_loss, geo_opt)

def reject_POSCAR(folder_path):
	i = 0 ; dist_min=[];
	for element in sorted(os.listdir(folder_path)):
	    elementx = os.path.join(folder_path, element)
	    struct = elementx                                                                                                                                         
	    p = Poscar.from_file(struct)
	    radius = 4.
	    Natoms = len(p.structure)
	    s = mg.Structure.from_file(struct)
	    allneig = s.get_all_neighbors(radius)
	    distance = []
	    for i in range(len(allneig)):
	        for j in range(len(allneig[i])):
	            distance.append(allneig[i][j][1])
	    mini = min(distance)
	    dist_min.append(mini)
	    if mini>1.8:
	    	print(elementx)
	    else:
	    	os.remove(elementx)

