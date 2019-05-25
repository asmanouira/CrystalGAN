#!/usr/local/lib/python2.7 python 
# -*- coding=utf-8 -*-

import sys
import pymatgen as mg
from pymatgen.io.vasp.inputs import Poscar
from math import fabs


def all_neighbors(radius=4., struct="POSCAR"):
    th = 0.001

    # get all neighbors
    s = mg.Structure.from_file(struct)
    allneig = s.get_all_neighbors(radius)

    iat = 0
    for site, neiglist in zip(s, allneig):
        iat += 1

        print  (site.specie.symbol,iat)
        neiglist.sort(key=lambda x: x[1])
        uneq = list()
        for neig in neiglist:
            site, d = neig
            if len(uneq) == 0:
                uneq.append((site, d, 1))
		
            else:
                new = True
                for i, u in enumerate(uneq):
                    usite, ud, n = u
                    if usite.specie.symbol == site.specie.symbol and \
                            fabs(d - ud) < th:
                        new = False
                        n += 1
                        uneq[i] = (usite, ud, n)
                        break
                if new:
                    uneq.append((site, d, 1))
                    for neig in uneq:
                    	site, d, n = neig
                    	print  ("    %3s %10.4f %2d " % (site.specie.symbol,d, n))

