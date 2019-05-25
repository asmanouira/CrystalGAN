# !/usr/local/lib/python2.7 python 
# -*- coding=utf-8 -*-  


def clean_data(generated_data):
# Removing useless informations in the generated data
# Removing padded zeros
# in order to obtain POSCAR format

	for ip in range(4):
		for it in range(len(generated_data)):
			for t in range(len(generated_data[it][ip])-1):
			    if ((generated_data[it][ip][t][0] == generated_data[it][ip][t+1][0]) and 
			    	(generated_data[it][ip][t][1] == generated_data[it][ip][t+1][1]) and 
			    	(generated_data[it][ip][t][2] == generated_data[it][ip][t+1][2])):
			        
			        generated_data[it][ip] = generated_data[it][ip][0:t]
			        break
			    else:
			        continue
	return generated_data


def write_POSCAR(folder, generated_data):
	# Prints generated data
	# in POSCAR format
	for ix in range(len(generated_data)):
	   file = open(folder + "POSCAR" + str(ix), "a")
	   file.write('H M1 M2' + '\n') # Please change the atoms  "M1" and "M2" according to the chemical elements included in the input POSCAR file
	   file.write(str(1.0) + '\n')
	   # Latice vectors
	   for (a,b,c) in generated_data[ix][0]:
	       file.write('   '+ str("%.8f" % a)+'   '+str("%.8f" % b)+'   '+str("%.8f" % c)+ '\n')
	   file.write('H M1 M2' + '\n') # Please change the atoms names M1 and M2 
	   # Number of atoms per chemical element 
	   file.write(str(len(generated_data[ix][1]))+ '  ' +str(len(generated_data[ix][2]))+'  '+str(len(generated_data[ix][3])) +'\n')
	   # Positions in cartesian coordinates
	   file.write('Cartesian' + '\n')
	   # Positions of atom H
	   for (x_h,y_h,z_h) in generated_data[ix][1]:
	       file.write('   ' + str("%.8f" % x_h) + '   ' + str("%.8f" % y_h) + '   '+ str("%.8f" % z_h) + '\n')
	   # Positions of atom M1 
	   for (x_m1,y_m1,z_m1) in generated_data[ix][2]:
	       file.write('   ' + str("%.8f" % x_m1) + '   ' + str("%.8f" % y_m1) + '   '+ str("%.8f" % z_m1) + '\n')
	   # Positions of atom M2
	   for (x_m2,y_m2,z_m2) in generated_data[ix][3]:
	       file.write('   ' + str("%.8f" % x_m2) + '   ' + str("%.8f" % y_m2) + '   '+ str("%.8f" % z_m2) + '\n')
	   file.close()