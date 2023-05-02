# Importing libraries
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re
import os
import cv2

######################## Calculate Matrix ########################################

#Data
df_succ=pd.read_csv("../Results/GoodShoot.csv",sep = ',')
df_fail=pd.read_csv("../Results/BadShoot.csv",sep = ',')

#L_ankle
L_ankle_succ=df_succ["L_Ankle"]
L_ankle_succ
L_ankle_succ.to_numpy()
x_succ=[]
y_succ=[]

# initializing substrings
sub1 = "("
sub2 = ","
sub3 =")"
 
s1=str(re.escape(sub1))
s2=str(re.escape(sub2))
s3=str(re.escape(sub3))

for row in L_ankle_succ :
  x_succ.append(float(re.findall(s1+"(.*)"+s2,row)[0]))
  y_succ.append(float(re.findall(s2+"(.*)"+s3,row)[0]))
print(len(x_succ))

L_ankle_fail=df_fail["L_Ankle"]
L_ankle_fail.to_numpy()
x_fail=[]
y_fail=[]


for row in L_ankle_fail :
  x_fail.append(float(re.findall(s1+"(.*)"+s2,row)[0]))
  y_fail.append(float(re.findall(s2+"(.*)"+s3,row)[0]))

L_Ankle_x = []
for i in range(40):
    L_Ankle_x.append(x_succ[i] - x_fail[i])

L_Ankle_y = []
for i in range(40):
    L_Ankle_y.append(y_succ[i] - y_fail[i])

L_Ankle_x= [x*20 for x in L_Ankle_x]
L_Ankle_y = [y*20 for y in L_Ankle_y]
L_Ankle_x = [round(x, 2) for x in L_Ankle_x]
L_Ankle_y= [round(y, 2) for y in L_Ankle_y]

L_Ankle = []
for i in range(40):    
    my_tuple = (L_Ankle_x[i], L_Ankle_y[i])
    L_Ankle.append(my_tuple)




#L_Elbow

L_Elbow_succ=df_succ["L_Elbow"]
L_Elbow_succ.to_numpy()
x_succ=[]
y_succ=[]

# initializing substrings
sub1 = "("
sub2 = ","
sub3 =")"
 
s1=str(re.escape(sub1))
s2=str(re.escape(sub2))
s3=str(re.escape(sub3))

for row in L_Elbow_succ :
  x_succ.append(float(re.findall(s1+"(.*)"+s2,row)[0]))
  y_succ.append(float(re.findall(s2+"(.*)"+s3,row)[0]))

L_Elbow_fail=df_fail["L_Elbow"]
L_Elbow_fail.to_numpy()
x_fail=[]
y_fail=[]


for row in L_Elbow_fail :
  x_fail.append(float(re.findall(s1+"(.*)"+s2,row)[0]))
  y_fail.append(float(re.findall(s2+"(.*)"+s3,row)[0]))

L_Elbow_x = []
for i in range(40):
    L_Elbow_x.append(x_succ[i] - x_fail[i])

L_Elbow_y = []
for i in range(40):
    L_Elbow_y.append(y_succ[i] - y_fail[i])

L_Elbow_x= [x*20 for x in L_Elbow_x]
L_Elbow_y = [y*20 for y in L_Elbow_y]
L_Elbow_x = [round(x, 2) for x in L_Elbow_x]
L_Elbow_y= [round(y, 2) for y in L_Elbow_y]

L_Elbow = []
for i in range(40):
    my_tuple = (L_Elbow_x[i], L_Elbow_y[i])
    L_Elbow.append(my_tuple)

#L_Hip

L_Hip_succ=df_succ["L_Hip"]
L_Hip_succ.to_numpy()
x_succ=[]
y_succ=[]

# initializing substrings
sub1 = "("
sub2 = ","
sub3 =")"
 
s1=str(re.escape(sub1))
s2=str(re.escape(sub2))
s3=str(re.escape(sub3))

for row in L_Hip_succ :
  x_succ.append(float(re.findall(s1+"(.*)"+s2,row)[0]))
  y_succ.append(float(re.findall(s2+"(.*)"+s3,row)[0]))


L_Hip_fail=df_fail["L_Hip"]
L_Hip_fail.to_numpy()
x_fail=[]
y_fail=[]


for row in L_Hip_fail :
  x_fail.append(float(re.findall(s1+"(.*)"+s2,row)[0]))
  y_fail.append(float(re.findall(s2+"(.*)"+s3,row)[0]))

L_Hip_x = []
for i in range(40):
    L_Hip_x.append(x_succ[i] - x_fail[i])

L_Hip_y = []
for i in range(40):
    L_Hip_y.append(y_succ[i] - y_fail[i])

L_Hip_x= [x*20 for x in L_Hip_x]
L_Hip_y = [y*20 for y in L_Hip_y]
L_Hip_x = [round(x, 2) for x in L_Hip_x]
L_Hip_y= [round(y, 2) for y in L_Hip_y]

L_Hip = []
for i in range(len(L_Hip_x)):
    my_tuple = (L_Hip_x[i], L_Hip_y[i])
    L_Hip.append(my_tuple)
  


#L_Knee

L_Knee_succ=df_succ["L_knee"]
L_Knee_succ.to_numpy()
x_succ=[]
y_succ=[]

# initializing substrings
sub1 = "("
sub2 = ","
sub3 =")"
 
s1=str(re.escape(sub1))
s2=str(re.escape(sub2))
s3=str(re.escape(sub3))

for row in L_Knee_succ :
  x_succ.append(float(re.findall(s1+"(.*)"+s2,row)[0]))
  y_succ.append(float(re.findall(s2+"(.*)"+s3,row)[0]))

L_Knee_fail=df_fail["L_knee"]
L_Knee_fail.to_numpy()
x_fail=[]
y_fail=[]


for row in L_Knee_fail :
  x_fail.append(float(re.findall(s1+"(.*)"+s2,row)[0]))
  y_fail.append(float(re.findall(s2+"(.*)"+s3,row)[0]))

L_Knee_x = []
for i in range(40):
    L_Knee_x.append(x_succ[i] - x_fail[i])

L_Knee_y = []
for i in range(40):
    L_Knee_y.append(y_succ[i] - y_fail[i])

L_Knee_x= [x*20 for x in L_Knee_x]
L_Knee_y = [y*20 for y in L_Knee_y]
L_Knee_x = [round(x, 2) for x in L_Knee_x]
L_Knee_y= [round(y, 2) for y in L_Knee_y]

L_Knee = []
for i in range(40):
    my_tuple = (L_Knee_x[i], L_Knee_y[i])
    L_Knee.append(my_tuple)


#L_Sholder

L_Sholder_succ=df_succ["L_Sholder"]
L_Sholder_succ.to_numpy()
x_succ=[]
y_succ=[]

# initializing substrings
sub1 = "("
sub2 = ","
sub3 =")"
 
s1=str(re.escape(sub1))
s2=str(re.escape(sub2))
s3=str(re.escape(sub3))

for row in L_Sholder_succ :
  x_succ.append(float(re.findall(s1+"(.*)"+s2,row)[0]))
  y_succ.append(float(re.findall(s2+"(.*)"+s3,row)[0]))


L_Sholder_fail=df_fail["L_Sholder"]
L_Sholder_fail.to_numpy()
x_fail=[]
y_fail=[]


for row in L_Sholder_fail :
  x_fail.append(float(re.findall(s1+"(.*)"+s2,row)[0]))
  y_fail.append(float(re.findall(s2+"(.*)"+s3,row)[0]))

L_Sholder_x = []
for i in range(40):
    L_Sholder_x.append(x_succ[i] - x_fail[i])

L_Sholder_y = []
for i in range(40):
    L_Sholder_y.append(y_succ[i] - y_fail[i])

L_Sholder_x= [x*20 for x in L_Sholder_x]
L_Sholder_y = [y*20 for y in L_Sholder_y]
L_Sholder_x = [round(x, 2) for x in L_Sholder_x]
L_Sholder_y = [round(y, 2) for y in L_Sholder_y]

L_Sholder = []
for i in range(40):
    my_tuple = (L_Sholder_x[i], L_Sholder_y[i])
    L_Sholder.append(my_tuple)


#L_Wrest

L_Wrest_succ=df_succ["L_Wrest"]
L_Wrest_succ.to_numpy()
x_succ=[]
y_succ=[]

# initializing substrings
sub1 = "("
sub2 = ","
sub3 =")"
 
s1=str(re.escape(sub1))
s2=str(re.escape(sub2))
s3=str(re.escape(sub3))

for row in L_Wrest_succ :
  x_succ.append(float(re.findall(s1+"(.*)"+s2,row)[0]))
  y_succ.append(float(re.findall(s2+"(.*)"+s3,row)[0]))


L_Wrest_fail=df_fail["L_Wrest"]
L_Wrest_fail.to_numpy()
x_fail=[]
y_fail=[]


for row in L_Wrest_fail :
  x_fail.append(float(re.findall(s1+"(.*)"+s2,row)[0]))
  y_fail.append(float(re.findall(s2+"(.*)"+s3,row)[0]))

L_Wrest_x = []

for i in range(40):
    L_Wrest_x.append(x_succ[i] - x_fail[i])

L_Wrest_y = []
for i in range(40):
    L_Wrest_y.append(y_succ[i] - y_fail[i])

L_Wrest_x= [x*20 for x in L_Wrest_x]
L_Wrest_y = [y*20 for y in L_Wrest_y]
L_Wrest_x = [round(x, 2) for x in L_Wrest_x]
L_Wrest_y = [round(y, 2) for y in L_Wrest_y]

L_Wrest= []
for i in range(40):
    my_tuple = (L_Wrest_x[i], L_Wrest_y[i])
    L_Wrest.append(my_tuple)

#neck

neck_succ=df_succ["Neck"]
neck_succ.to_numpy()
x_succ=[]
y_succ=[]

# initializing substrings
sub1 = "("
sub2 = ","
sub3 =")"
 
s1=str(re.escape(sub1))
s2=str(re.escape(sub2))
s3=str(re.escape(sub3))

for row in neck_succ :
  x_succ.append(float(re.findall(s1+"(.*)"+s2,row)[0]))
  y_succ.append(float(re.findall(s2+"(.*)"+s3,row)[0]))

neck_fail=df_fail["Neck"]
neck_fail.to_numpy()
x_fail=[]
y_fail=[]


for row in neck_fail :
  x_fail.append(float(re.findall(s1+"(.*)"+s2,row)[0]))
  y_fail.append(float(re.findall(s2+"(.*)"+s3,row)[0]))

Neck_x = []

for i in range(40):
    Neck_x.append(x_succ[i] - x_fail[i])

Neck_y = []
for i in range(40):
    Neck_y.append(y_succ[i] - y_fail[i])

Neck_x= [x*20 for x in Neck_x]
Neck_y = [y*20 for y in Neck_y]
Neck_x = [round(x, 2) for x in Neck_x]
Neck_y = [round(y, 2) for y in Neck_y]

Neck = []
for i in range(40):
    my_tuple = (Neck_x[i], Neck_y[i])
    Neck.append(my_tuple)


#Nose

Nose_succ=df_succ["Noze"]
Nose_succ.to_numpy()
x_succ=[]
y_succ=[]

# initializing substrings
sub1 = "("
sub2 = ","
sub3 =")"
 
s1=str(re.escape(sub1))
s2=str(re.escape(sub2))
s3=str(re.escape(sub3))

for row in Nose_succ :
  x_succ.append(float(re.findall(s1+"(.*)"+s2,row)[0]))
  y_succ.append(float(re.findall(s2+"(.*)"+s3,row)[0]))

Nose_fail=df_fail["Noze"]
Nose_fail.to_numpy()
x_fail=[]
y_fail=[]


for row in Nose_fail :
  x_fail.append(float(re.findall(s1+"(.*)"+s2,row)[0]))
  y_fail.append(float(re.findall(s2+"(.*)"+s3,row)[0]))

Nose_x = []
for i in range(40):
    Nose_x.append(x_succ[i] - x_fail[i])

Nose_y = []
for i in range(40):
    Nose_y.append(y_succ[i] - y_fail[i])

Nose_x= [x*20 for x in Nose_x]
Nose_y = [y*20 for y in Nose_y]
Nose_x = [round(x, 2) for x in Nose_x]
Nose_y = [round(y, 2) for y in Nose_y]

Nose = []
for i in range(40):
    my_tuple = (Nose_x[i], Nose_y[i])
    Nose.append(my_tuple)



#R_Ankle

R_Ankle_succ=df_succ["Noze"]
R_Ankle_succ.to_numpy()
x_succ=[]
y_succ=[]

# initializing substrings
sub1 = "("
sub2 = ","
sub3 =")"
 
s1=str(re.escape(sub1))
s2=str(re.escape(sub2))
s3=str(re.escape(sub3))

for row in R_Ankle_succ :
  x_succ.append(float(re.findall(s1+"(.*)"+s2,row)[0]))
  y_succ.append(float(re.findall(s2+"(.*)"+s3,row)[0]))


R_Ankle_fail=df_fail["R_Ankle"]
R_Ankle_fail.to_numpy()
x_fail=[]
y_fail=[]


for row in R_Ankle_fail :
  x_fail.append(float(re.findall(s1+"(.*)"+s2,row)[0]))
  y_fail.append(float(re.findall(s2+"(.*)"+s3,row)[0]))

R_Ankle_x = []

for i in range(40):
    R_Ankle_x.append(x_succ[i] - x_fail[i])

R_Ankle_y = []
for i in range(40):
    R_Ankle_y.append(y_succ[i] - y_fail[i])

R_Ankle_x= [x*20 for x in R_Ankle_x]
R_Ankle_y = [y*20 for y in R_Ankle_y]
R_Ankle_x = [round(x, 2) for x in R_Ankle_x]
R_Ankle_y = [round(y, 2) for y in R_Ankle_y]

R_Ankle = []
for i in range(40):
    my_tuple = (R_Ankle_x[i], R_Ankle_y[i])
    R_Ankle.append(my_tuple)


#R_Elbow

R_Elbow_succ=df_succ["R_Elbow"]
R_Elbow_succ.to_numpy()
x_succ=[]
y_succ=[]

# initializing substrings
sub1 = "("
sub2 = ","
sub3 =")"
 
s1=str(re.escape(sub1))
s2=str(re.escape(sub2))
s3=str(re.escape(sub3))

for row in R_Elbow_succ :
  x_succ.append(float(re.findall(s1+"(.*)"+s2,row)[0]))
  y_succ.append(float(re.findall(s2+"(.*)"+s3,row)[0]))

R_Elbow_fail=df_fail["R_Elbow"]
R_Elbow_fail.to_numpy()
x_fail=[]
y_fail=[]


for row in R_Elbow_fail :
  x_fail.append(float(re.findall(s1+"(.*)"+s2,row)[0]))
  y_fail.append(float(re.findall(s2+"(.*)"+s3,row)[0]))

R_Elbow_x= []

for i in range(40):
    R_Elbow_x.append(x_succ[i] - x_fail[i])

R_Elbow_y = []
for i in range(40):
    R_Elbow_y.append(y_succ[i] - y_fail[i])

R_Elbow_x= [x*20 for x in R_Elbow_x]
R_Elbow_y = [y*20 for y in R_Elbow_y]
R_Elbow_x= [round(x, 2) for x in R_Elbow_x]
R_Elbow_y = [round(y, 2) for y in R_Elbow_y]

R_Elbow= []
for i in range(40):
    my_tuple = (R_Elbow_x[i], R_Elbow_y[i])
    R_Elbow.append(my_tuple)

#R_Hip

R_Hip_succ=df_succ["R_Hip"]
R_Hip_succ.to_numpy()
x_succ=[]
y_succ=[]

# initializing substrings
sub1 = "("
sub2 = ","
sub3 =")"
 
s1=str(re.escape(sub1))
s2=str(re.escape(sub2))
s3=str(re.escape(sub3))

for row in R_Hip_succ :
  x_succ.append(float(re.findall(s1+"(.*)"+s2,row)[0]))
  y_succ.append(float(re.findall(s2+"(.*)"+s3,row)[0]))

R_Hip_fail=df_fail["R_Hip"]
R_Hip_fail.to_numpy()
x_fail=[]
y_fail=[]


for row in R_Hip_fail :
  x_fail.append(float(re.findall(s1+"(.*)"+s2,row)[0]))
  y_fail.append(float(re.findall(s2+"(.*)"+s3,row)[0]))

R_Hip_x= []

for i in range(40):
    R_Hip_x.append(x_succ[i] - x_fail[i])

R_Hip_y = []
for i in range(40):
    R_Hip_y.append(y_succ[i] - y_fail[i])

R_Hip_x= [x*20 for x in R_Hip_x]
R_Hip_y = [y*20 for y in R_Hip_y]
R_Hip_x= [round(x, 2) for x in R_Hip_x]
R_Hip_y = [round(y, 2) for y in R_Hip_y]

R_Hip= []
for i in range(40):
    my_tuple = (R_Hip_x[i], R_Hip_y[i])
    R_Hip.append(my_tuple)

#R_Knee

R_Knee_succ=df_succ["R_Knee"]
R_Knee_succ.to_numpy()
x_succ=[]
y_succ=[]


# initializing substrings
sub1 = "("
sub2 = ","
sub3 =")"
 
s1=str(re.escape(sub1))
s2=str(re.escape(sub2))
s3=str(re.escape(sub3))

for row in R_Knee_succ :
  x_succ.append(float(re.findall(s1+"(.*)"+s2,row)[0]))
  y_succ.append(float(re.findall(s2+"(.*)"+s3,row)[0]))

R_Knee_fail=df_fail["R_Knee"]
R_Knee_fail.to_numpy()
x_fail=[]
y_fail=[]


for row in R_Knee_fail :
  x_fail.append(float(re.findall(s1+"(.*)"+s2,row)[0]))
  y_fail.append(float(re.findall(s2+"(.*)"+s3,row)[0]))

R_Knee_x = []
for i in range(40):
    R_Knee_x.append(x_succ[i] - x_fail[i])

R_Knee_y = []
for i in range(40):
    R_Knee_y.append(y_succ[i] - y_fail[i])

R_Knee_x= [x*20 for x in R_Knee_x]
R_Knee_y = [y*20 for y in R_Knee_y]
R_Knee_x = [round(x, 2) for x in R_Knee_x]
R_Knee_y = [round(y, 2) for y in R_Knee_y]

R_Knee = []
for i in range(40):
    my_tuple = (R_Knee_x[i], R_Knee_y[i])
    R_Knee.append(my_tuple)

#R_Sholder

R_Sholder_succ=df_succ["R_Sholder"]
R_Sholder_succ.to_numpy()
x_succ=[]
y_succ=[]

# initializing substrings
sub1 = "("
sub2 = ","
sub3 =")"
 
s1=str(re.escape(sub1))
s2=str(re.escape(sub2))
s3=str(re.escape(sub3))

for row in R_Sholder_succ :
  x_succ.append(float(re.findall(s1+"(.*)"+s2,row)[0]))
  y_succ.append(float(re.findall(s2+"(.*)"+s3,row)[0]))


R_Sholder_fail=df_fail["R_Sholder"]
R_Sholder_fail.to_numpy()
x_fail=[]
y_fail=[]


for row in R_Sholder_fail :
  x_fail.append(float(re.findall(s1+"(.*)"+s2,row)[0]))
  y_fail.append(float(re.findall(s2+"(.*)"+s3,row)[0]))

R_Sholder_x= []
for i in range(40):
    R_Sholder_x.append(x_succ[i] - x_fail[i])

R_Sholder_y = []
for i in range(40):
    R_Sholder_y.append(y_succ[i] - y_fail[i])

R_Sholder_x= [x*20 for x in R_Sholder_x]
R_Sholder_y = [y*20 for y in R_Sholder_y]
R_Sholder_x= [round(x, 2) for x in R_Sholder_x]
R_Sholder_y = [round(y, 2) for y in R_Sholder_y]

R_Sholder= []
for i in range(40):
    my_tuple = (R_Sholder_x[i], R_Sholder_y[i])
    R_Sholder.append(my_tuple)



# R_wrest

R_Wrest_succ=df_succ["R_Wrest"]
R_Wrest_succ.to_numpy()
x_succ=[]
y_succ=[]

# initializing substrings
sub1 = "("
sub2 = ","
sub3 =")"
 
s1=str(re.escape(sub1))
s2=str(re.escape(sub2))
s3=str(re.escape(sub3))

for row in R_Wrest_succ :
  x_succ.append(float(re.findall(s1+"(.*)"+s2,row)[0]))
  y_succ.append(float(re.findall(s2+"(.*)"+s3,row)[0]))


R_Wrest_fail=df_fail["R_Wrest"]
R_Wrest_fail.to_numpy()
x_fail=[]
y_fail=[]


for row in R_Wrest_fail :
  x_fail.append(float(re.findall(s1+"(.*)"+s2,row)[0]))
  y_fail.append(float(re.findall(s2+"(.*)"+s3,row)[0]))

R_Wrest_x= []
for i in range(40):
    R_Wrest_x.append(x_succ[i] - x_fail[i])

R_Wrest_y = []
for i in range(40):
    R_Wrest_y.append(y_succ[i] - y_fail[i])

R_Wrest_x= [x*20 for x in R_Wrest_x]
R_Wrest_y = [y*20 for y in R_Wrest_y]
R_Wrest_x= [round(x, 2) for x in R_Wrest_x]
R_Wrest_y = [round(y, 2) for y in R_Wrest_y]

R_Wrest= []
for i in range(40):
    my_tuple = (R_Wrest_x[i], R_Wrest_y[i])
    R_Wrest.append(my_tuple)

matrix = np.empty((14, 40), dtype=tuple)

matrix[0] = L_Ankle
matrix[1] = L_Elbow
matrix[2] = L_Hip
matrix[3] = L_Knee
matrix[4] = L_Sholder
matrix[5] = L_Wrest
matrix[6] = Neck
matrix[7] = Nose
matrix[8] = R_Ankle
matrix[9] = R_Elbow
matrix[10] = R_Hip
matrix[11] = R_Knee
matrix[12] = R_Sholder
matrix[13] = R_Wrest

print(matrix)

########################################### Draw vectors ##########################################

filenames =[]
for filename in os.listdir('./failed_frames/') :
    if filename.endswith('.jpg') :
       filenames.append(filename)



# Define the fixed length of the vectors
vec_length = 20
df_fail=pd.read_csv("../Results/BadShoot.csv",sep = ',')   

#L_Ankle
L_Ankle_fail=df_fail["L_Ankle"]
L_Ankle_fail.to_numpy()
x_fail=[]
y_fail=[]

sub1 = "("
sub2 = ","
sub3 =")"
 
s1=str(re.escape(sub1))
s2=str(re.escape(sub2))
s3=str(re.escape(sub3))

for row in L_Ankle_fail :
  x_fail.append(float(re.findall(s1+"(.*)"+s2,row)[0]))
  y_fail.append(float(re.findall(s2+"(.*)"+s3,row)[0]))
L_Ankle_start = []
for i in range(40):    
    my_tuple = (x_fail[i], y_fail[i])
    L_Ankle_start.append(my_tuple)


# Define the starting positions and directions of the vectors
start_pos = np.array(L_Ankle_start)
vec_dir = np.array(matrix[0].tolist())

# Define the fixed length of the vectors
vec_length = 20

# Calculate the endpoint positions of the vectors
end_pos = start_pos + vec_length * vec_dir
# Draw the vectors on the image
img = cv2.imread('./failed_frames/frame_correction1.jpg')
# Draw the vectors on the image         
cv2.arrowedLine(img, tuple(start_pos[26].astype(int)), tuple(end_pos[26].astype(int)), (0, 0, 255), thickness=2)
cv2.imwrite(os.path.join('./failed_frames/frame_correction1.jpg'), img)

img = cv2.imread('./failed_frames/frame_correction2.jpg')
# Draw the vectors on the image         
cv2.arrowedLine(img, tuple(start_pos[24].astype(int)), tuple(end_pos[24].astype(int)), (0, 0, 255), thickness=2)
cv2.imwrite(os.path.join('./failed_frames/frame_correction2.jpg'), img)

#L_Elbow
L_Elbow_fail=df_fail["L_Elbow"]
L_Elbow_fail.to_numpy()
x_fail=[]
y_fail=[]

sub1 = "("
sub2 = ","
sub3 =")"
 
s1=str(re.escape(sub1))
s2=str(re.escape(sub2))
s3=str(re.escape(sub3))

for row in L_Elbow_fail :
  x_fail.append(float(re.findall(s1+"(.*)"+s2,row)[0]))
  y_fail.append(float(re.findall(s2+"(.*)"+s3,row)[0]))
L_Elbow_start = []
for i in range(40):    
    my_tuple = (x_fail[i], y_fail[i])
    L_Elbow_start.append(my_tuple)

# Define the starting positions and directions of the vectors
start_pos = np.array(L_Elbow_start)
vec_dir = np.array(matrix[1].tolist())

# Define the fixed length of the vectors
vec_length = 20

# Calculate the endpoint positions of the vectors
end_pos = start_pos + vec_length * vec_dir
# Draw the vectors on the image
img = cv2.imread('./failed_frames/frame_correction1.jpg')
# Draw the vectors on the image         
cv2.arrowedLine(img, tuple(start_pos[26].astype(int)), tuple(end_pos[26].astype(int)), (0, 0, 255), thickness=2)
cv2.imwrite(os.path.join('./failed_frames/frame_correction1.jpg'), img)

img = cv2.imread('./failed_frames/frame_correction2.jpg')
# Draw the vectors on the image         
cv2.arrowedLine(img, tuple(start_pos[24].astype(int)), tuple(end_pos[24].astype(int)), (0, 0, 255), thickness=2)
cv2.imwrite(os.path.join('./failed_frames/frame_correction2.jpg'), img)


#L_Hip
L_Hip_fail=df_fail["L_Hip"]
L_Hip_fail.to_numpy()
x_fail=[]
y_fail=[]

sub1 = "("
sub2 = ","
sub3 =")"
 
s1=str(re.escape(sub1))
s2=str(re.escape(sub2))
s3=str(re.escape(sub3))

for row in L_Hip_fail :
  x_fail.append(float(re.findall(s1+"(.*)"+s2,row)[0]))
  y_fail.append(float(re.findall(s2+"(.*)"+s3,row)[0]))
L_Hip_start = []
for i in range(40):    
    my_tuple = (x_fail[i], y_fail[i])
    L_Hip_start.append(my_tuple)


# Define the starting positions and directions of the vectors
start_pos = np.array(L_Hip_start)
vec_dir = np.array(matrix[2].tolist())

# Define the fixed length of the vectors
vec_length = 20

# Calculate the endpoint positions of the vectors
end_pos = start_pos + vec_length * vec_dir
# Draw the vectors on the image
img = cv2.imread('./failed_frames/frame_correction1.jpg')
# Draw the vectors on the image         
cv2.arrowedLine(img, tuple(start_pos[26].astype(int)), tuple(end_pos[26].astype(int)), (0, 0, 255), thickness=2)
cv2.imwrite(os.path.join('./failed_frames/frame_correction1.jpg'), img)

img = cv2.imread('./failed_frames/frame_correction2.jpg')
# Draw the vectors on the image         
cv2.arrowedLine(img, tuple(start_pos[24].astype(int)), tuple(end_pos[24].astype(int)), (0, 0, 255), thickness=2)
cv2.imwrite(os.path.join('./failed_frames/frame_correction2.jpg'), img)


#L_Knee
L_Knee_fail=df_fail["L_knee"]
L_Knee_fail.to_numpy()
x_fail=[]
y_fail=[]

sub1 = "("
sub2 = ","
sub3 =")"
 
s1=str(re.escape(sub1))
s2=str(re.escape(sub2))
s3=str(re.escape(sub3))

for row in L_Knee_fail :
  x_fail.append(float(re.findall(s1+"(.*)"+s2,row)[0]))
  y_fail.append(float(re.findall(s2+"(.*)"+s3,row)[0]))
L_Knee_start = []
for i in range(40):    
    my_tuple = (x_fail[i], y_fail[i])
    L_Knee_start.append(my_tuple)


# Define the starting positions and directions of the vectors
start_pos = np.array(L_Knee_start)
vec_dir = np.array(matrix[3].tolist())

# Define the fixed length of the vectors
vec_length = 20

# Calculate the endpoint positions of the vectors
end_pos = start_pos + vec_length * vec_dir
# Draw the vectors on the image
img = cv2.imread('./failed_frames/frame_correction1.jpg')
# Draw the vectors on the image         
cv2.arrowedLine(img, tuple(start_pos[26].astype(int)), tuple(end_pos[26].astype(int)), (0, 0, 255), thickness=2)
cv2.imwrite(os.path.join('./failed_frames/frame_correction1.jpg'), img)

img = cv2.imread('./failed_frames/frame_correction2.jpg')
# Draw the vectors on the image         
cv2.arrowedLine(img, tuple(start_pos[24].astype(int)), tuple(end_pos[24].astype(int)), (0, 0, 255), thickness=2)
cv2.imwrite(os.path.join('./failed_frames/frame_correction2.jpg'), img)
        

#L_Sholder
L_Sholder_fail=df_fail["L_Sholder"]
L_Sholder_fail.to_numpy()
x_fail=[]
y_fail=[]

sub1 = "("
sub2 = ","
sub3 =")"
 
s1=str(re.escape(sub1))
s2=str(re.escape(sub2))
s3=str(re.escape(sub3))

for row in L_Sholder_fail :
  x_fail.append(float(re.findall(s1+"(.*)"+s2,row)[0]))
  y_fail.append(float(re.findall(s2+"(.*)"+s3,row)[0]))
L_Sholder_start = []
for i in range(40):    
    my_tuple = (x_fail[i], y_fail[i])
    L_Sholder_start.append(my_tuple)


# Define the starting positions and directions of the vectors
start_pos = np.array(L_Sholder_start)
vec_dir = np.array(matrix[4].tolist())

# Define the fixed length of the vectors
vec_length = 20

# Calculate the endpoint positions of the vectors
end_pos = start_pos + vec_length * vec_dir
# Draw the vectors on the image
img = cv2.imread('./failed_frames/frame_correction1.jpg')
# Draw the vectors on the image         
cv2.arrowedLine(img, tuple(start_pos[26].astype(int)), tuple(end_pos[26].astype(int)), (0, 0, 255), thickness=2)
cv2.imwrite(os.path.join('./failed_frames/frame_correction1.jpg'), img)

img = cv2.imread('./failed_frames/frame_correction2.jpg')
# Draw the vectors on the image         
cv2.arrowedLine(img, tuple(start_pos[24].astype(int)), tuple(end_pos[24].astype(int)), (0, 0, 255), thickness=2)
cv2.imwrite(os.path.join('./failed_frames/frame_correction2.jpg'), img)
 
#L_Wrest
L_Wrest_fail=df_fail["L_Wrest"]
L_Wrest_fail.to_numpy()
x_fail=[]
y_fail=[]

sub1 = "("
sub2 = ","
sub3 =")"
 
s1=str(re.escape(sub1))
s2=str(re.escape(sub2))
s3=str(re.escape(sub3))

for row in L_Wrest_fail :
  x_fail.append(float(re.findall(s1+"(.*)"+s2,row)[0]))
  y_fail.append(float(re.findall(s2+"(.*)"+s3,row)[0]))
L_Wrest_start = []
for i in range(40):    
    my_tuple = (x_fail[i], y_fail[i])
    L_Wrest_start.append(my_tuple)


# Define the starting positions and directions of the vectors
start_pos = np.array(L_Wrest_start)
vec_dir = np.array(matrix[5].tolist())

# Define the fixed length of the vectors
vec_length = 20

# Calculate the endpoint positions of the vectors
end_pos = start_pos + vec_length * vec_dir
# Draw the vectors on the image
img = cv2.imread('./failed_frames/frame_correction1.jpg')
# Draw the vectors on the image         
cv2.arrowedLine(img, tuple(start_pos[26].astype(int)), tuple(end_pos[26].astype(int)), (0, 0, 255), thickness=2)
cv2.imwrite(os.path.join('./failed_frames/frame_correction1.jpg'), img)

img = cv2.imread('./failed_frames/frame_correction2.jpg')
# Draw the vectors on the image         
cv2.arrowedLine(img, tuple(start_pos[24].astype(int)), tuple(end_pos[24].astype(int)), (0, 0, 255), thickness=2)
cv2.imwrite(os.path.join('./failed_frames/frame_correction2.jpg'), img)

#Neck
Neck_fail=df_fail["Neck"]
Neck_fail.to_numpy()
x_fail=[]
y_fail=[]

sub1 = "("
sub2 = ","
sub3 =")"
 
s1=str(re.escape(sub1))
s2=str(re.escape(sub2))
s3=str(re.escape(sub3))

for row in Neck_fail :
  x_fail.append(float(re.findall(s1+"(.*)"+s2,row)[0]))
  y_fail.append(float(re.findall(s2+"(.*)"+s3,row)[0]))
Neck_start = []
for i in range(40):    
    my_tuple = (x_fail[i], y_fail[i])
    Neck_start.append(my_tuple)


# Define the starting positions and directions of the vectors
start_pos = np.array(Neck_start)
vec_dir = np.array(matrix[6].tolist())

# Define the fixed length of the vectors
vec_length = 20

# Calculate the endpoint positions of the vectors
end_pos = start_pos + vec_length * vec_dir
# Draw the vectors on the image
img = cv2.imread('./failed_frames/frame_correction1.jpg')
# Draw the vectors on the image         
cv2.arrowedLine(img, tuple(start_pos[26].astype(int)), tuple(end_pos[26].astype(int)), (0, 0, 255), thickness=2)
cv2.imwrite(os.path.join('./failed_frames/frame_correction1.jpg'), img)

img = cv2.imread('./failed_frames/frame_correction2.jpg')
# Draw the vectors on the image         
cv2.arrowedLine(img, tuple(start_pos[24].astype(int)), tuple(end_pos[24].astype(int)), (0, 0, 255), thickness=2)
cv2.imwrite(os.path.join('./failed_frames/frame_correction2.jpg'), img)

#Nose
Nose_fail=df_fail["Noze"]
Nose_fail.to_numpy()
x_fail=[]
y_fail=[]

sub1 = "("
sub2 = ","
sub3 =")"
 
s1=str(re.escape(sub1))
s2=str(re.escape(sub2))
s3=str(re.escape(sub3))

for row in Nose_fail :
  x_fail.append(float(re.findall(s1+"(.*)"+s2,row)[0]))
  y_fail.append(float(re.findall(s2+"(.*)"+s3,row)[0]))
Nose_start = []
for i in range(40):    
    my_tuple = (x_fail[i], y_fail[i])
    Nose_start.append(my_tuple)


# Define the starting positions and directions of the vectors
start_pos = np.array(Nose_start)
vec_dir = np.array(matrix[7].tolist())

# Define the fixed length of the vectors
vec_length = 20

# Calculate the endpoint positions of the vectors
end_pos = start_pos + vec_length * vec_dir
# Draw the vectors on the image
img = cv2.imread('./failed_frames/frame_correction1.jpg')
# Draw the vectors on the image         
cv2.arrowedLine(img, tuple(start_pos[26].astype(int)), tuple(end_pos[26].astype(int)), (0, 0, 255), thickness=2)
cv2.imwrite(os.path.join('./failed_frames/frame_correction1.jpg'), img)

img = cv2.imread('./failed_frames/frame_correction2.jpg')
# Draw the vectors on the image         
cv2.arrowedLine(img, tuple(start_pos[24].astype(int)), tuple(end_pos[24].astype(int)), (0, 0, 255), thickness=2)
cv2.imwrite(os.path.join('./failed_frames/frame_correction2.jpg'), img)

#R_Ankle
R_Ankle_fail=df_fail["R_Ankle"]
R_Ankle_fail.to_numpy()
x_fail=[]
y_fail=[]

sub1 = "("
sub2 = ","
sub3 =")"
 
s1=str(re.escape(sub1))
s2=str(re.escape(sub2))
s3=str(re.escape(sub3))

for row in R_Ankle_fail :
  x_fail.append(float(re.findall(s1+"(.*)"+s2,row)[0]))
  y_fail.append(float(re.findall(s2+"(.*)"+s3,row)[0]))
R_Ankle_start = []
for i in range(40):    
    my_tuple = (x_fail[i], y_fail[i])
    R_Ankle_start.append(my_tuple)


# Define the starting positions and directions of the vectors
start_pos = np.array(R_Ankle_start)
vec_dir = np.array(matrix[8].tolist())

# Define the fixed length of the vectors
vec_length = 20

# Calculate the endpoint positions of the vectors
end_pos = start_pos + vec_length * vec_dir
# Draw the vectors on the image
img = cv2.imread('./failed_frames/frame_correction1.jpg')
# Draw the vectors on the image         
cv2.arrowedLine(img, tuple(start_pos[26].astype(int)), tuple(end_pos[26].astype(int)), (0, 0, 255), thickness=2)
cv2.imwrite(os.path.join('./failed_frames/frame_correction1.jpg'), img)

img = cv2.imread('./failed_frames/frame_correction2.jpg')
# Draw the vectors on the image         
cv2.arrowedLine(img, tuple(start_pos[24].astype(int)), tuple(end_pos[24].astype(int)), (0, 0, 255), thickness=2)
cv2.imwrite(os.path.join('./failed_frames/frame_correction2.jpg'), img)


#R_Elbow
R_Elbow_fail=df_fail["R_Elbow"]
R_Elbow_fail.to_numpy()
x_fail=[]
y_fail=[]

sub1 = "("
sub2 = ","
sub3 =")"
 
s1=str(re.escape(sub1))
s2=str(re.escape(sub2))
s3=str(re.escape(sub3))

for row in R_Elbow_fail :
  x_fail.append(float(re.findall(s1+"(.*)"+s2,row)[0]))
  y_fail.append(float(re.findall(s2+"(.*)"+s3,row)[0]))
R_Elbow_start = []
for i in range(40):    
    my_tuple = (x_fail[i], y_fail[i])
    R_Elbow_start.append(my_tuple)


# Define the starting positions and directions of the vectors
start_pos = np.array(R_Elbow_start)
vec_dir = np.array(matrix[9].tolist())

# Define the fixed length of the vectors
vec_length = 20

# Calculate the endpoint positions of the vectors
end_pos = start_pos + vec_length * vec_dir
# Draw the vectors on the image
img = cv2.imread('./failed_frames/frame_correction1.jpg')
# Draw the vectors on the image         
cv2.arrowedLine(img, tuple(start_pos[26].astype(int)), tuple(end_pos[26].astype(int)), (0, 0, 255), thickness=2)
cv2.imwrite(os.path.join('./failed_frames/frame_correction1.jpg'), img)

img = cv2.imread('./failed_frames/frame_correction2.jpg')
# Draw the vectors on the image         
cv2.arrowedLine(img, tuple(start_pos[24].astype(int)), tuple(end_pos[24].astype(int)), (0, 0, 255), thickness=2)
cv2.imwrite(os.path.join('./failed_frames/frame_correction2.jpg'), img)


#R_Hip
R_Hip_fail=df_fail["R_Hip"]
R_Hip_fail.to_numpy()
x_fail=[]
y_fail=[]

sub1 = "("
sub2 = ","
sub3 =")"
 
s1=str(re.escape(sub1))
s2=str(re.escape(sub2))
s3=str(re.escape(sub3))

for row in R_Hip_fail :
  x_fail.append(float(re.findall(s1+"(.*)"+s2,row)[0]))
  y_fail.append(float(re.findall(s2+"(.*)"+s3,row)[0]))
R_Hip_start = []
for i in range(40):    
    my_tuple = (x_fail[i], y_fail[i])
    R_Hip_start.append(my_tuple)


# Define the starting positions and directions of the vectors
start_pos = np.array(R_Hip_start)
vec_dir = np.array(matrix[10].tolist())

# Define the fixed length of the vectors
vec_length = 20

# Calculate the endpoint positions of the vectors
end_pos = start_pos + vec_length * vec_dir
# Draw the vectors on the image
img = cv2.imread('./failed_frames/frame_correction1.jpg')
# Draw the vectors on the image         
cv2.arrowedLine(img, tuple(start_pos[26].astype(int)), tuple(end_pos[26].astype(int)), (0, 0, 255), thickness=2)
cv2.imwrite(os.path.join('./failed_frames/frame_correction1.jpg'), img)

img = cv2.imread('./failed_frames/frame_correction2.jpg')
# Draw the vectors on the image         
cv2.arrowedLine(img, tuple(start_pos[24].astype(int)), tuple(end_pos[24].astype(int)), (0, 0, 255), thickness=2)
cv2.imwrite(os.path.join('./failed_frames/frame_correction2.jpg'), img)


#R_Knee
R_Knee_fail=df_fail["R_Knee"]
R_Knee_fail.to_numpy()
x_fail=[]
y_fail=[]

sub1 = "("
sub2 = ","
sub3 =")"
 
s1=str(re.escape(sub1))
s2=str(re.escape(sub2))
s3=str(re.escape(sub3))

for row in R_Knee_fail :
  x_fail.append(float(re.findall(s1+"(.*)"+s2,row)[0]))
  y_fail.append(float(re.findall(s2+"(.*)"+s3,row)[0]))
R_Knee_start = []
for i in range(40):    
    my_tuple = (x_fail[i], y_fail[i])
    R_Knee_start.append(my_tuple)


# Define the starting positions and directions of the vectors
start_pos = np.array(R_Knee_start)
vec_dir = np.array(matrix[11].tolist())

# Define the fixed length of the vectors
vec_length = 20

# Calculate the endpoint positions of the vectors
end_pos = start_pos + vec_length * vec_dir
# Draw the vectors on the image
img = cv2.imread('./failed_frames/frame_correction1.jpg')
# Draw the vectors on the image         
cv2.arrowedLine(img, tuple(start_pos[26].astype(int)), tuple(end_pos[26].astype(int)), (0, 0, 255), thickness=2)
cv2.imwrite(os.path.join('./failed_frames/frame_correction1.jpg'), img)

img = cv2.imread('./failed_frames/frame_correction2.jpg')
# Draw the vectors on the image         
cv2.arrowedLine(img, tuple(start_pos[24].astype(int)), tuple(end_pos[24].astype(int)), (0, 0, 255), thickness=2)
cv2.imwrite(os.path.join('./failed_frames/frame_correction2.jpg'), img)


#R_Sholder
R_Sholder_fail=df_fail["R_Sholder"]
R_Sholder_fail.to_numpy()
x_fail=[]
y_fail=[]

sub1 = "("
sub2 = ","
sub3 =")"
 
s1=str(re.escape(sub1))
s2=str(re.escape(sub2))
s3=str(re.escape(sub3))

for row in R_Sholder_fail :
  x_fail.append(float(re.findall(s1+"(.*)"+s2,row)[0]))
  y_fail.append(float(re.findall(s2+"(.*)"+s3,row)[0]))
R_Sholder_start = []
for i in range(40):    
    my_tuple = (x_fail[i], y_fail[i])
    R_Sholder_start.append(my_tuple)


# Define the starting positions and directions of the vectors
start_pos = np.array(R_Sholder_start)
vec_dir = np.array(matrix[12].tolist())

# Define the fixed length of the vectors
vec_length = 20

# Calculate the endpoint positions of the vectors
end_pos = start_pos + vec_length * vec_dir
# Draw the vectors on the image
img = cv2.imread('./failed_frames/frame_correction1.jpg')
# Draw the vectors on the image         
cv2.arrowedLine(img, tuple(start_pos[26].astype(int)), tuple(end_pos[26].astype(int)), (0, 0, 255), thickness=2)
cv2.imwrite(os.path.join('./failed_frames/frame_correction1.jpg'), img)

img = cv2.imread('./failed_frames/frame_correction2.jpg')
# Draw the vectors on the image         
cv2.arrowedLine(img, tuple(start_pos[24].astype(int)), tuple(end_pos[24].astype(int)), (0, 0, 255), thickness=2)
cv2.imwrite(os.path.join('./failed_frames/frame_correction2.jpg'), img)
        

#R_Wrest
R_Wrest_fail=df_fail["R_Wrest"]
R_Wrest_fail.to_numpy()
x_fail=[]
y_fail=[]

sub1 = "("
sub2 = ","
sub3 =")"
 
s1=str(re.escape(sub1))
s2=str(re.escape(sub2))
s3=str(re.escape(sub3))

for row in R_Wrest_fail :
  x_fail.append(float(re.findall(s1+"(.*)"+s2,row)[0]))
  y_fail.append(float(re.findall(s2+"(.*)"+s3,row)[0]))
R_Wrest_start = []
for i in range(40):    
    my_tuple = (x_fail[i], y_fail[i])
    R_Wrest_start.append(my_tuple)


# Define the starting positions and directions of the vectors
start_pos = np.array(R_Wrest_start)
vec_dir = np.array(matrix[13].tolist())

# Define the fixed length of the vectors
vec_length = 20

# Calculate the endpoint positions of the vectors
end_pos = start_pos + vec_length * vec_dir
# Draw the vectors on the image

img = cv2.imread('./failed_frames/frame_correction1.jpg')
# Draw the vectors on the image         
cv2.arrowedLine(img, tuple(start_pos[26].astype(int)), tuple(end_pos[26].astype(int)), (0, 0, 255), thickness=2)
cv2.imwrite(os.path.join('./failed_frames/frame_correction1.jpg'), img)

img = cv2.imread('./failed_frames/frame_correction2.jpg')
# Draw the vectors on the image         
cv2.arrowedLine(img, tuple(start_pos[24].astype(int)), tuple(end_pos[24].astype(int)), (0, 0, 255), thickness=2)
cv2.imwrite(os.path.join('./failed_frames/frame_correction2.jpg'), img)