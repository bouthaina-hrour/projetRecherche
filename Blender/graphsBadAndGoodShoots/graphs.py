
# Importing libraries
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re
df_succ=pd.read_csv("../Results/GoodShoot.csv",sep = ',')

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

df_fail=pd.read_csv("../Results/BadShoot.csv",sep = ',')

L_ankle_fail=df_fail["L_Ankle"]
L_ankle_fail.to_numpy()
x_fail=[]
y_fail=[]


for row in L_ankle_fail :
  x_fail.append(float(re.findall(s1+"(.*)"+s2,row)[0]))
  y_fail.append(float(re.findall(s2+"(.*)"+s3,row)[0]))


# Plotting both the curves simultaneously
plt.plot(x_fail, y_fail, color='r', label='Bad Shoot')
plt.plot(x_succ, y_succ, color='g', label='Good Shoot')
  
# Naming the x-axis, y-axis and the whole graph
plt.title("L_Ankle positions for both shoots")
  
# Adding legend, which helps us recognize the curve according to it's color
plt.legend()
  
# To load the display window
plt.show()
##################L_ELBOW########################
df_succ=pd.read_csv("../Results/GoodShoot.csv",sep = ',')

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

df_fail=pd.read_csv("../Results/BadShoot.csv",sep = ',')

L_Elbow_fail=df_fail["L_Elbow"]
L_Elbow_fail.to_numpy()
x_fail=[]
y_fail=[]


for row in L_Elbow_fail :
  x_fail.append(float(re.findall(s1+"(.*)"+s2,row)[0]))
  y_fail.append(float(re.findall(s2+"(.*)"+s3,row)[0]))


# Plotting both the curves simultaneously
plt.plot(x_fail, y_fail, color='r', label='Bad Shoot')
plt.plot(x_succ, y_succ, color='g', label='Good Shoot')
  
# Naming the x-axis, y-axis and the whole graph
plt.title("L_Elbow positions for both shoots")
  
# Adding legend, which helps us recognize the curve according to it's color
plt.legend()
  
# To load the display window
plt.show()

##################L_HIP########################
df_succ=pd.read_csv("../Results/GoodShoot.csv",sep = ',')

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

df_fail=pd.read_csv("../Results/BadShoot.csv",sep = ',')

L_Hip_fail=df_fail["L_Hip"]
L_Hip_fail.to_numpy()
x_fail=[]
y_fail=[]


for row in L_Hip_fail :
  x_fail.append(float(re.findall(s1+"(.*)"+s2,row)[0]))
  y_fail.append(float(re.findall(s2+"(.*)"+s3,row)[0]))


# Plotting both the curves simultaneously
plt.plot(x_fail, y_fail, color='r', label='Bad Shoot')
plt.plot(x_succ, y_succ, color='g', label='Good Shoot')
  
# Naming the x-axis, y-axis and the whole graph
plt.title("L_Hip positions for both shoots")
  
# Adding legend, which helps us recognize the curve according to it's color
plt.legend()
  
# To load the display window
plt.show()

##################L_Knee########################
df_succ=pd.read_csv("../Results/GoodShoot.csv",sep = ',')

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

df_fail=pd.read_csv("../Results/BadShoot.csv",sep = ',')

L_Knee_fail=df_fail["L_knee"]
L_Knee_fail.to_numpy()
x_fail=[]
y_fail=[]


for row in L_Knee_fail :
  x_fail.append(float(re.findall(s1+"(.*)"+s2,row)[0]))
  y_fail.append(float(re.findall(s2+"(.*)"+s3,row)[0]))


# Plotting both the curves simultaneously
plt.plot(x_fail, y_fail, color='r', label='Bad Shoot')
plt.plot(x_succ, y_succ, color='g', label='Good Shoot')
  
# Naming the x-axis, y-axis and the whole graph
plt.title("L_Knee positions for both shoots")
  
# Adding legend, which helps us recognize the curve according to it's color
plt.legend()
  
# To load the display window
plt.show()

##################L_Sholder########################
df_succ=pd.read_csv("../Results/GoodShoot.csv",sep = ',')

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

df_fail=pd.read_csv("../Results/BadShoot.csv",sep = ',')

L_Sholder_fail=df_fail["L_Sholder"]
L_Sholder_fail.to_numpy()
x_fail=[]
y_fail=[]


for row in L_Sholder_fail :
  x_fail.append(float(re.findall(s1+"(.*)"+s2,row)[0]))
  y_fail.append(float(re.findall(s2+"(.*)"+s3,row)[0]))


# Plotting both the curves simultaneously
plt.plot(x_fail, y_fail, color='r', label='Bad Shoot')
plt.plot(x_succ, y_succ, color='g', label='Good Shoot')
  
# Naming the x-axis, y-axis and the whole graph
plt.title("L_Sholder positions for both shoots")
  
# Adding legend, which helps us recognize the curve according to it's color
plt.legend()
  
# To load the display window
plt.show()

##################L_Wrest########################
df_succ=pd.read_csv("../Results/GoodShoot.csv",sep = ',')

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

df_fail=pd.read_csv("../Results/BadShoot.csv",sep = ',')

L_Wrest_fail=df_fail["L_Wrest"]
L_Wrest_fail.to_numpy()
x_fail=[]
y_fail=[]


for row in L_Wrest_fail :
  x_fail.append(float(re.findall(s1+"(.*)"+s2,row)[0]))
  y_fail.append(float(re.findall(s2+"(.*)"+s3,row)[0]))


# Plotting both the curves simultaneously
plt.plot(x_fail, y_fail, color='r', label='Bad Shoot')
plt.plot(x_succ, y_succ, color='g', label='Good Shoot')
  
# Naming the x-axis, y-axis and the whole graph
plt.title("L_Wrest positions for both shoots")
  
# Adding legend, which helps us recognize the curve according to it's color
plt.legend()
  
# To load the display window
plt.show()

##################neck########################
df_succ=pd.read_csv("../Results/GoodShoot.csv",sep = ',')

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

df_fail=pd.read_csv("../Results/BadShoot.csv",sep = ',')

neck_fail=df_fail["Neck"]
neck_fail.to_numpy()
x_fail=[]
y_fail=[]


for row in neck_fail :
  x_fail.append(float(re.findall(s1+"(.*)"+s2,row)[0]))
  y_fail.append(float(re.findall(s2+"(.*)"+s3,row)[0]))


# Plotting both the curves simultaneously
plt.plot(x_fail, y_fail, color='r', label='Bad Shoot')
plt.plot(x_succ, y_succ, color='g', label='Good Shoot')
  
# Naming the x-axis, y-axis and the whole graph
plt.title("Neck positions for both shoots")
  
# Adding legend, which helps us recognize the curve according to it's color
plt.legend()
  
# To load the display window
plt.show()

##################Nose########################
df_succ=pd.read_csv("../Results/GoodShoot.csv",sep = ',')

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

df_fail=pd.read_csv("../Results/BadShoot.csv",sep = ',')

Nose_fail=df_fail["Noze"]
Nose_fail.to_numpy()
x_fail=[]
y_fail=[]


for row in Nose_fail :
  x_fail.append(float(re.findall(s1+"(.*)"+s2,row)[0]))
  y_fail.append(float(re.findall(s2+"(.*)"+s3,row)[0]))


# Plotting both the curves simultaneously
plt.plot(x_fail, y_fail, color='r', label='Bad Shoot')
plt.plot(x_succ, y_succ, color='g', label='Good Shoot')
  
# Naming the x-axis, y-axis and the whole graph
plt.title("Nose positions for both shoots")
  
# Adding legend, which helps us recognize the curve according to it's color
plt.legend()
  
# To load the display window
plt.show()

##################R_Ankle########################
df_succ=pd.read_csv("../Results/GoodShoot.csv",sep = ',')

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

df_fail=pd.read_csv("../Results/BadShoot.csv",sep = ',')

R_Ankle_fail=df_fail["R_Ankle"]
R_Ankle_fail.to_numpy()
x_fail=[]
y_fail=[]


for row in R_Ankle_fail :
  x_fail.append(float(re.findall(s1+"(.*)"+s2,row)[0]))
  y_fail.append(float(re.findall(s2+"(.*)"+s3,row)[0]))


# Plotting both the curves simultaneously
plt.plot(x_fail, y_fail, color='r', label='Bad Shoot')
plt.plot(x_succ, y_succ, color='g', label='Good Shoot')
  
# Naming the x-axis, y-axis and the whole graph
plt.title("R_Ankle positions for both shoots")
  
# Adding legend, which helps us recognize the curve according to it's color
plt.legend()
  
# To load the display window
plt.show()

##################R_Elbow########################
df_succ=pd.read_csv("../Results/GoodShoot.csv",sep = ',')

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

df_fail=pd.read_csv("../Results/BadShoot.csv",sep = ',')

R_Elbow_fail=df_fail["R_Elbow"]
R_Elbow_fail.to_numpy()
x_fail=[]
y_fail=[]


for row in R_Elbow_fail :
  x_fail.append(float(re.findall(s1+"(.*)"+s2,row)[0]))
  y_fail.append(float(re.findall(s2+"(.*)"+s3,row)[0]))


# Plotting both the curves simultaneously
plt.plot(x_fail, y_fail, color='r', label='Bad Shoot')
plt.plot(x_succ, y_succ, color='g', label='Good Shoot')
  
# Naming the x-axis, y-axis and the whole graph
plt.title("R_Elbow positions for both shoots")
  
# Adding legend, which helps us recognize the curve according to it's color
plt.legend()
  
# To load the display window
plt.show()

##################R_Hip########################
df_succ=pd.read_csv("../Results/GoodShoot.csv",sep = ',')

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

df_fail=pd.read_csv("../Results/BadShoot.csv",sep = ',')

R_Hip_fail=df_fail["R_Hip"]
R_Hip_fail.to_numpy()
x_fail=[]
y_fail=[]


for row in R_Hip_fail :
  x_fail.append(float(re.findall(s1+"(.*)"+s2,row)[0]))
  y_fail.append(float(re.findall(s2+"(.*)"+s3,row)[0]))


# Plotting both the curves simultaneously
plt.plot(x_fail, y_fail, color='r', label='Bad Shoot')
plt.plot(x_succ, y_succ, color='g', label='Good Shoot')
  
# Naming the x-axis, y-axis and the whole graph
plt.title("R_Hip positions for both shoots")
  
# Adding legend, which helps us recognize the curve according to it's color
plt.legend()
  
# To load the display window
plt.show()

##################R_Sholder########################
df_succ=pd.read_csv("../Results/GoodShoot.csv",sep = ',')

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

df_fail=pd.read_csv("../Results/BadShoot.csv",sep = ',')

R_Sholder_fail=df_fail["R_Sholder"]
R_Sholder_fail.to_numpy()
x_fail=[]
y_fail=[]


for row in R_Sholder_fail :
  x_fail.append(float(re.findall(s1+"(.*)"+s2,row)[0]))
  y_fail.append(float(re.findall(s2+"(.*)"+s3,row)[0]))


# Plotting both the curves simultaneously
plt.plot(x_fail, y_fail, color='r', label='Bad Shoot')
plt.plot(x_succ, y_succ, color='g', label='Good Shoot')
  
# Naming the x-axis, y-axis and the whole graph
plt.title("R_Sholder positions for both shoots")
  
# Adding legend, which helps us recognize the curve according to it's color
plt.legend()
  
# To load the display window
plt.show()

##################R_Wrest########################
df_succ=pd.read_csv("../Results/GoodShoot.csv",sep = ',')

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

df_fail=pd.read_csv("../Results/BadShoot.csv",sep = ',')

R_Wrest_fail=df_fail["R_Wrest"]
R_Wrest_fail.to_numpy()
x_fail=[]
y_fail=[]


for row in R_Wrest_fail :
  x_fail.append(float(re.findall(s1+"(.*)"+s2,row)[0]))
  y_fail.append(float(re.findall(s2+"(.*)"+s3,row)[0]))


# Plotting both the curves simultaneously
plt.plot(x_fail, y_fail, color='r', label='Bad Shoot')
plt.plot(x_succ, y_succ, color='g', label='Good Shoot')
  
# Naming the x-axis, y-axis and the whole graph
plt.title("R_Wrest positions for both shoots")
  
# Adding legend, which helps us recognize the curve according to it's color
plt.legend()
  
# To load the display window
plt.show()

