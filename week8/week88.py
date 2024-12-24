import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the previous color image:
I = cv2.imread('matting1.jpg')                                        
I = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)/255                        
n_rows = I.shape[0]                           
n_cols = I.shape[1]                              
n_pixels = n_rows * n_cols                      
I = np.reshape(I, (n_pixels, 3))

# Load exact matte:
alpha_ex = cv2.imread('matting1.jpg', cv2.IMREAD_GRAYSCALE)/255 
alpha_ex = np.reshape(alpha_ex, (n_pixels, 1))   

# Generate a version with a green screen:
G_B = 1
I = alpha_ex * I + (1 - alpha_ex) * [0, G_B, 0] 
I = np.reshape(I, (n_rows, n_cols, 3))
plt.imshow(I), plt.axis('off'), plt.show();

# RGB components:
R_I = I[:, :, 0]  
G_I = I[:, :, 1] 
B_I = I[:, :, 2] 

# Coefficients:
a_0 = 0
a_1 = 0.5
a_2 = 0

# Compute alpha with formula:
alpha = (B_I - a_1*(G_I - G_B) - a_2*R_I)/(a_0 + a_1*G_B)     
alpha[alpha < 0] = 0
alpha[alpha > 1] = 1
plt.imshow(alpha, cmap='gray'), plt.axis('off'), plt.show();

# Compute relative error with exact alpha:
alpha = np.reshape(alpha, (n_pixels, 1))   
error = np.linalg.norm(alpha - alpha_ex)/np.linalg.norm(alpha_ex) 
print(f'Error (alpha): {error:.2e}')  

# Load background K:
K = cv2.imread('kor.jpeg')               
K = cv2.cvtColor(K, cv2.COLOR_BGR2RGB)/255   
K = K[:n_rows, :n_cols, :]                  
K = np.reshape(K, (n_pixels, 3))

# Compute new image J with background K:
alpha = np.reshape(alpha, (n_pixels, 1)) 
R_I = np.reshape(R_I, (n_pixels, 1))
B_I = np.reshape(B_I, (n_pixels, 1))
G_I = np.reshape(G_I, (n_pixels, 1))
tmp = np.minimum(G_I, (B_I - a_0*alpha - a_2*R_I)/a_1)
J = np.concatenate((R_I, tmp, B_I), axis=1) + (1 - alpha)*K
J[J < 0] = 0
J[J > 1] = 1
J = np.reshape(J, (n_rows, n_cols, 3))
plt.imshow(J), plt.axis('off'), plt.show();  
