import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from tqdm import tqdm
import os
import time
import json
import shutil

#Crear directorios
folder_padre = os.path.dirname(os.path.abspath(__file__))

folder_densidad = folder_padre + '/Densidades'
if os.path.exists(folder_densidad):
    shutil.rmtree(folder_densidad)
os.makedirs(folder_densidad)

folder_velocidad = folder_padre + '/Velocidades'
if os.path.exists(folder_velocidad):
    shutil.rmtree(folder_velocidad)
os.makedirs(folder_velocidad)

#Malla
n_x = 50
n_y = 50
n_z = 170

x = np.arange(n_x)
y = np.arange(n_y)
z = np.arange(n_z)
X, Y, Z = np.meshgrid(x, y, z, indexing = "ij")

#Parámetros simulación
n_iter = 30_000
plot_n_steps = 100
skip_first_iter = 0

#Obstáculo
obs_x = n_x / 2
obs_y = n_y / 2
obs_z = n_z / 5
obs_r = n_y / 6

obstacle = np.sqrt((X-obs_x)**2 + (Z-obs_z)**2) < obs_r
obstacle[0, :, :] = True
obstacle[-1, :, :] = True
obstacle[:, 0, :] = True
obstacle[:, -1, :] = True

#Parámetros fluido
Reynolds_number = 80
inflow_vel = 0.04
viscosity = (inflow_vel * obs_r) / (Reynolds_number)
tau = 3.0 * viscosity + 0.5
omega = (1.0) / (3.0 * viscosity + 0.5)

#Velocidad
vel_profile = np.zeros((n_x, n_y, n_z, 3))
vel_profile[:, :, :, -1] = inflow_vel

n_discret_vel = 15

lattice_vel = np.array([
    [0, 1, 0, 0, -1,  0,  0, 1,  1,  1,  1, -1, -1, -1, -1],
    [0, 0, 1, 0,  0, -1,  0, 1,  1, -1, -1,  1,  1, -1, -1],
    [0, 0, 0, 1,  0,  0, -1, 1, -1,  1, -1,  1, -1,  1, -1]
])

lattice_ind = np.array([
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14
])

opposite_ind = np.array([
    0, 4, 5, 6, 1, 2, 3, 14, 13, 12, 11, 10, 9, 8, 7
])

lattice_w = np.array([
    2/9, 
    1/9, 1/9, 1/9, 1/9, 1/9, 1/9, 
    1/72, 1/72, 1/72, 1/72, 1/72, 1/72, 1/72, 1/72
])

x_neg_vel = np.array([4, 11, 12, 13, 14])
x_0_vel = np.array([0, 2, 3, 5, 6])
x_pos_vel = np.array([1, 7, 8, 9, 10])

y_neg_vel = np.array([5, 9, 10, 13, 14])
y_0_vel = np.array([0, 1, 3, 4, 6])
y_pos_vel = np.array([2, 7, 8, 11, 12])

z_neg_vel = np.array([6, 8, 10, 12, 14])
z_0_vel = np.array([0, 1, 2, 4, 5])
z_pos_vel = np.array([3, 7, 9, 11, 13])

#Gravedad
gravity = -0.0001
F = np.zeros((n_x, n_y, n_z, 3))
F[:, :, :, 1] = gravity

proj_force = np.einsum('LNMd,dQ->LNMQ', F, lattice_vel)
gravity_force = 3 * proj_force * (2 * tau - 1)/(2 * tau) 

#Funciones
def get_density(discrete_vel):
    density = np.sum(discrete_vel, axis=-1, dtype= np.float32)
    
    return density

def get_macro_vel(discrete_vel, density):
    macro_vel = np.einsum('LMNQ, dQ -> LMNd', discrete_vel, lattice_vel)/ density[..., np.newaxis]

    return macro_vel

def get_f_eq(macro_vel, density):
    gravity_macro_vel = macro_vel + (F / (2 * density[..., np.newaxis]))

    proj_discete_vel = np.einsum("dQ,LMNd->LMNQ", lattice_vel, gravity_macro_vel)
    
    macro_vel_mag = np.linalg.norm(gravity_macro_vel, axis=-1)
    
    f_eq = (density[..., np.newaxis] * lattice_w[np.newaxis, np.newaxis, np.newaxis, :] * (
            1 + 3 * proj_discete_vel + 9/2 * proj_discete_vel**2 - 3/2 * macro_vel_mag[..., np.newaxis]**2
        )
    )

    return f_eq

#----------------------- SIMULACIÓN -----------------------

def main():
    def update(discrete_vel_0):
        #(1) Frontera salida
        discrete_vel_0[:, :, -1, z_neg_vel] = discrete_vel_0[:, :, -2, z_neg_vel]

        #(2) Velocidades macro
        density_0 = get_density(discrete_vel_0)
        macro_vel_0 = get_macro_vel(discrete_vel_0, density_0)

        #(3) Frontera entrada Dirichlet
        macro_vel_0[:, :, 0, :] = vel_profile[:, :, 0, :]
        density_0[:, :, 0] = (get_density(discrete_vel_0[:, :, 0, z_0_vel]) + 2 * get_density(discrete_vel_0[:, :, 0, z_neg_vel])) / (1 - macro_vel_0[:, :, 0, -1])

        #(4) f_eq 
        f_eq = get_f_eq(macro_vel_0, density_0)

        #(3) 
        discrete_vel_0[:, :, 0, z_pos_vel] = f_eq[:, :, 0, z_pos_vel]

        #(5) Colisión BGK
        discrete_vel_1 = discrete_vel_0 - omega * (discrete_vel_0 - f_eq) + gravity_force

        #(6) Condiciones de frontera obstaculo
        for i in range(n_discret_vel):
            discrete_vel_1[obstacle, lattice_ind[i]] = discrete_vel_0[obstacle, opposite_ind[i]]

        #(7) Difusión
        discrete_vel_2 = discrete_vel_1
        for i in range(n_discret_vel):
            discrete_vel_2[:, :, :, i] = np.roll(
                np.roll(
                    np.roll(
                        discrete_vel_1[:, :, :, i],
                        lattice_vel[0, i],
                        axis=0,
                    ),
                    lattice_vel[1, i],
                    axis=1,
                ),
                lattice_vel[2, i],
                axis= 2,
            )

        return discrete_vel_2

    discrete_vel_0 = get_f_eq(vel_profile, np.ones((n_x, n_y, n_z)))

    n = 0 #Contador
    dat = 0 #Contador

    for iter in tqdm(range(n_iter)):
        
        inicio = time.time()

        discrete_vel_1 = update(discrete_vel_0)
        discrete_vel_0 = discrete_vel_1

        final = time.time()
        tiempo_ejecucion = final - inicio

        file_name_tiempos = os.path.join(folder_padre, 'Tiempos de simulación (LB3D).txt')
        if n == 0:
            open(file_name_tiempos, 'w')
        
        open(file_name_tiempos, 'a').write('\n Frame %i, %f'%(n, tiempo_ejecucion))

        n += 1

        if iter % plot_n_steps == 0  and iter >= skip_first_iter:
            dat += 1

            density = get_density(discrete_vel_1)
            macro_vel = get_macro_vel(discrete_vel_1, density)
            
            file_name_densidad = os.path.join(folder_densidad, 'Densidades_' + str(dat) + '(LB3D).json')
            file_name_velocidad = os.path.join(folder_velocidad, 'Velocidades_' + str(dat) + '(LB3D).json')

            density_lista = density.tolist()
            with open(file_name_densidad, 'w') as archivo:
                json.dump(density_lista, archivo)

            macro_vel_lista = macro_vel.tolist()
            with open(file_name_velocidad, 'w') as archivo:
                json.dump(macro_vel_lista, archivo)

if __name__ == '__main__':
    main()
