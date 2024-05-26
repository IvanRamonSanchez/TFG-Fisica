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
n_x = 170
n_y = 50

x = np.arange(n_x)
y = np.arange(n_y)
X, Y = np.meshgrid(x, y, indexing = "ij")

#Parámetros simulación
n_iter = 30_000
plot_n_steps = 100
skip_first_iter = 0

#Obstáculo
obs_x = n_x / 5
obs_y = n_y / 2
obs_r = n_y / 6

obstacle = np.sqrt((X-obs_x)**2 + (Y-obs_y)**2) < obs_r
obstacle[:, 0] = True
obstacle[:, -1] = True

#Parámetros fluido
Reynolds_number = 80
inflow_vel = 0.04
viscosity = (inflow_vel * obs_r) / (Reynolds_number)
omega = (1.0)/ (3.0 * viscosity + 0.5)

#Velocidad
vel_profile = np.zeros((n_x, n_y, 2))
vel_profile[:, :, 0] = inflow_vel

n_discret_vel = 9

lattice_vel = np.array([
    [0, 1, 0, -1,  0, 1, -1, -1,  1],
    [0, 0, 1,  0, -1, 1,  1, -1, -1]
])

lattice_ind = np.array([
    0, 1, 2, 3, 4, 5, 6, 7, 8,
])

opposite_ind = np.array([
    0, 3, 4, 1, 2, 7, 8, 5, 6
])

lattice_w = np.array([
    4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36
])

right_vel = np.array([1, 5, 8])
up_vel = np.array([2, 5, 6])
left_vel = np.array([3, 6, 7])
down_vel = np.array([4, 7, 8])
vertical_vel = np.array([0, 2, 4])
horizontal_vel = np.array([0, 1, 3])

#Funciones
def get_density(discrete_vel):
    density = np.sum(discrete_vel, axis=-1)
    
    return density

def get_macro_vel(discrete_vel, density):
    macro_vel = np.einsum('NMQ, dQ -> NMd', discrete_vel, lattice_vel)/ density[..., np.newaxis]

    return macro_vel

def get_f_eq(macro_vel, density):
    proj_discete_vel = np.einsum("dQ,NMd->NMQ", lattice_vel, macro_vel)
    
    macro_vel_mag = np.linalg.norm(macro_vel, axis=-1, ord=2)
    
    f_eq = (density[..., np.newaxis] * lattice_w[np.newaxis, np.newaxis, :] * (
            1 + 3 * proj_discete_vel + 9/2 * proj_discete_vel**2 - 3/2 * macro_vel_mag[..., np.newaxis]**2
        )
    )

    return f_eq

#----------------------- SIMULACIÓN -----------------------

def main():
    def update(discrete_vel_0):
        #(1) Frontera salida
        discrete_vel_0[-1, :, left_vel] = discrete_vel_0[-2, :, left_vel]
        
        #(2) Velocidades macro
        density_0 = get_density(discrete_vel_0)
        macro_vel_0 = get_macro_vel(discrete_vel_0, density_0)

        #(3) Frontera entrada Dirichlet
        macro_vel_0[0, :, 0] = vel_profile[0, :, 0]
        density_0[0, :] = (get_density(discrete_vel_0[0, :, vertical_vel].T) + 2 * get_density(discrete_vel_0[0, :, left_vel].T)) / (1 - macro_vel_0[0, :, 0])

        #(4) f_eq 
        f_eq = get_f_eq(macro_vel_0, density_0)

        #(3) 
        discrete_vel_0[0, :, right_vel] = f_eq[0, :, right_vel]

        #(5) Colisión BGK
        discrete_vel_1 = discrete_vel_0 - omega * (discrete_vel_0 - f_eq)

        #(6) Condiciones de frontera obstaculo
        for i in range(n_discret_vel):
            discrete_vel_1[obstacle, lattice_ind[i]] = discrete_vel_0[obstacle, opposite_ind[i]]

        #(7) Condiciones de frontera paredes
        discrete_vel_2 = discrete_vel_1
        for i in range(n_discret_vel):
            discrete_vel_2[:, :, i] = np.roll(
                np.roll(
                    discrete_vel_1[:, :, i],
                    lattice_vel[0, i],
                    axis=0,
                ),
                lattice_vel[1, i],
                axis=1,
            )
        
        return discrete_vel_2
    
    discrete_vel_0 = get_f_eq(vel_profile, np.ones((n_x, n_y)))

    n = 0 #Contador
    dat = 0 #Contador

    for iter in tqdm(range(n_iter)):
        
        inicio = time.time()

        discrete_vel_1 = update(discrete_vel_0)
        discrete_vel_0 = discrete_vel_1

        final = time.time()
        tiempo_ejecucion = final - inicio

        file_name_tiempos = os.path.join(folder_padre, 'Tiempos de simulación (LB2D).txt')
        if n == 0:
            open(file_name_tiempos, 'w')
        
        open(file_name_tiempos, 'a').write('\n Frame %i, %f'%(n, tiempo_ejecucion))

        n += 1

        if iter % plot_n_steps == 0 and iter >= skip_first_iter:
            dat += 1

            density = get_density(discrete_vel_1)
            macro_vel = get_macro_vel(discrete_vel_1, density)

            file_name_densidad = os.path.join(folder_densidad, 'Densidades_' + str(dat) + '(LB2D).json')
            file_name_velocidad = os.path.join(folder_velocidad, 'Velocidades_' + str(dat) + '(LB2D).json')

            density_lista = density.tolist()
            with open(file_name_densidad, 'w') as archivo:
                json.dump(density_lista, archivo)

            macro_vel_lista = macro_vel.tolist()
            with open(file_name_velocidad, 'w') as archivo:
                json.dump(macro_vel_lista, archivo)

if __name__ == '__main__':
    main()