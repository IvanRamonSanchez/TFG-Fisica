import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from tqdm import tqdm
import os
import time

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

diametro_tuberia = 1 #metros
l_real = diametro_tuberia / n_y 
t_real = np.sqrt(3) * l_real #paso de tiempo de cada iteración (s)

def escalar_ejes(valor, _):
    return f"{valor * l_real:.1f}"

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

#Máscara
x_obs, y_obs = np.where(obstacle[:, :])

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

#Streamlines
def get_vel_field(x, y, macro_vel, field):
    i_0 = int(x)
    j_0 = int(y)

    tx = x - i_0
    ty = y - j_0

    sx = 1 - tx
    sy = 1 - ty

    value = (sx*sy*macro_vel[i_0, j_0, field] + tx*sy*macro_vel[i_0 + 1 , j_0, field] 
            + sx*ty*macro_vel[i_0, j_0 + 1, field] + tx*ty*macro_vel[i_0 + 1, j_0 + 1, field])
         
    return value


def get_streamlines(i, j, macro_vel):
    seg_L = 0.2
    seg_Num = 50

    x_0 = i
    y_0 = j

    x = [x_0]
    y = [y_0]

    for n in range(seg_Num):
        if (x_0 >= n_x-1) or (x_0 <= 0) or (y_0 >= n_y-1) or (y_0 <= 0):
            break 

        u = get_vel_field(x_0, y_0, macro_vel, 0)
        v = get_vel_field(x_0, y_0, macro_vel, 1)
        
        l = np.sqrt(u**2 + v**2)

        if l==0:
            break

        x_0 += u * seg_L / l
        y_0 += v * seg_L / l

        x = np.append(x, x_0)
        y = np.append(y, y_0)

    return x, y

#----------------------- SIMULACIÓN -----------------------

def main():
    def update(discrete_vel_0):
        #(1) Frontera salida
        discrete_vel_0[-1, :, left_vel] = discrete_vel_0[-2, :, left_vel]
        
        #(2) Velocidades macro
        density_0 = get_density(discrete_vel_0)
        macro_vel_0 = get_macro_vel(discrete_vel_0, density_0)

        #(3) Frontera entrada Dirichlet (Zou/He scheme)
        macro_vel_0[0, 1:-1, :] = vel_profile[0, 1:-1, :]
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

    for iter in tqdm(range(n_iter)):
        
        inicio = time.time()

        discrete_vel_1 = update(discrete_vel_0)
        discrete_vel_0 = discrete_vel_1

        final = time.time()
        tiempo_ejecucion = final - inicio

        file_name = os.path.join("C:/Users/ramon/Desktop/SIMULACIONES_TFG_FÍSICA/Lattice Boltzmann 2D", 'Tiempos de simulación (LB2D).txt')
        if n == 0:
            open(file_name, 'w')
        
        open(file_name, 'a').write('\n Frame %i, %f'%(n, tiempo_ejecucion))

        n += 1

        if iter % plot_n_steps == 0 and iter >= skip_first_iter:

            density = get_density(discrete_vel_1)
            macro_vel = get_macro_vel(discrete_vel_1, density)
            vel_magnitude = np.linalg.norm(macro_vel, axis=-1, ord=2)

            #Plot 
            fig, ax1 = plt.subplots(1, 1, sharex= 'row', figsize= (10, 6))

            t_iter = f"{iter * t_real:.2f}"

            im1 = ax1.imshow(vel_magnitude.T, origin='lower', cmap= 'rainbow', vmin= 0, vmax= 2*inflow_vel)
            ax1.set_title('Método Lattice Boltzmann 2D (t = ' + str(t_iter) + ' s)', fontsize= 20)
            
            ax1.scatter(x_obs, y_obs, s= 10, marker='s', color='black')

            #Plot streamlines
            for i in range(1, n_x, 5):
                for j in range(1, n_y, 5):
                    if obstacle[i, j] == False:
                        x, y = get_streamlines(i, j, macro_vel)
                        ax1.plot(x, y, color= 'black', linewidth = 0.5)

            plt.gca().xaxis.set_major_formatter(FuncFormatter(escalar_ejes))
            plt.gca().yaxis.set_major_formatter(FuncFormatter(escalar_ejes))

            cbar = fig.colorbar(im1, ax= ax1, shrink=0.6)
            cbar.set_label('Módulo del vector velocidad')

            frame_iter = "{:06d}".format(iter)

            file_name = os.path.join("C:/Users/ramon/Desktop/SIMULACIONES_TFG_FÍSICA/Lattice Boltzmann 2D", 'Frame_' + str(frame_iter))
            fig.savefig(file_name)
            plt.clf()
            plt.close()

if __name__ == '__main__':
    main()