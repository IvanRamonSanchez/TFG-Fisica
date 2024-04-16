import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from tqdm import tqdm
import os
import time

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

diametro_tuberia = 1 #metros
l_real = diametro_tuberia / n_y 
t_real = np.sqrt(3) * l_real #paso de tiempo de cada iteración (s)

def escalar_ejes(valor, _):
    return f"{valor * l_real:.1f}"

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

#Máscara
y_planta = int(n_y/2) 
x_obs_planta, z_obs_planta = np.where(obstacle[:, y_planta, :])

x_alzado = int(n_x/2)
y_obs_alzado, z_obs_alzado = np.where(obstacle[x_alzado, :, :])

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

#Streamlines
def get_vel_field(x, y, z, macro_vel, field):
    i_0 = int(x)
    j_0 = int(y)
    k_0 = int(z)

    tx = x - i_0
    ty = y - j_0
    tz = z - k_0

    sx = 1 - tx
    sy = 1 - ty
    sz = 1 - tz

    value = (sx*sy*sz*macro_vel[i_0, j_0, k_0, field] + tx*sy*sz*macro_vel[i_0 + 1 , j_0, k_0, field] 
            + sx*ty*sz*macro_vel[i_0, j_0 + 1, k_0, field] + tx*ty*sz*macro_vel[i_0 + 1, j_0 + 1, k_0, field]
            + sx*sy*tz*macro_vel[i_0, j_0, k_0 + 1, field] + tx*sy*tz*macro_vel[i_0 + 1, j_0, k_0 + 1, field]
            + sx*ty*tz*macro_vel[i_0, j_0 + 1, k_0 +1, field] + tx*ty*tz*macro_vel[i_0 + 1, j_0 + 1, k_0 +1, field])
         
    return value


def get_streamlines(i, j, k, macro_vel):
    seg_L = 0.2
    seg_Num = 50

    x_0 = i
    y_0 = j
    z_0 = k

    x = [x_0]
    y = [y_0]
    z = [z_0]

    for n in range(seg_Num):
        if (x_0 >= n_x-1) or (x_0 <= 0) or (y_0 >= n_y-1) or (y_0 <= 0) or (z_0 >= n_z-1) or (z_0 <= 0):
            break 

        u = get_vel_field(x_0, y_0, z_0, macro_vel, 0)
        v = get_vel_field(x_0, y_0, z_0, macro_vel, 1)
        w = get_vel_field(x_0, y_0, z_0, macro_vel, -1)

        l = np.sqrt(u**2 + v**2 + w**2)

        if l==0:
            break

        x_0 += u * seg_L / l
        y_0 += v * seg_L / l
        z_0 += w * seg_L / l

        x = np.append(x, x_0)
        y = np.append(y, y_0)
        z = np.append(z, z_0)

    return x, y, z

#----------------------- SIMULACIÓN -----------------------

def main():
    def update(discrete_vel_0):
        #(1) Frontera salida
        discrete_vel_0[:, :, -1, z_neg_vel] = discrete_vel_0[:, :, -2, z_neg_vel]
        
        #discrete_vel_0[:, -1, :, :] = discrete_vel_0[:, -2, :, :]
        #discrete_vel_0[0, :, :, :] = discrete_vel_0[1, :, :, :]
        #discrete_vel_0[-1, :, :, :] = discrete_vel_0[-2, :, :, :]

        #(2) Velocidades macro
        density_0 = get_density(discrete_vel_0)
        macro_vel_0 = get_macro_vel(discrete_vel_0, density_0)

        #(3) Frontera entrada Dirichlet (Zou/He scheme)
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

    for iter in tqdm(range(n_iter)):
        
        inicio = time.time()

        discrete_vel_1 = update(discrete_vel_0)
        discrete_vel_0 = discrete_vel_1

        final = time.time()
        tiempo_ejecucion = final - inicio

        tiempos = os.path.join("C:/Users/ramon/Desktop/SIMULACIONES_TFG_FÍSICA/Lattice Boltzmann 3D", 'Tiempos de simulación (LB3D).txt')
        if n == 0:
            open(tiempos, 'w')
        
        open(tiempos, 'a').write('\n Frame %i, %f'%(n, tiempo_ejecucion))

        n += 1

        if iter % plot_n_steps == 0  and iter >= skip_first_iter:

            density = get_density(discrete_vel_1)
            macro_vel = get_macro_vel(discrete_vel_1, density)
            vel_magnitude = np.linalg.norm(macro_vel, axis=-1)

            #Plot
            fig, axs = plt.subplots(2, 1, sharex= 'row', figsize= (10, 6)) 
            t_iter = f"{iter * t_real:.2f}"
            fig.suptitle('Método Lattice Boltzmann 3D (t = ' + str(t_iter) + ' s)', fontsize= 20)
    
            #Plot planta
            vel_magnitude_planta = vel_magnitude[:, y_planta, :]

            im1 = axs[0].imshow(vel_magnitude_planta, origin= 'lower', cmap= 'rainbow', vmin= 0, vmax= 2*inflow_vel)
            axs[0].set_title('Vista de Planta (Y = %i)' %(y_planta), loc='center', fontsize= 10)

            axs[0].scatter(z_obs_planta, x_obs_planta, s= 10, marker='s', color='black')
 
            for i in range(1, n_x, 5):
                for k in range(1, n_z, 5):
                    if obstacle[i, y_planta, k] == False:
                        x, y, z = get_streamlines(i, y_planta, k, macro_vel)
                        axs[0].plot(z, x, color= 'black', linewidth = 0.5)

            #Plot alzado
            vel_magnitude_alzado = vel_magnitude[x_alzado, :, :]
            
            im2 = axs[1].imshow(vel_magnitude_alzado, origin= 'lower', cmap= 'rainbow', vmin= 0, vmax= 2*inflow_vel)
            axs[1].set_title('Vista de Alzado (X = %i)' %(x_alzado), loc='center', fontsize= 10)
            
            axs[1].scatter(z_obs_alzado, y_obs_alzado, s= 10, marker='s', color='black')

            for j in range(1, n_y, 5):
                for k in range(1, n_z, 5):
                    if obstacle[x_alzado, j, k] == False:
                        x, y, z = get_streamlines(x_alzado, j, k, macro_vel)
                        axs[1].plot(z, y, color= 'black', linewidth = 0.5)

            axs[0].xaxis.set_major_formatter(plt.FuncFormatter(escalar_ejes))
            axs[0].yaxis.set_major_formatter(plt.FuncFormatter(escalar_ejes))

            axs[1].xaxis.set_major_formatter(plt.FuncFormatter(escalar_ejes))
            axs[1].yaxis.set_major_formatter(plt.FuncFormatter(escalar_ejes))

            cbar = fig.colorbar(im1, ax=axs.ravel().tolist())
            cbar.set_label('Módulo del vector velocidad')

            frame_iter = "{:06d}".format(iter)
            
            frame_name = os.path.join("C:/Users/ramon/Desktop/SIMULACIONES_TFG_FÍSICA/Lattice Boltzmann 3D", 'Frame_' + str(frame_iter))
            fig.savefig(frame_name)
            plt.clf()
            plt.close()

if __name__ == '__main__':
    main()