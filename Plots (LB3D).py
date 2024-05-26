import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from tqdm import tqdm
import os
import json
import cv2
import shutil

#Crear directorios
folder_padre = os.path.dirname(os.path.abspath(__file__))
folder_densidad = folder_padre + '/Densidades'
folder_velocidad = folder_padre + '/Velocidades'

folder_plots_densidad = folder_padre + '/Plots_densidad'
if os.path.exists(folder_plots_densidad):
    shutil.rmtree(folder_plots_densidad)
os.makedirs(folder_plots_densidad)

folder_plots_velocidad = folder_padre + '/Plots_velocidad'
if os.path.exists(folder_plots_velocidad):
    shutil.rmtree(folder_plots_velocidad)
os.makedirs(folder_plots_velocidad)

#Parámetros simulación
numero_frames = 10_000
fps = 10

n_x = 50
n_y = 50
n_z = 170

x = np.arange(n_x)
y = np.arange(n_y)
z = np.arange(n_z)
X, Y, Z = np.meshgrid(x, y, z, indexing = "ij")

diametro_canal = 1 #metros
l_real = diametro_canal / n_y 
t_real = np.sqrt(3) * l_real #paso de tiempo de cada iteración (s)

def escalar_ejes(valor, _):
    return f"{valor * l_real:.1f}"

obs_x = n_x / 2
obs_y = n_y / 2
obs_z = n_z / 5
obs_r = n_y / 6

obstacle = np.sqrt((X-obs_x)**2 + (Z-obs_z)**2) < obs_r
obstacle[0, :, :] = True
obstacle[-1, :, :] = True
obstacle[:, 0, :] = True
obstacle[:, -1, :] = True

inflow_vel = 0.04

y_planta = int(n_y/2) 
x_obs_planta, z_obs_planta = np.where(obstacle[:, y_planta, :])

x_alzado = int(n_x/2)
y_obs_alzado, z_obs_alzado = np.where(obstacle[x_alzado, :, :])

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

for dat in tqdm(range(numero_frames)):
    dat += 1
    archivo_densidad = os.path.join(folder_densidad, 'Densidades_' + str(dat) + '(LB3D).json')
    archivo_velocidad = os.path.join(folder_velocidad, 'Velocidades_' + str(dat) + '(LB3D).json')

    if os.path.exists(archivo_densidad) == True:
        with open(archivo_densidad, 'r') as archivo:
            density_json = json.load(archivo)
        density = np.array(density_json)

        with open(archivo_velocidad, 'r') as archivo:
            macro_vel_json = json.load(archivo)
        macro_vel = np.array(macro_vel_json)
        vel_magnitude = np.linalg.norm(macro_vel, axis=-1, ord=2)

        #Plot_Densidad
        fig, axs = plt.subplots(2, 1, sharex= 'row', figsize= (10, 6)) 
        
        iter = 100 * (dat - 1)
        t_iter = f"{iter * t_real:.2f}"
        fig.suptitle('Método Lattice Boltzmann 3D (t = ' + str(t_iter) + ' s)', fontsize= 20)
    
            #Plot planta
        density_planta = density[:, y_planta, :]

        im1 = axs[0].imshow(density_planta, origin= 'lower', cmap= 'rainbow')
            
        axs[0].set_title('Vista de Planta (Y = %i)' %(y_planta), loc='center', fontsize= 10)

        axs[0].scatter(z_obs_planta, x_obs_planta, s= 10, marker='s', color='black')
 
        for i in range(1, n_x, 5):
            for k in range(1, n_z, 5):
                if obstacle[i, y_planta, k] == False:
                    x, y, z = get_streamlines(i, y_planta, k, macro_vel)
                    axs[0].plot(z, x, color= 'black', linewidth = 0.5)

            #Plot alzado
        density_alzado = density[x_alzado, :, :]
            
        im2 = axs[1].imshow(density_alzado, origin= 'lower', cmap= 'rainbow')
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
        cbar.set_label('Densidad')

        frame_iter = "{:06d}".format(iter)
            
        frame_name_densidad = os.path.join(folder_plots_densidad, 'Frame_' + str(frame_iter))
        fig.savefig(frame_name_densidad)
        plt.clf()
        plt.close()

        #Plot Velocidad
        fig, axs = plt.subplots(2, 1, sharex= 'row', figsize= (10, 6)) 
        
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
            
        frame_name_velocidad = os.path.join(folder_plots_velocidad, 'Frame_' + str(frame_iter))
        fig.savefig(frame_name_velocidad)
        plt.clf()
        plt.close()

def create_video_from_images(image_folder, video_name, fps):
    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'DIVX'), fps, (width,height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()

#Compilador vídeo Densidad
video_name = folder_padre + '/LB3D_densidad.avi'
create_video_from_images(folder_plots_densidad, video_name, fps)

#Compilador vídeo Velocidad
video_name = folder_padre + '/LB3D_velocidad.avi'
create_video_from_images(folder_plots_velocidad, video_name, fps)
