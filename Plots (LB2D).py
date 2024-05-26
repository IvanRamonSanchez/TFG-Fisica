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

n_x = 170
n_y = 50

x = np.arange(n_x)
y = np.arange(n_y)
X, Y = np.meshgrid(x, y, indexing = "ij")

diametro_canal = 1 #metros
l_real = diametro_canal / n_y 
t_real = np.sqrt(3) * l_real #paso de tiempo de cada iteración (s)

def escalar_ejes(valor, _):
    return f"{valor * l_real:.1f}"

obs_x = n_x / 5
obs_y = n_y / 2
obs_r = n_y / 6

obstacle = np.sqrt((X-obs_x)**2 + (Y-obs_y)**2) < obs_r
obstacle[:, 0] = True
obstacle[:, -1] = True

inflow_vel = 0.04

x_obs, y_obs = np.where(obstacle[:, :])

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

for dat in tqdm(range(numero_frames)):
    dat += 1
    archivo_densidad = os.path.join(folder_densidad, 'Densidades_' + str(dat) + '(LB2D).json')
    archivo_velocidad = os.path.join(folder_velocidad, 'Velocidades_' + str(dat) + '(LB2D).json')

    if os.path.exists(archivo_densidad) == True:
        with open(archivo_densidad, 'r') as archivo:
            density_json = json.load(archivo)
        density = np.array(density_json)

        with open(archivo_velocidad, 'r') as archivo:
            macro_vel_json = json.load(archivo)
        macro_vel = np.array(macro_vel_json)
        vel_magnitude = np.linalg.norm(macro_vel, axis=-1, ord=2)

        #Plot_Densidad
        fig, ax1 = plt.subplots(1, 1, sharex= 'row', figsize= (10, 6))

        iter = 100 * (dat - 1)
        t_iter = f"{iter * t_real:.2f}"

        im1 = ax1.imshow(density.T, origin='lower', cmap= 'rainbow')
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
        cbar.set_label('Densidad')

        frame_iter = "{:06d}".format(iter)

        file_name_densidad = os.path.join(folder_plots_densidad, 'Frame_' + str(frame_iter))
        fig.savefig(file_name_densidad)
        plt.clf()
        plt.close()

        #Plot_Velocidad
        fig, ax1 = plt.subplots(1, 1, sharex= 'row', figsize= (10, 6))

        im1 = ax1.imshow(vel_magnitude.T, origin='lower', cmap= 'rainbow', vmin = 0, vmax = 2 * inflow_vel)
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

        file_name_velocidad = os.path.join(folder_plots_velocidad, 'Frame_' + str(frame_iter))
        fig.savefig(file_name_velocidad)
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
video_name = folder_padre + '/LB2D_densidad.avi'
create_video_from_images(folder_plots_densidad, video_name, fps)

#Compilador vídeo Velocidad
video_name = folder_padre + '/LB2D_velocidad.avi'
create_video_from_images(folder_plots_velocidad, video_name, fps)
