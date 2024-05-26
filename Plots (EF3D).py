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
folder_presion = folder_padre + '/Presiones'
folder_velocidad = folder_padre + '/Velocidades'

folder_plots_presion = folder_padre + '/Plots_presion'
if os.path.exists(folder_plots_presion):
    shutil.rmtree(folder_plots_presion)
os.makedirs(folder_plots_presion)

folder_plots_velocidad = folder_padre + '/Plots_velocidad'
if os.path.exists(folder_plots_velocidad):
    shutil.rmtree(folder_plots_velocidad)
os.makedirs(folder_plots_velocidad)

#Parámetros simulación
numero_frames = 10_000
fps = 10

param = {
    #Parámetros malla
    'numX': 50,
    'numY': 50,
    'numZ': 170,
    'h': 0.02,
    'dt': np.sqrt(3)*0.02,

    #Parámetros fluido
    'densidad': 1000,
    'flujo': 2.0,
    'gravedad': 0.588, 

    #Parámetros convergencia
    'numIter': 20,
    'overRelaxation': 1.9,

    #Parámetros simulación
    'frames': 1000,
    'plotFrec': 1,
    'skip_first_frames': 0
}

def escalar_ejes(valor, _):
    return f"{valor*param['h']:.1f}"

class Fluid:
    def __init__(self, numX, numY, numZ, h):
        self.X = numX + 2
        self.Y = numY + 2
        self.Z = numZ + 2
        self.x = np.arange(numX+2)
        self.y = np.arange(numY+2)
        self.z = np.arange(numZ+2)

        self.h = h
        self.n = 0 #contador
        self.t = 0 #contador tiempo

        self.vel = np.zeros([self.X, self.Y, self.Z, 3])
        self.presion = np.zeros([self.X, self.Y, self.Z])
        self.New_vel = np.zeros([self.X, self.Y, self.Z, 3])
        self.estado = np.ones([self.X, self.Y, self.Z])

    def contorno(self):
        self.estado[:, :, 0] = 0
        self.estado[:, 0, :] = 0
        self.estado[:, -1, :] = 0
        self.estado[0, :, :] = 0
        self.estado[-1, :, :] = 0

    def obstaculo(self):
        obs_x = self.X / 2
        obs_y = self.Y / 2
        obs_z = self.Z / 5
        obs_r = self.Y / 6

        for i in range(self.X):
            for j in range(self.Y):
                for k in range(self.Z):
                    if np.sqrt((i-obs_x)**2 + (k-obs_z)**2) < obs_r:
                        self.estado[i, j, k] = 0
                        self.vel[i, j, k, :] = 0
                        
    def vel_field(self, x, y, z, field): #field=-1,0,1 indicando vz,vx,vy     
        h1 = 1 / param['h']
        h2 = param['h'] / 2  
        
        if field == 0:
            xf = 0
            yf = 1
            zf = 1
        if field == 1:
            xf = 1
            yf = 0
            zf = 1
        if field == -1:
            xf = 1
            yf = 1
            zf = 0

        x = max(min(x, self.X * self.h), self.h)
        y = max(min(y, self.Y * self.h), self.h)
        z = max(min(z, self.Z *  self.h),  self.h)

        i0 = int(min(np.floor((x - xf*h2)*h1), self.X-1))
        i1 = int(min(i0+1, self.X-1))
        tx = (x - xf*h2)*h1 - i0
        sx = 1 - tx

        j0 = int(min(np.floor((y - yf*h2)*h1), self.Y-1))
        j1 = int(min(j0+1, self.Y-1))
        ty = (y - yf*h2)*h1 - j0
        sy = 1 - ty

        k0 = int(min(np.floor((z - zf*h2)*h1), self.Z-1))
        k1 = int(min(k0+1, self.Z-1))
        tz = (z - zf*h2)*h1 - k0
        sz = 1 - tz

        value = (sx*sy*sz*velocidad[i0, j0, k0, field] + tx*sy*sz*velocidad[i1, j0, k0, field] 
               + sx*ty*sz*velocidad[i0, j1, k0, field] + tx*ty*sz*velocidad[i1, j1, k0, field]
               + sx*sy*tz*velocidad[i0, j0, k1, field] + tx*sy*tz*velocidad[i1, j0, k1, field]
               + sx*ty*tz*velocidad[i0, j1, k1, field] + tx*ty*tz*velocidad[i1, j1, k1, field])
        
        return value

f = Fluid(param['numX'], param['numY'], param['numZ'], param['h'])
f.obstaculo()
f.contorno()

y_planta = int(param['numY']/2)
x_obs_planta, z_obs_planta = np.where(f.estado[:, y_planta, :] == 0)

x_alzado = int(param['numX']/2)
y_obs_alzado, z_obs_alzado = np.where(f.estado[x_alzado, :, :] == 0)

def get_streamlines(i, j, k):
    seg_L = 0.2*param['h']
    seg_Num = 50
    h2 = param['h']* 0.5

    x_0 = i*param['h'] + h2
    y_0 = j*param['h'] + h2
    z_0 = k*param['h'] + h2

    x = [x_0]
    y = [y_0]
    z = [z_0]

    for n in range(seg_Num):
        if (x_0 >= (param['numX']-1)*param['h']) or (x_0 <= 0) or (y_0 >= (param['numY']-1)*param['h']) or (y_0 <= 0) or (z_0 >= (param['numZ']-1)*param['h']) or (z_0 <= 0):
            break 

        u = f.vel_field(x_0, y_0, z_0, 0)
        v = f.vel_field(x_0, y_0, z_0, 1)
        w = f.vel_field(x_0, y_0, z_0, -1)

        l = np.sqrt(u**2 + v**2 + w**2)

        if l == 0:
            break

        x_0 += u * seg_L / l
        y_0 += v * seg_L / l
        z_0 += w * seg_L / l

        if (x_0 > f.X*param['h']) or (y_0 > f.Y*param['h']) or (z_0 > f.Z*param['h']):
            break

        x = np.append(x, x_0)
        y = np.append(y, y_0)
        z = np.append(z, z_0)

    return x, y, z

for dat in tqdm(range(numero_frames)):
    dat += 1
    archivo_presion = os.path.join(folder_presion, 'Presiones_' + str(dat) + '(EF3D).json')
    archivo_velocidad = os.path.join(folder_velocidad, 'Velocidades_' + str(dat) + '(EF3D).json')

    if os.path.exists(archivo_presion) == True:
        with open(archivo_presion, 'r') as archivo:
            pressure_json = json.load(archivo)
        pressure = np.array(pressure_json)

        with open(archivo_velocidad, 'r') as archivo:
            velocidad_json = json.load(archivo)
        velocidad = np.array(velocidad_json)
        vel_magnitude = np.linalg.norm(velocidad, axis=-1, ord=2)

        #Plot_Presion
        fig, axs = plt.subplots(2, 1, sharex= 'row', figsize= (10, 6)) 
            
        iter = (dat - 1)
        tiempo_iter = f"{iter * param['dt']:.2f}"
        fig.suptitle('Método Euler Fluid 3D (t = ' + str(tiempo_iter) + ' s)', fontsize= 20)

            #Plot planta
        presion_planta = pressure[:, y_planta, :]
        
        im1 = axs[0].imshow(presion_planta, origin= 'lower', cmap= 'rainbow', vmin=-3000, vmax=3000)
        axs[0].set_title('Vista de Planta (Y = %i)' %(y_planta), loc='center', fontsize= 10)

        axs[0].scatter(z_obs_planta, x_obs_planta, s= 10, marker='s', color='black')

        for i in range(1, f.X, 5):
            for k in range(1, f.Z, 5):
                if f.estado[i, y_planta, k] == 1 and f.estado[i, y_planta, k-1] == 1 and i < f.X - 1:
                    x, y, z = get_streamlines(i, y_planta, k)
                    z_1 = [z_aux / param['h'] for z_aux in z]
                    x_1 = [x_aux / param['h'] for x_aux in x]
                    axs[0].plot(z_1, x_1, color= 'black', linewidth = 0.5)

            #Plot alzado
        presion_alzado = pressure[x_alzado, :, :]
                    
        im2 = axs[1].imshow(presion_alzado, origin= 'lower', cmap= 'rainbow', vmin=-3000, vmax=3000)
        axs[1].set_title('Vista de Alzado (X = %i)' %(x_alzado), loc='center', fontsize= 10)
            
        axs[1].scatter(z_obs_alzado, y_obs_alzado, s= 10, marker='s', color='black')

        for j in range(1, f.Y, 5):
            for k in range(1, f.Z, 5):
                if f.estado[x_alzado, j, k] == 1 and f.estado[x_alzado,  j, k-1] == 1 and j < f.Y - 1:
                    x, y, z = get_streamlines(x_alzado, j, k)
                    y_2 = [y_aux / param['h'] for y_aux in y]
                    z_2 = [z_aux / param['h'] for z_aux in z]
                    axs[1].plot(z_2, y_2, color= 'black', linewidth = 0.5)

        axs[0].xaxis.set_major_formatter(plt.FuncFormatter(escalar_ejes))
        axs[0].yaxis.set_major_formatter(plt.FuncFormatter(escalar_ejes))

        axs[1].xaxis.set_major_formatter(plt.FuncFormatter(escalar_ejes))
        axs[1].yaxis.set_major_formatter(plt.FuncFormatter(escalar_ejes))

        cbar = fig.colorbar(im1, ax=axs.ravel().tolist())
        cbar.set_label('Presión')

        frame_iter = "{:06d}".format(iter)

        frame_name_presion = os.path.join(folder_plots_presion, 'Frame_' + str(frame_iter))
        fig.savefig(frame_name_presion)
        plt.clf()
        plt.close()

        #Plot_Velocidad
        fig, axs = plt.subplots(2, 1, sharex= 'row', figsize= (10, 6)) 
            
        fig.suptitle('Método Euler Fluid 3D (t = ' + str(tiempo_iter) + ' s)', fontsize= 20)

            #Plot planta
        velocidad_planta = vel_magnitude[:, y_planta, :]

        im1 = axs[0].imshow(velocidad_planta, origin= 'lower', cmap= 'rainbow', vmin = 0, vmax = 1.5 * param['flujo'])
        axs[0].set_title('Vista de Planta (Y = %i)' %(y_planta), loc='center', fontsize= 10)

        axs[0].scatter(z_obs_planta, x_obs_planta, s= 10, marker='s', color='black')

        for i in range(1, f.X, 5):
            for k in range(1, f.Z, 5):
                if f.estado[i, y_planta, k] == 1 and f.estado[i, y_planta, k-1] == 1 and i < f.X - 1:
                    x, y, z = get_streamlines(i, y_planta, k)
                    z_1 = [z_aux / param['h'] for z_aux in z]
                    x_1 = [x_aux / param['h'] for x_aux in x]
                    axs[0].plot(z_1, x_1, color= 'black', linewidth = 0.5)

            #Plot alzado
        velocidad_alzado = vel_magnitude[x_alzado, :, :]
            
        im2 = axs[1].imshow(velocidad_alzado, origin= 'lower', cmap= 'rainbow', vmin = 0, vmax = 1.5 * param['flujo'])
        axs[1].set_title('Vista de Alzado (X = %i)' %(x_alzado), loc='center', fontsize= 10)
            
        axs[1].scatter(z_obs_alzado, y_obs_alzado, s= 10, marker='s', color='black')

        for j in range(1, f.Y, 5):
            for k in range(1, f.Z, 5):
                if f.estado[x_alzado, j, k] == 1 and f.estado[x_alzado,  j, k-1] == 1 and j < f.Y - 1:
                    x, y, z = get_streamlines(x_alzado, j, k)
                    y_2 = [y_aux / param['h'] for y_aux in y]
                    z_2 = [z_aux / param['h'] for z_aux in z]
                    axs[1].plot(z_2, y_2, color= 'black', linewidth = 0.5)

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

#Compilador vídeo Presion
video_name = folder_padre + '/EF3D_presion.avi'
create_video_from_images(folder_plots_presion, video_name, fps)

#Compilador vídeo Velocidad
video_name = folder_padre + '/EF3D_velocidad.avi'
create_video_from_images(folder_plots_velocidad, video_name, fps)
