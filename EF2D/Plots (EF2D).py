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
    'numX': 170,
    'numY': 50,
    'h': 0.02,
    'dt': np.sqrt(3)*0.02,

    #Parámetros fluido
    'densidad': 1000,
    'flujo': 2,

    #Parámetros convergencia
    'numIter': 20,
    'overRelaxation': 1.9,

    #Parámetros simulación
    'frames': 1000,
    'plotFrec': 5,
    'skip_first_frames': 0
}

def escalar_ejes(valor, _):
    return f"{valor * param['h']:.1f}"

#Streamlines
class Fluid:
    def __init__(self, numX, numY, h):
        self.X = numX + 2
        self.Y = numY + 2
        self.x = np.arange(numX+2)
        self.y = np.arange(numY+2)

        self.h = h
        self.n = 0 #contador
        self.dat = 0 #contador
        self.t = 0 #contador tiempo

        self.vel = np.zeros([self.X, self.Y, 2])
        self.presion = np.zeros([self.X, self.Y])
        self.New_vel = np.zeros([self.X, self.Y, 2])
        self.estado = np.ones([self.X, self.Y])

    def contorno(self):
        self.estado[0, :] = 0
        self.estado[:, 0] = 0
        self.estado[:, -1] = 0

    def obstaculo(self):
        obs_x = self.X / 5
        obs_y = self.Y / 2
        obs_r = self.Y / 6

        for i in range(self.X):
            for j in range(self.Y):
                    if np.sqrt((i-obs_x)**2 + (j-obs_y)**2) < obs_r:
                        self.estado[i, j] = 0
                        self.vel[i, j, :] = 0
    
    def vel_field(self, x, y, field): #field= 0,1 indicando vx,vy     
        h1 = 1 / param['h']
        h2 = param['h'] / 2  
        
        if field == 0:
            xf = 0
            yf = 1
        if field == 1:
            xf = 1
            yf = 0

        x = max(min(x, self.X * self.h), self.h)
        y = max(min(y, self.Y * self.h), self.h)
        
        i0 = int(min(np.floor((x - xf*h2)*h1), self.X-1))
        i1 = int(min(i0+1, self.X-1))
        tx = (x - xf*h2)*h1 - i0
        sx = 1 - tx

        j0 = int(min(np.floor((y - yf*h2)*h1), self.Y-1))
        j1 = int(min(j0+1, self.Y-1))
        ty = (y - yf*h2)*h1 - j0
        sy = 1 - ty

        value = (sx*sy*velocidad[i0, j0, field] + tx*sy*velocidad[i1, j0, field] 
               + sx*ty*velocidad[i0, j1, field] + tx*ty*velocidad[i1, j1, field])
        
        return value

f = Fluid(param['numX'], param['numY'], param['h'])
f.obstaculo()
f.contorno()

x_obs, y_obs = np.where(f.estado == 0)

def get_streamlines(i, j):
    seg_L = 0.2*param['h']
    seg_Num = 50
    h2 = param['h']* 0.5

    x_0 = i*param['h'] + h2
    y_0 = j*param['h'] + h2

    x = [x_0]
    y = [y_0]

    for n in range(seg_Num):
        if (x_0 >= (param['numX']-1)*param['h']) or (x_0 <= 0) or (y_0 >= (param['numY']-1)*param['h']) or (y_0 <= 0):
            break 
        
        u = f.vel_field(x_0, y_0, 0)
        v = f.vel_field(x_0, y_0, 1)

        l = np.sqrt(u**2 + v**2)

        if l == 0:
            break

        x_0 += u * seg_L / l
        y_0 += v * seg_L / l

        x = np.append(x, x_0)
        y = np.append(y, y_0)

    return x, y

for dat in tqdm(range(numero_frames)):
    dat += 1
    archivo_presion = os.path.join(folder_presion, 'Presiones_' + str(dat) + '(EF2D).json')
    archivo_velocidad = os.path.join(folder_velocidad, 'Velocidades_' + str(dat) + '(EF2D).json')

    if os.path.exists(archivo_presion) == True:
        with open(archivo_presion, 'r') as archivo:
            pressure_json = json.load(archivo)
        pressure = np.array(pressure_json)

        with open(archivo_velocidad, 'r') as archivo:
            velocidad_json = json.load(archivo)
        velocidad = np.array(velocidad_json)
        vel_magnitude = np.linalg.norm(velocidad, axis=-1, ord=2)

        #Plot_Presion
        fig, ax1 = plt.subplots(1, 1, sharex= 'row', figsize= (10, 6))

        iter = (dat - 1)
        tiempo_iter = f"{iter * param['dt']:.2f}"

        im1 = ax1.imshow(pressure.T, origin='lower', cmap= 'rainbow', vmin=-3000, vmax=3000)
        ax1.set_title('Método Euler Fluid 2D (t = ' + str(tiempo_iter) + ' s)', fontsize= 20)
            
        ax1.scatter(x_obs, y_obs, s= 10, marker='s', color='black')

        #Plot streamlines
        for i in range(1, f.X, 5):
            for j in range(1, f.Y, 5):
                if f.estado[i, j] == 1 and f.estado[i-1, j] == 1 and j <f.Y-1:
                        x, y = get_streamlines(i, j)
                        x_2 = [x_aux / param['h'] for x_aux in x]
                        y_2 = [y_aux / param['h'] for y_aux in y]
                        ax1.plot(x_2, y_2, color= 'black', linewidth = 0.5)
        
        plt.gca().xaxis.set_major_formatter(FuncFormatter(escalar_ejes))
        plt.gca().yaxis.set_major_formatter(FuncFormatter(escalar_ejes))

        cbar = fig.colorbar(im1, ax= ax1, shrink=0.6)
        cbar.set_label('Presión')

        frame_iter = "{:06d}".format(iter)

        file_name_presion = os.path.join(folder_plots_presion, 'Frame_' + str(frame_iter))
        fig.savefig(file_name_presion)
        plt.clf()
        plt.close()

        #Plot_Velocidad
        fig, ax1 = plt.subplots(1, 1, sharex= 'row', figsize= (10, 6))

        im1 = ax1.imshow(vel_magnitude.T, origin='lower', cmap= 'rainbow', vmin = 0, vmax = 1.5 * param['flujo'])
        ax1.set_title('Método Euler Fluid 2D (t = ' + str(tiempo_iter) + ' s)', fontsize= 20)
            
        ax1.scatter(x_obs, y_obs, s= 10, marker='s', color='black')

        #Plot streamlines
        for i in range(1, f.X, 5):
            for j in range(1, f.Y, 5):
                if f.estado[i, j] == 1 and f.estado[i-1, j] == 1 and j <f.Y-1:
                        x, y = get_streamlines(i, j)
                        x_2 = [x_aux / param['h'] for x_aux in x]
                        y_2 = [y_aux / param['h'] for y_aux in y]
                        ax1.plot(x_2, y_2, color= 'black', linewidth = 0.5)

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

#Compilador vídeo Presion
video_name = folder_padre + '/EF2D_presion.avi'
create_video_from_images(folder_plots_presion, video_name, fps)

#Compilador vídeo Velocidad
video_name = folder_padre + '/EF2D_velocidad.avi'
create_video_from_images(folder_plots_velocidad, video_name, fps)
