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

folder_presion = folder_padre + '/Presiones'
if os.path.exists(folder_presion):
    shutil.rmtree(folder_presion)
os.makedirs(folder_presion)

folder_velocidad = folder_padre + '/Velocidades'
if os.path.exists(folder_velocidad):
    shutil.rmtree(folder_velocidad)
os.makedirs(folder_velocidad)

param = {
    #Parámetros malla
    'numX': 170,
    'numY': 50,
    'h': 0.02,
    'dt': np.sqrt(3)*0.02,

    #Parámetros fluido
    'densidad': 1000,
    'flujo': 2.0,

    #Parámetros convergencia
    'numIter': 20,
    'overRelaxation': 1.9,

    #Parámetros simulación
    'frames': 300,
    'plotFrec': 1,
    'skip_first_frames': 0
}

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
                        

    def flujo(self):
        self.vel[1, :, 0] = param['flujo']

    def incompressibility(self):
        cp = param['densidad'] * param['h'] / param['dt']
        div = np.zeros([self.X, self.Y])
        for iter in range(param['numIter']):
            for i in range(1, self.X-1):
                for j in range(1, self.Y-1):
                        if self.estado[i,j] == 0:
                            continue
                        
                        sx0 = self.estado[i-1, j]
                        sx1 = self.estado[i+1, j]
                        sy0 = self.estado[i, j-1]
                        sy1 = self.estado[i, j+1]

                        s = sx0 + sx1 + sy0 + sy1

                        if s == 0:
                            continue
                        
                        div[i, j] = (self.vel[i+1, j, 0] - self.vel[i, j, 0]
                                  + self.vel[i, j+1, 1] - self.vel[i, j, 1])
                        
                        p = - param['overRelaxation'] * div[i, j] / s
                        self.presion[i,j] += cp*p

                        self.vel[i, j, 0] -= sx0 * p
                        self.vel[i, j, 1] -= sy0 * p
                        self.vel[i+1, j, 0] += sx1 * p
                        self.vel[i, j+1, 1] += sy1 * p

    def extrapolate(self):
         #Contorno eje X
        self.vel[0, :, 1] = self.vel[1, :, 1]
        self.vel[-1, :, 1] = self.vel[-2, :, 1]
         #Contorno eje Y
        self.vel[:, 0, 0] = self.vel[:, 1, 0]
        self.vel[:, -1, 0] = self.vel[:, -2, 0]
        
    def average(self, i, j, Avg, Advec): 
        #Advec: Campo de Advección; 0, 1 ---> vx, vy
        #Avg: Campo que se promedia para la Advección; 0, 1 ---> vx, vy

        if Advec == 0:
            if Avg == 1:
                v = 0.25*(self.vel[i,j, Avg] + self.vel[i-1, j, Avg] + self.vel[i-1, j+1, Avg] + self.vel[i, j+1, Avg])
                return v
        if Advec == 1:
            if Avg == 0:
                u = 0.25*(self.vel[i,j, Avg] + self.vel[i, j-1, Avg] + self.vel[i+1, j-1, Avg] + self.vel[i+1, j, Avg])
                return u

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

        value = (sx*sy*self.vel[i0, j0, field] + tx*sy*self.vel[i1, j0, field] 
               + sx*ty*self.vel[i0, j1, field] + tx*ty*self.vel[i1, j1, field])
        
        return value

    def advection(self):
        x_h = self.x*self.h
        x_h2 = (self.x+0.5)*self.h
        y_h = self.y*self.h
        y_h2 = (self.y+0.5)*self.h
        
        NEW_vel = np.copy(self.vel)
        
        for i in range(1, self.X):
            for j in range(1, self.Y):
                    #Componente eje X
                    if self.estado[i,j] != 0 and self.estado[i-1, j] != 0 and j < self.Y-1:
                        u = self.vel[i,j, 0]
                        v = self.average(i,j, 1, 0)

                        x = x_h[i] - u*param['dt']
                        y = y_h2[j] - v*param['dt']

                        u = self.vel_field(x, y, 0)
                        NEW_vel[i,j, 0] = u

                    #Componente eje Y
                    if self.estado[i,j] != 0 and self.estado[i, j-1] != 0 and i < self.X-1:
                        u = self.average(i,j, 0, 1)
                        v = self.vel[i,j, 1]
                       
                        x = x_h2[i] - u*param['dt']
                        y = y_h[j] - v*param['dt']
                        
                        v = self.vel_field(x, y, 1)
                        NEW_vel[i,j, 1] = v

        self.vel = np.copy(NEW_vel)

    def simulation(self):
        inicio = time.time()
        
        self.presion = np.zeros([self.X, self.Y])
        self.incompressibility()
        self.extrapolate()
        self.advection()

        final = time.time()
        tiempo_ejecucion = final - inicio
        self.n += 1
        self.t += param['dt']

        tiempos = os.path.join(folder_padre, 'Tiempos de simulación (EF2D).txt')
        if self.n == 1:
            open(tiempos, 'w')
    
        open(tiempos, 'a').write('\n Frame %i, %f'%(self.n, tiempo_ejecucion))

        return self.presion, self.vel

f = Fluid(param['numX'], param['numY'], param['h'])
f.flujo()
f.obstaculo()
f.contorno()

#----------------------- SIMULACIÓN -----------------------

def main():   
    for iter in tqdm(range(param['frames'])):

        pressure, velocidad = f.simulation()
        
        if iter % param['plotFrec'] == 0 and iter >= param['skip_first_frames']:
            f.dat += 1
            
            file_name_presion = os.path.join(folder_presion, 'Presiones_' + str(f.dat) + '(EF2D).json')
            file_name_velocidad = os.path.join(folder_velocidad, 'Velocidades_' + str(f.dat) + '(EF2D).json')

            density_lista = pressure.tolist()
            with open(file_name_presion, 'w') as archivo:
                json.dump(density_lista, archivo)

            velocidad_lista = velocidad.tolist()
            with open(file_name_velocidad, 'w') as archivo:
                json.dump(velocidad_lista, archivo)

if __name__ == '__main__':
    main()