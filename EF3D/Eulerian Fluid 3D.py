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
    'frames': 300,
    'plotFrec': 1,
    'skip_first_frames': 0
}

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
        self.dat = 0 #contador
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
                        
    def flujo(self):
        self.vel[:, :, 1, -1] = param['flujo']    
        
    def integrate(self):
        for i in range(self.X):
            for j in range(self.Y-1):
                for k in range(self.Z):
                    if self.estado[i,j,k] != 0 and self.estado[i, j-1, k] != 0:
                        self.vel[i, j, k, 1] -= param['gravedad'] * param['dt']

    def incompressibility(self):
        cp = param['densidad'] * param['h'] / param['dt']
        for iter in range(param['numIter']):
            for i in range(1, self.X-1):
                for j in range(1, self.Y-1):
                    for k in range(1, self.Z-1):
                        if self.estado[i,j,k] == 0:
                            continue
                        
                        sx0 = self.estado[i-1, j, k]
                        sx1 = self.estado[i+1, j, k]
                        sy0 = self.estado[i, j-1, k]
                        sy1 = self.estado[i, j+1, k]
                        sz0 = self.estado[i, j, k-1]
                        sz1 = self.estado[i, j, k+1]

                        s = sx0 + sx1 + sy0 + sy1 + sz0 + sz1

                        if s == 0:
                            continue
                        
                        div = (self.vel[i+1, j, k, 0] - self.vel[i, j, k, 0]
                                  + self.vel[i, j+1, k, 1] - self.vel[i, j, k, 1]
                                  + self.vel[i, j, k+1, -1] - self.vel[i, j, k, -1])
                        
                        p = - param['overRelaxation'] * div/ s
                        self.presion[i,j,k] += cp*p

                        self.vel[i, j, k, 0] -= sx0 * p
                        self.vel[i, j, k, 1] -= sy0 * p
                        self.vel[i, j, k, -1] -= sz0 * p

                        self.vel[i+1, j, k, 0] += sx1 * p
                        self.vel[i, j+1, k, 1] += sy1 * p
                        self.vel[i, j, k+1, -1] += sz1 * p 

    def extrapolate(self):
         #Contorno eje X
        self.vel[0, :, :, 1] = self.vel[1, :, :, 1]
        self.vel[-1, :, :, 1] = self.vel[-2, :, :, 1]
        self.vel[0, :, :, -1] = self.vel[1, :, :, -1]
        self.vel[-1, :, :, -1] = self.vel[-2, :, :, -1]
         #Contorno eje Y
        self.vel[:, 0, :, 0] = self.vel[:, 1, :, 0]
        self.vel[:, -1, :, 0] = self.vel[:, -2, :, 0]
        self.vel[:, 0, :, -1] = self.vel[:, 1, :, -1]
        self.vel[:, -1, :, -1] = self.vel[:, -2, :, -1]
         #Contorno eje Z
        self.vel[:, :, 0, 0] = self.vel[:, :, 1, 0]
        self.vel[:, :, -1, 0] = self.vel[:, :, -2, 0]
        self.vel[:, :, 0, 1] = self.vel[:, :, 1, 1]
        self.vel[:, :, -1, 1] = self.vel[:, :, -2, 1]

    def average(self, i, j, k, Avg, Advec): 
        #Advec: Campo de Advección; -1, 0, 1 ---> vz, vx, vy
        #Avg: Campo que se promedia para la Advección; -1, 0, 1 ---> vz, vx, vy

        if Advec == 0:
            if Avg == 1:
                v = 0.25*(self.vel[i,j,k, Avg] + self.vel[i-1, j, k, Avg] + self.vel[i-1, j+1, k, Avg] + self.vel[i, j+1, k, Avg])
                return v
            if Avg == -1:
                w = 0.25*(self.vel[i,j,k, Avg] + self.vel[i-1, j, k, Avg] + self.vel[i-1, j, k+1, Avg] + self.vel[i, j, k+1, Avg])
                return w
        if Advec == 1:
            if Avg == 0:
                u = 0.25*(self.vel[i,j,k, Avg] + self.vel[i, j-1, k, Avg] + self.vel[i+1, j-1, k, Avg] + self.vel[i+1, j, k, Avg])
                return u
            if Avg == -1:
                w = 0.25*(self.vel[i,j,k, Avg] + self.vel[i, j-1, k, Avg] + self.vel[i, j-1, k+1, Avg] + self.vel[i, j, k+1, Avg])
                return w
        if Advec == -1:
            if Avg == 0:
                u = 0.25*(self.vel[i,j,k, Avg] + self.vel[i, j, k-1, Avg] + self.vel[i+1, j, k-1, Avg] + self.vel[i+1, j, k, Avg])
                return u
            if Avg == 1:
                v = 0.25*(self.vel[i,j,k, Avg] + self.vel[i, j, k-1, Avg] + self.vel[i, j+1, k-1, Avg] + self.vel[i, j+1, k, Avg])
                return v

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

        value = (sx*sy*sz*self.vel[i0, j0, k0, field] + tx*sy*sz*self.vel[i1, j0, k0, field] 
               + sx*ty*sz*self.vel[i0, j1, k0, field] + tx*ty*sz*self.vel[i1, j1, k0, field]
               + sx*sy*tz*self.vel[i0, j0, k1, field] + tx*sy*tz*self.vel[i1, j0, k1, field]
               + sx*ty*tz*self.vel[i0, j1, k1, field] + tx*ty*tz*self.vel[i1, j1, k1, field])
        
        return value

    def advection(self):
        x_h = self.x*self.h
        x_h2 = (self.x+0.5)*self.h
        y_h = self.y*self.h
        y_h2 = (self.y+0.5)*self.h
        z_h = self.z*self.h
        z_h2 = (self.z+0.5)*self.h
        
        NEW_vel = np.copy(self.vel)
        
        for i in range(1, self.X):
            for j in range(1, self.Y):
                for k in range(1, self.Z):
                    #Componente eje X
                    if self.estado[i,j,k] != 0 and self.estado[i-1, j, k] != 0 and j < self.Y-1 and k < self.Z-1:
                        u = self.vel[i,j,k, 0]
                        v = self.average(i,j,k, 1, 0)
                        w = self.average(i,j,k, -1, 0)

                        x = x_h[i] - u*param['dt']
                        y = y_h2[j] - v*param['dt']
                        z = z_h2[k] - w*param['dt']
                        
                        u = self.vel_field(x, y, z, 0)
                        NEW_vel[i,j,k, 0] = u

                    #Componente eje Y
                    if self.estado[i,j,k] != 0 and self.estado[i, j-1, k] != 0 and i < self.X-1 and k < self.Z-1:
                        u = self.average(i,j,k, 0, 1)
                        v = self.vel[i,j,k, 1]
                        w = self.average(i,j,k, -1, 1)

                        x = x_h2[i] - u*param['dt']
                        y = y_h[j] - v*param['dt']
                        z = z_h2[k] - w*param['dt']

                        v = self.vel_field(x, y, z, 1)
                        NEW_vel[i,j,k, 1] = v

                    #Componente eje Z
                    if self.estado[i,j,k] != 0 and self.estado[i, j, k-1] != 0 and i < self.X-1 and j < self.Y-1:
                        u = self.average(i,j,k, 0, -1)
                        v = self.average(i,j,k, 1, -1)
                        w = self.vel[i,j,k, -1]

                        x = x_h2[i] - u*param['dt']
                        y = y_h2[j] - v*param['dt']
                        z = z_h[k] - w*param['dt']

                        w = self.vel_field(x, y, z, -1)
                        NEW_vel[i,j,k, -1] = w

        self.vel = np.copy(NEW_vel)

    def simulation(self):
        inicio = time.time()
        
        self.integrate()
        self.presion = np.zeros([self.X, self.Y, self.Z])
        self.incompressibility()
        self.extrapolate()
        self.advection()

        final = time.time()
        tiempo_ejecucion = final - inicio
        self.n += 1
        self.t += param['dt']

        tiempos = os.path.join(folder_padre, 'Tiempos de simulación (EF3D).txt')
        if self.n == 1:
            open(tiempos, 'w')
    
        open(tiempos, 'a').write('\n Frame %i, %f'%(self.n, tiempo_ejecucion))

        return self.presion, self.vel

f = Fluid(param['numX'], param['numY'], param['numZ'], param['h'])
f.flujo()
f.obstaculo()
f.contorno()

#----------------------- SIMULACIÓN -----------------------

def main():   
    for iter in tqdm(range(param['frames'])):
        
        pressure, velocidad = f.simulation()

        if iter % param['plotFrec'] == 0 and iter >= param['skip_first_frames']:
            f.dat += 1

            file_name_presion = os.path.join(folder_presion, 'Presiones_' + str(f.dat) + '(EF3D).json')
            file_name_velocidad = os.path.join(folder_velocidad, 'Velocidades_' + str(f.dat) + '(EF3D).json')

            density_lista = pressure.tolist()
            with open(file_name_presion, 'w') as archivo:
                json.dump(density_lista, archivo)

            velocidad_lista = velocidad.tolist()
            with open(file_name_velocidad, 'w') as archivo:
                json.dump(velocidad_lista, archivo)
           
if __name__ == '__main__':
    main()
