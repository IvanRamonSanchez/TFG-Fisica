import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from tqdm import tqdm
import os
import time

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

        tiempos = os.path.join("C:/Users/ramon/Desktop/SIMULACIONES_TFG_FÍSICA/Eulerian Fluid 3D", 'Tiempos de simulación (EF3D).txt')
        if self.n == 1:
            open(tiempos, 'w')
    
        open(tiempos, 'a').write('\n Frame %i, %f'%(self.n, tiempo_ejecucion))

        return self.presion, self.vel

f = Fluid(param['numX'], param['numY'], param['numZ'], param['h'])
f.flujo()
f.obstaculo()
f.contorno()

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

y_planta = int(param['numY']/2)
x_obs_planta, z_obs_planta = np.where(f.estado[:, y_planta, :] == 0)

x_alzado = int(param['numX']/2)
y_obs_alzado, z_obs_alzado = np.where(f.estado[x_alzado, :, :] == 0)

#----------------------- SIMULACIÓN -----------------------

def main():   
    for iter in tqdm(range(param['frames'])):
        pressure, velocidad = f.simulation()
        tiempo_iter = f.t

        if iter % param['plotFrec'] == 0 and iter >= param['skip_first_frames']:
            
            fig, axs = plt.subplots(2, 1, sharex= 'row', figsize= (10, 6)) 
            tiempo_iter = f"{f.t:.2f}"
            fig.suptitle('Método Euler Fluid 3D (t = ' + str(tiempo_iter) + ' s)', fontsize= 20)

            #Plot planta
            presion_planta = pressure[:, y_planta, :]
            velocidad_planta = velocidad[:, y_planta, :, :]

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
            velocidad_alzado = velocidad[x_alzado, :, :, :]
            
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

            frame_name = os.path.join("C:/Users/ramon/Desktop/SIMULACIONES_TFG_FÍSICA/Eulerian Fluid 3D", 'Frame_' + str(frame_iter))
            fig.savefig(frame_name)
            plt.clf()
            plt.close()

if __name__ == '__main__':
    main()