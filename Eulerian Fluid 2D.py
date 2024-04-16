import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from tqdm import tqdm
import os
import time

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
    'plotFrec': 1,
    'skip_first_frames': 0
}

def escalar_ejes(valor, _):
    return f"{valor*param['h']:.1f}"

class Fluid:
    def __init__(self, numX, numY, h):
        self.X = numX + 2
        self.Y = numY + 2
        self.x = np.arange(numX+2)
        self.y = np.arange(numY+2)

        self.h = h
        self.n = 0 #contador
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

        tiempos = os.path.join("C:/Users/ramon/Desktop/SIMULACIONES_TFG_FÍSICA/Eulerian Fluid 2D", 'Tiempos de simulación (EF2D).txt')
        if self.n == 1:
            open(tiempos, 'w')
    
        open(tiempos, 'a').write('\n Frame %i, %f'%(self.n, tiempo_ejecucion))

        return self.presion, self.vel

f = Fluid(param['numX'], param['numY'], param['h'])
f.flujo()
f.obstaculo()
f.contorno()

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

x_obs, y_obs = np.where(f.estado == 0)

#----------------------- SIMULACIÓN -----------------------

def main():   
    for iter in tqdm(range(param['frames'])):

        pressure, velocidad = f.simulation()
        
        if iter % param['plotFrec'] == 0 and iter >= param['skip_first_frames']:
            
            #Plot
            fig, ax1 = plt.subplots(1, 1, sharex= 'row', figsize= (10, 6))

            tiempo_iter = f"{f.t:.2f}"

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

            frame_name = os.path.join("C:/Users/ramon/Desktop/SIMULACIONES_TFG_FÍSICA/Eulerian Fluid 2D", 'Frame_' + str(frame_iter))
            fig.savefig(frame_name)
            plt.clf()
            plt.close()

if __name__ == '__main__':
    main()