import numpy as np

def make_shepp_logan_image(resolution):
   """Builds some Shepp-Logan data.
   """

   phantomData = np.zeros((resolution, resolution))
   fourier_n = int(resolution/2)

   for j in range(1, 2*fourier_n + 1):
      for k in range(1, 2*fourier_n + 1):
         x = (j-fourier_n-1)/fourier_n
         y = (k-fourier_n-1)/fourier_n
         xi = (x-0.22)*np.cos(0.4*np.pi) + y*np.sin(0.4*np.pi)
         eta = y*np.cos(0.4*np.pi) - (x-0.22)*np.sin(0.4*np.pi)

         z = 0
         if ( (x/0.69)**2 + (y/0.92)**2 ) <= 1:
            z = 2
         
         if ( (x/0.6624)**2 + ((y+.0184)/.874)**2 ) <= 1:
            z = z - 0.98

         if ( (xi/0.31)**2 + (eta/0.11)**2 ) <= 1:
            z = z - 0.8;

         xi = (x + 0.22)*np.cos(0.6*np.pi)+y*np.sin(0.6*np.pi);
         eta = y*np.cos(0.6*np.pi)-(x+0.22)*np.sin(0.6*np.pi);

         if ( (xi/0.41)**2 + (eta/0.16)**2 ) <= 1:
            z = z - 0.8
         
         if ( (x/0.21)**2 + ((y - 0.35)/0.25)**2 ) <= 1:
            z = z + 0.4

         if ( (x/.046)**2 + ((y - 0.1)/.046)**2 ) <= 1:
            z = z + 0.4
         
         if ( (x/.046)**2 + ((y + 0.1)/.046)**2 ) <= 1:
            z = z + 0.4
         
         if ( ((x + 0.08)/.046)**2 + ((y+.605)/.023)**2 ) <= 1:
            z = z + 0.4
         
         if ( (x/.023)**2 + ((y+.605)/.023)**2 ) <= 1:
            z = z + 0.4
         
         if ( ((x-.06)/.023)**2 + ((y+.605)/.046)**2 ) <= 1:
            z = z + 0.4

         phantomData[j-1,k-1] = z

   phantomData = phantomData.T
   phantomData = np.flip(phantomData)
   phantomData = np.fliplr(phantomData)

   return phantomData




def line_mask(L, Ny, Nx):
    """Returns the indicator of the domain in 2D fourier space for the specified line geometry.
    """


    thc = np.linspace(0, np.pi-np.pi/L, L)
    M = np.zeros((Ny,Nx)).astype(bool)

    # Full mask
    for ll in range(1,L+1):
        
        if ((thc[ll-1] <= np.pi/4) or (thc[ll-1] > 3*np.pi/4)):
            yr = np.round( np.tan(thc[ll-1])*np.arange(-Nx/2 + 1, Nx/2, 1) ) + np.floor(Ny/2) + 1
            for nn in range(1, Nx):
                if (yr[nn-1] > 0) and (yr[nn-1] < Ny + 1):
                    M[int(yr[nn-1])-1, nn] = True
        
        else:
            xc = np.round( (1/np.tan(thc[ll-1]))*np.arange(-Ny/2 + 1, Ny/2, 1) ) + np.floor(Nx/2) + 1
            for nn in range(1, Ny):
                if (xc[nn-1] > 0) and (xc[nn-1] < Nx+1):
                    M[nn, int(xc[nn-1])-1] = True

    # Upper half plane mask (not including origin)

    Mh = M.copy()
    Mh[int(np.floor(Ny/2))+1:Ny-1,:] = False
    Mh[int(np.floor(Ny/2)), int(np.floor(Nx/2))+1:Nx-1] = False

    M = np.fft.ifftshift(M)
    mi = np.where(M)
    Mh = np.fft.ifftshift(Mh)
    mhi = np.where(Mh)

    return M, Mh, mi, mhi