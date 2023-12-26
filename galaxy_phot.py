import requests
import numpy as np
import sympy as smp
import pandas as pd
from io import BytesIO
from extinction import fm07
from astropy.io import fits
from astropy.wcs import WCS
from snpy import kcorr, fset
from astropy import units as u
from astropy.table import Table
import matplotlib.pyplot as plt
from scipy.special import gamma
from scipy.signal import convolve 
from scipy.signal import convolve2d
from astroquery.vizier import Vizier
from astropy.cosmology import Planck18
from scipy.odr import Model, Data, ODR
from skimage.measure import EllipseModel
from astropy.coordinates import SkyCoord
from matplotlib.colors import ListedColormap
from astropy.convolution import  Moffat2DKernel
from skimage.measure import label as ConnectRegion
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import curve_fit, fsolve, minimize


Vizier.ROW_LIMIT = 1000
disk_proj = Vizier.get_catalogs('J/A+A/553/A80/tablea1')[0].to_pandas()
disk_dust = Vizier.get_catalogs('J/A+A/553/A80/dexp')[0].to_pandas()

# files from DR2 22/12/2023
df_host = pd.read_csv('dr2/tables/.dataset_creation/host_prop/host_match/ztfdr2_matched_hosts.csv', index_col=0) # host coords
df_coord = pd.read_csv('dr2/tables/ztfdr2_coordinates.csv', index_col=0) # sn coords
df_red = pd.read_csv('dr2/tables/ztfdr2_redshifts.csv', index_col=0) # redshift
df_salt = pd.read_csv('dr2/tables/ztfdr2_salt2_params.csv', index_col=0) # salt, mw
df_class = pd.read_csv('csv_files/ztfdr2_subclassifications.csv', index_col=0) # class
df_mass = pd.read_csv('dr2/tables/ztfdr2_globalhost_prop.csv', index_col=0) # host mass
si_umut = pd.read_csv('csv_files/sample_features.csv') # silicon

df10 = Table.read('fits_files/survey-bricks-dr10-south.fits.gz', format='fits')
df9 = Table.read('fits_files/survey-bricks-dr9-north.fits.gz', format='fits')

df_iso = pd.read_csv('csv_files/galaxy_isophotes_.csv', index_col=0)
df_bd1 = pd.read_csv('csv_files/bulge_pars_.csv', index_col=0)
df_bd2 = pd.read_csv('csv_files/bulge_disk_pars_.csv', index_col=0)
df_bd3 = pd.read_csv('csv_files/bulge_bar_disk_pars_.csv', index_col=0)
df_bd0 = pd.read_csv('csv_files/disk_pars_.csv', index_col=0)

df_E = pd.read_csv('templates/E',delim_whitespace=True,names=['wl','fl'])
df_s0 = pd.read_csv('templates/s0',delim_whitespace=True,names=['wl','fl'])
df_sab =  pd.read_csv('templates/sb',delim_whitespace=True,names=['wl','fl'])
df_sc =  pd.read_csv('templates/sc',delim_whitespace=True,names=['wl','fl'])
df_bulge = fits.open('templates/old_temp/bulge_template.fits')
wl_b, fl_b = df_bulge[1].data['WAVELENGTH'], df_bulge[1].data['FLUX']
df_b = pd.DataFrame(np.column_stack([wl_b, fl_b]), columns=['wl', 'fl'])


class HostGal:
    def __init__(self, verbose):
        self.verbose = verbose
        self.cutout = {}
        self.gal = {}
        self.brick = {}
        self.survey = 'legacy'
 
    def init_query(self, host_name, catalog):
        self.host_name = host_name
        self.sn_name = 'None'

        if catalog == 'virgo':
            query = Vizier.query_object(host_name, catalog='J/AJ/90/1681')
            sc = SkyCoord(ra=query[0]['_RA.icrs'][0], dec=query[0]['_DE.icrs'][0],  unit=(u.hourangle, u.deg))
            self.gal = {'host': [sc.ra.deg, sc.dec.deg], 'z': 0.004, 'sn': [sc.ra.deg, sc.dec.deg],
                        'A': 0, 'z_source': 'None'}

    def init_dr2(self, sn_name):
        self.host_name = 'DR2'
        self.sn_name = sn_name

        z, z_err, z_source = df_red.loc[sn_name]
        mwebv, mwr_v = df_salt[['mwebv','mwr_v']].loc[sn_name]
        sn_ra, sn_dec = df_coord[['ra', 'dec']].loc[sn_name]
        host_ra, host_dec = df_host[['ra', 'dec']].loc[sn_name]
        if np.isnan(host_ra) or np.isnan(host_dec) or (host_dec == -80):
            host_ra, host_dec = sn_ra, sn_dec
            print('no host') if self.verbose else None
        if self.verbose:
            print(f'{sn_name}, ra={host_ra:.3f}, dec={host_dec:.3f}')
            print(f'z={z:.4f}, z_source={z_source}')
        self.gal = {'host': [host_ra, host_dec], 'z': z, 'sn': [sn_ra, sn_dec],
                     'A': mwebv * mwr_v, 'z_err': z_err, 'z_source': z_source}
           
    def get_cutout(self, size, band, scale=0.262):
        ra, dec = self.gal['host']
        layer = 'ls-dr10'
        service = 'https://www.legacysurvey.org/viewer/'
        url = f'{service}fits-cutout?ra={ra}&dec={dec}&layer={layer}&pixscale={scale}&bands={band}&size={size}&subimage'
        res = requests.get(url, timeout=10)
        if len(res.content) > 1000:
            return fits.open(BytesIO(res.content))
        else:
            print('No legacy coverage') if self.verbose else None
            return []

    def flux2mag(self, flux, band, scale, soft=True):
        leg_filters = {'g': 4769.90, 'r': 6370.44 ,'i': 7774.30, 'z': 9154.88} # DECam
        A_ = fm07(np.array([leg_filters[band]]), self.gal['A'])

        def mag_soft(flux, zp):
            b = {'g': 0.9e-10, 'r': 1.2e-10, 'i': 1.8e-10, 'z': 7.4e-10}
            return zp + -2.5/np.log(10) * (np.arcsinh(flux/(2*b[band]*scale**2)) + np.log(b[band])) - A_
        
        def mag_(flux, zp):
            return zp - 2.5*np.log10(flux/scale**2) - A_
        
        zp = 22.5
        self.brick['sky_mag'] = float(mag_soft(self.brick['sky'], zp))
        return mag_soft(flux, zp) if soft else mag_(flux, zp)

    @staticmethod
    def calc_aperature(size, redshift, scale):
        if size == 'z':
            if np.sign(redshift) == -1:
                return 600
            else:
                aperature = 0.07 
                dist = Planck18.angular_diameter_distance(redshift).value
                return min(round(np.rad2deg(aperature/dist)*3600/scale),  600)
        else:
            return size
        
    def construct_image(self, fits_list, size, band):
        def get_data(ind):
            flux, invvar, wcs = fits_list[ind].data, fits_list[ind+1].data, WCS(fits_list[ind].header)
            brick_name = fits_list[ind].header['brick']
            brick_data = {'dr10': df10[df10['brickname'] == brick_name], 'dr9': df9[df9['brickname'] == brick_name]}
            brick_dr = 'dr10' if len(brick_data['dr10']) == 1 else 'dr9'
            df_brick = brick_data[brick_dr]
            brick = {'brickname': brick_name, 'psfsize': df_brick[f'psfsize_{band}'][0], 'psfdepth': df_brick[f'psfdepth_{band}'][0], 'galdepth': df_brick[f'galdepth_{band}'][0],
                        'sky': df_brick[f'cosky_{band}'][0], 'dr': brick_dr}
            return flux, invvar, wcs, brick

        if len(fits_list) <= 1:
            print('no file') if self.verbose else None
            return [], [], [], []
        else:
            data_shapes = np.zeros(len(fits_list))
            for i in range(1, len(fits_list), 2):
                shape_i = fits_list[i].data.shape
                data_shapes[i] = np.sum(shape_i)
                if shape_i == (size, size):
                    return get_data(i)
        
        shape_order = np.argpartition(data_shapes, -2)
        li, si = shape_order[-1], shape_order[-2]
        if data_shapes[li] == data_shapes[si]:
            shape_order = np.argpartition(data_shapes, -3)
            si = shape_order[-3]
        shape_l, shape_s = fits_list[li].data.shape, fits_list[si].data.shape

        vector_offset = np.sign([fits_list[li].header['CRVAL1'] - fits_list[si].header['CRVAL1'], fits_list[li].header['CRVAL2'] - fits_list[si].header['CRVAL2']])
        axis_offset = np.sign([fits_list[li].header['NAXIS1'] - fits_list[si].header['NAXIS1'], fits_list[li].header['NAXIS2'] - fits_list[si].header['NAXIS2']])
        vec = (vector_offset*axis_offset).astype(int)
        attach = (vec*np.array([1, -1]))[np.where(vec!=0)[0]]
        if len(attach) != 1:
            return [],[],[],[]

        slice_dictx = [[shape_s[0]-(size-shape_l[0]), shape_s[0]], [0, size-shape_l[0]]]
        slice_dicty = [[0, size-shape_l[1]], [shape_s[1]-(size-shape_l[1]), shape_s[1]]]
        slice_x = slice(*slice_dictx[(vec[1]+1)//2]) if vec[1] != 0 else slice(None, None)
        slice_y = slice(*slice_dicty[(vec[0]+1)//2]) if vec[0] != 0 else slice(None, None)

        def merge_image(data1, data2):
            return np.concatenate([data1, data2[slice_x, slice_y]][::int(attach)], axis=int(np.where(vec==0)[0]))
        
        flux_l, invvar_l, wcs_l, brick_l = get_data(li)
        flux_s, invvar_s, wcs_s, brick_s = get_data(si)

        flux = merge_image(flux_l, flux_s)
        invvar = merge_image(invvar_l, invvar_s)

        wcs = WCS(naxis=2)
        wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN']
        wcs.wcs.crval = self.gal['host']
        wcs.wcs.crpix = [(size-1)/2, (size-1)/2]
        wcs.wcs.cdelt = [-0.262/3600, 0.262/3600]

        def bmean(key): return (brick_l[key] + brick_s[key])/2
        brick = {'brickname': brick_l['brickname'], 'brickname_merge': brick_s['brickname'], 'psfsize': bmean('psfsize'), 'psfdepth': bmean('psfdepth'), 
                 'galdepth':  bmean('galdepth'), 'sky': bmean('sky'), 'dr': brick_l['dr']}
        return flux, invvar, wcs, brick
        
    def get_image(self, source, size, band, scale=0.262):
        path = f'dr2_fits_{band}/{self.sn_name}.fits'
        output_size = self.calc_aperature(size, self.gal['z'], scale)
        if source == 'query':
            fits_data = self.get_cutout(output_size, band, scale)
        elif source == 'save':
            try:
                fits_data = fits.open(path)
            except FileNotFoundError:
                fits_data = []
        elif source == 'query_save':
            fits_data = self.get_cutout(output_size, band, scale)
            if type(fits_data) == list:
                return len(fits_data)
            else:
                fits_data.writeto(path, overwrite=True)
                results = len(fits_data)
                fits_data.close()
                return results

        flux, invvar, wcs, brick = self.construct_image(fits_data, output_size, band)
        if len(flux) != 0:
            self.brick = brick
            mag = self.flux2mag(flux, band, scale, soft=True)
            mag_raw = self.flux2mag(flux, band, scale, soft=False)
            self.cutout = {'flux': flux, 'mag': mag, 'mag_raw': mag_raw, 'invvar': invvar, 'wcs': wcs, 'scale': scale, 'band': band}
            
            print(brick['dr'], *mag.shape) if self.verbose else None
            fits_data.close()
        
    def plot(self):
        wcs, mag = self.cutout['wcs'], self.cutout['mag']
        (sn_ra, sn_dec) = self.gal['sn']

        fig, ax = plt.subplots(figsize=(7, 6), dpi=100, subplot_kw={'projection': wcs})
        map_ = ax.imshow(mag, cmap='gray')
        c = SkyCoord(ra=sn_ra*u.degree, dec=sn_dec*u.degree, frame='icrs')
        sn_px = wcs.world_to_pixel(c)
        ax.scatter(sn_px[0], sn_px[1], c='cyan', s=40, lw=2, fc='None', ec='cyan', zorder=10)

        fig.colorbar(map_, ax=ax, label=r'mag arcsec$^{-2}$')
        ax.set_ylabel('Declination')
        ax.set_xlabel('Right Ascension')
        ax.set_title(f'{self.host_name}, {self.sn_name}')
        ax.coords[0].set_format_unit(u.deg, decimal=True)
        ax.coords[1].set_format_unit(u.deg, decimal=True)
        return fig, ax
    


class galaxy_decomp:
    def __init__(self, target_name, verbose, source, size='z', catalog='ztf'):
        self.verbose = verbose
        self.name = target_name
        self.gobj = {'g': HostGal(verbose=verbose), 'r': HostGal(verbose=False)}
        self.contours = {'g': {}, 'r': {}}
        if catalog == 'ztf':
            self.gobj['g'].init_dr2(target_name)
            self.gobj['r'].init_dr2(target_name)
        else:
            self.gobj['g'].init_query(target_name, catalog)
            self.gobj['r'].init_query(target_name, catalog)
        self.gobj['g'].get_image(source=source, size=size, band='g', scale=0.262)
        self.gobj['r'].get_image(source=source, size=size, band='r', scale=0.262)
        
        if (len(self.gobj['g'].cutout) == 0) or (len(self.gobj['r'].cutout) == 0):
            print('no image') if verbose else None
            return

        self.image = {'g': self.gobj['g'].cutout['mag'], 'r': self.gobj['r'].cutout['mag']}
        self.invvar = {'g': self.gobj['g'].cutout['invvar'], 'r': self.gobj['r'].cutout['invvar']} 
        self.flux = {'g': self.gobj['g'].cutout['flux'], 'r': self.gobj['r'].cutout['flux']} 
        self.center = (np.array(self.image['g'].shape))/2

    def plot_fit(self, isophotes, band, width=0.1, zoom=False):
        fig, (ax1, ax2) = plt.subplots(figsize=(12, 6), dpi=100, ncols=2)
        mag, wcs = self.image[band], self.gobj[band].cutout['wcs']
        ax2.imshow(mag, cmap='gray', origin='lower')

        (sn_ra, sn_dec) = self.gobj[band].gal['sn']
        c = SkyCoord(ra=sn_ra*u.degree, dec=sn_dec*u.degree, frame='icrs')
        sn_px = wcs.world_to_pixel(c)
        ax2.scatter(sn_px[0], sn_px[1], c='cyan', s=40, lw=2, fc='None', ec='cyan')
  
        targets = self.contours[band].keys() if isophotes == 'all' else isophotes
        for i, iso_i in enumerate(targets):
            params, px_data = self.contours[band][iso_i]
            a, b, theta, n, xc, yc, *err = params
            px_all, px_fit, window = px_data

            norm = np.stack(np.where((mag.T > iso_i-width) & (mag.T < iso_i+width))).T
            ax2.scatter(norm.T[0], norm.T[1], s=2, marker='o', zorder=0, color='red', label=f'{iso_i:.1f} $\pm$ {width}')
            if np.any(px_fit):
                ax1.scatter(px_fit.T[0], px_fit.T[1], s=2, marker='o', zorder=1, color='blue', label=f'{iso_i:.1f} fitted')
            if np.any(px_all):
                ax1.scatter(px_all[0], px_all[1], s=2, marker='o',zorder=0, color='red', label=f'{iso_i:.1f} removed')
           
            if xc != 0:
                self.patch_super_ellipse((a, b, theta, n), (xc, yc), ax1, 'red')
                self.patch_super_ellipse((a, b, theta, n), (xc, yc), ax2, 'lime')
                if zoom:
                    ax1.set_xlim([xc-a-10, xc+a+10])
                    ax2.set_xlim([xc-a-10, xc+a+10])
                    ax1.set_ylim([yc-a-10, yc+a+10])
                    ax2.set_ylim([yc-a-10, yc+a+10])

                else:
                    ax1.set_xlim(*ax2.get_xlim())
                    ax1.set_ylim(*ax2.get_ylim())

        ax1.legend(framealpha=0.8, markerscale=5, loc=2)
        ax2.legend(framealpha=1, markerscale=5, loc=2)
        plt.tight_layout()
        return ax1, ax2
        
    def prep_pixels(self, isophote, window, band, kernal):
        kernal5 = np.array([[ 1,  1,  1, 1, 1],
                            [ 1,  1,  1, 1, 1],
                            [ 1,  1,  1, 1, 1],
                            [ 1,  1,  1, 1, 1],
                            [1,  1,  1, 1, 1]])
        kernal_unit = kernal5 / np.sum(kernal5) if kernal == 'default' else kernal
        convolve_2 = convolve2d(self.image[band], kernal_unit, mode='same')

        convolve_2[:5,:] = 0
        convolve_2[-5:,:] = 0
        convolve_2[:,:5] = 0
        convolve_2[:,-5:] = 0
        
        contour = np.stack(np.where((convolve_2.T > isophote-window) & (convolve_2.T < isophote+window)))
        return contour

    @staticmethod
    def super_ellipse(phi, a_, b_, pa, n, polar=True):
        a, b = max(a_, b_), min(a_, b_)
        r = (np.abs(np.cos(phi-pa)/a)**n + np.abs(np.sin(phi-pa)/b)**n ) **(-1/n)
        if polar:
            return r
        else:
            x = r*np.cos(phi)
            y = r*np.sin(phi)
            return x, y

    def super_ellipse_fitter(self, data, err):
        ell = EllipseModel()
        ell.estimate(data)
        if ell.params is None:
            return np.zeros(4), np.zeros(4), (0,0)
        xc, yc, a, b, pa = ell.params
        x, y = data.T
        r = np.sqrt((x-xc)**2 + (y-yc)**2)
        phi = np.arctan2(y-yc, x-xc)
        out_pars = curve_fit(self.super_ellipse, phi, r, p0=[a, b, pa, 2], maxfev=5000, sigma=err)
        return out_pars[0], np.sqrt(np.diag(out_pars[1])), (xc, yc)

    def patch_super_ellipse(self, pars, center, ax, color, label=None):
        t_r = np.arange(-np.pi/2, np.pi/2, 0.01)+ pars[2]
        xse, yse = self.super_ellipse(t_r, *pars, polar=False) 
        xse_t = np.concatenate([xse, -xse])+ center[0]
        yse_t = np.concatenate([yse, -yse])+ center[1]
        ax.plot(xse_t, yse_t, 'r-', color=color, zorder=10, label=label)
        ax.plot(xse_t[[0, -1]], yse_t[[0, -1]], 'r-', color=color, zorder=10)

    def extract_regions(self, contour, band):
        def get_pixels(region, connect): return np.stack(np.where(connect == region))

        binary_image = np.zeros_like(self.image[band], dtype=np.uint8)
        binary_image[contour[0], contour[1]] = 1
        connect_ = ConnectRegion(binary_image, connectivity=2, background=0)
        region_count = np.asarray(np.unique(connect_, return_counts=True)).T[1:].T
        # galaxy_region = max(region_count.T, key=lambda x: x[1])[0]
        galaxy_region = min(region_count.T, key=lambda x: np.sum(np.mean(get_pixels(x[0], connect_).T , axis=0) - self.center)**2)[0]

        return get_pixels(galaxy_region, connect_).T         
           
    def contour_fit(self, isophote, band, kernal='default'):
        
        def fit_px(window, return_err):
            all_pixels = self.prep_pixels(isophote, window, band, kernal)
            try:
                connect_all = self.extract_regions(all_pixels, band)
                fit_vals = self.image[band].T[connect_all.T[0], connect_all.T[1]]
                fit_invvar = var_frac.T[connect_all.T[0], connect_all.T[1]]
                cuts = (fit_vals > isophote-window) & (fit_vals < isophote+window)
                connect_all, fit_vals, fit_invvar = connect_all[cuts], fit_vals[cuts], fit_invvar[cuts]
                if len(connect_all) < 25:
                    raise ValueError

                def linear(x, a, b): return a*x + b
                slope_corr = curve_fit(linear, fit_vals, fit_invvar)

                px_uncertainity =  fit_invvar/linear(fit_vals, *slope_corr[0])
                pars, pars_err, center = self.super_ellipse_fitter(connect_all, px_uncertainity)
            except (RuntimeError, ValueError, TypeError):
                if return_err:
                    return 1
                else:
                    return [[0,0,0,0,0,0,0,0,0,0,0,0], [[], [], window]]
            else:
                if return_err:
                    return np.sum((pars_err/pars)**2)
                else:
                    return [[*pars, *center, *pars_err, fit_invvar.mean(), fit_vals.mean()], [all_pixels, connect_all, window]]

        var_frac = np.abs(-2.5/np.log(10)*np.sqrt(1/self.invvar[band])/self.flux[band])
        try:
            result = minimize(fit_px, 0.2, args=(True), bounds=[(0.1, 0.5)], method='Powell', tol=0.1)
        except (ValueError, TypeError):
            self.contours[band][isophote] =  [[0,0,0,0,0,0,0,0,0,0,0,0], [[], [], 0]]
        else:
            self.contours[band][isophote] = fit_px(result.x, return_err=False)

    def main_run(self):
        step = 0.2
        sb_lim ={'dr10': {'g': 24.9, 'r': 24.5}, 
                  'dr9':  {'g': 24.3, 'r': 23.7}}
        max_galaxy_sb, min_galaxy_sb = 15, 25
        sb_break_lim = 23
        for band in ['g', 'r']:
            max_iso = min(round(self.gobj[band].brick['psfdepth'], 1), min_galaxy_sb)
            max_iso = sb_lim[self.gobj[band].brick['dr']][band] if max_iso == 0 else max_iso
            for iso in np.arange(max_iso, max_galaxy_sb, -step):
                iso = np.round(iso, 1)
                print(f'{band}-{iso}') if (iso%1==0.0 and self.verbose) else None
                self.contour_fit(iso, band=band)
                if len(self.contours[band][iso][1][1]) == 0:
                    del self.contours[band][iso]
                    if iso < sb_break_lim:
                        break
        
    def load_main_run(self):
        iso_data = df_iso[df_iso['sn_name'] == self.name]
        if (iso_data.iloc[0]['band'] != 'no_image') and (iso_data.iloc[0]['band'] != 'no_contours'):    
            for i in range(len(iso_data)):
                data_i = iso_data.iloc[i]
                self.contours[data_i['band']][np.round(data_i['mag'],1)] = [data_i.values[2:].astype(float)]



class BDdecomp:
    def __init__(self, name, gd):
        self.name = name
        self.contours = gd.contours
        self.image = gd.image
        self.center = gd.center
        self.gobj = gd.gobj
        self.scale = 0.262
        self.trunc_bulge = 5
        self.trunc_bar = 5
        self.decomp = [[[*np.zeros(7)], [*np.zeros(7)]], [[*np.zeros(13)], [*np.zeros(13)]], 
                       [[*np.zeros(17)], [*np.zeros(17)]], [[*np.zeros(6)], [*np.zeros(6)]]]
        mags_g, iso_data_g = self.extract_data('g')
        mags_r, iso_data_r = self.extract_data('r')
        self.mags = {'g': mags_g, 'r': mags_r}
        self.iso_data = {'g': iso_data_g, 'r': iso_data_r}
        if (len(iso_data_g) > 0) and  (len(iso_data_r) > 0):
            self.center, self.iso_stat = self.contour_stats()
        
        self.kernal = self.get_psf_kernals()
        
        
    def get_psf_kernals(self):
        power = 4.765
        fwhm_psf_g = self.gobj['g'].brick['psfsize']
        fwhm_psf_r = self.gobj['r'].brick['psfsize']
        fwhm_psf_g_mean = {'dr10': 1.51, 'dr9': 1.91}[self.gobj['g'].brick['dr']]
        fwhm_psf_r_mean = {'dr10': 1.36, 'dr9': 1.7}[self.gobj['r'].brick['dr']]
        if (fwhm_psf_g == 0) and (fwhm_psf_r == 0):
            fwhm_psf_g, fwhm_psf_r = fwhm_psf_g_mean, fwhm_psf_r_mean
        elif (fwhm_psf_g == 0) and (fwhm_psf_r != 0):
            fwhm_psf_g = fwhm_psf_r + (fwhm_psf_g_mean-fwhm_psf_r_mean)
        elif (fwhm_psf_g != 0) and (fwhm_psf_r == 0):
             fwhm_psf_r = fwhm_psf_g - (fwhm_psf_g_mean-fwhm_psf_r_mean)

        core_g = fwhm_psf_g/(2*np.sqrt(2**(1/power) - 1))/self.scale
        core_r = fwhm_psf_r/(2*np.sqrt(2**(1/power) - 1))/self.scale
        return {'g': Moffat2DKernel(gamma=core_g, alpha=power).array,
                'r': Moffat2DKernel(gamma=core_r, alpha=power).array}


    def extract_data(self, band):
        key_targs = list(self.contours[band].keys())
        if len(key_targs) == 0:
            return [], []
        pars_ = np.array([self.contours[band][iso_key][0] for iso_key in key_targs if len(self.contours[band][iso_key][0]) > 0]).T

        offsets = np.sqrt((pars_[4]-self.center[0])**2 + (pars_[5]-self.center[1])**2)
        cuts = np.where((offsets < 10))

        mags = np.array(key_targs)[cuts]
        pars_ = pars_.T[cuts].T.reshape((12, -1))

        if len(pars_) == 0:
            return [], []

        return mags, pars_
    
    def contour_stats(self):
        centerx = np.median(np.concatenate([self.iso_data['g'][4], self.iso_data['r'][4]]))
        centery = np.median(np.concatenate([self.iso_data['g'][5], self.iso_data['r'][5]]))
        pa = np.median(np.concatenate([self.iso_data['g'][2], self.iso_data['r'][2]]))

        offsets_g = np.sqrt((self.iso_data['g'][4]-centerx)**2 + (self.iso_data['g'][5]-centery)**2)
        offsets_r = np.sqrt((self.iso_data['r'][4]-centerx)**2 + (self.iso_data['r'][5]-centery)**2)
        self.iso_data['g'] = (self.iso_data['g'].T[(offsets_g < 5)]).T
        self.iso_data['r'] = (self.iso_data['r'].T[(offsets_r < 5)]).T

        return [centerx, centery], {'pa': pa}
    
    @staticmethod
    def super_ellipse(phi, a_, b_, pa, n, polar=True):
        a, b = max(abs(a_), abs(b_)), min(abs(a_), abs(b_))
        r = (np.abs(np.cos(phi-pa)/a)**n + np.abs(np.sin(phi-pa)/b)**n )**(-1/n)
        if polar:
            return r
        else:
            x = r*np.cos(phi)
            y = r*np.sin(phi)
            return x, y

    def patch_super_ellipse(self, pars, center, ax, color, label=None, lw=1):
        t_r = np.arange(-np.pi/2, np.pi/2, 0.01)+ pars[2]
        xse, yse = self.super_ellipse(t_r, *pars, polar=False) 
        xse_t = np.concatenate([xse, -xse])+ center[0]
        yse_t = np.concatenate([yse, -yse])+ center[1]
        ax.plot(xse_t, yse_t, 'r-', color=color, zorder=6, label=label ,lw=lw)
        ax.plot(xse_t[[0, -1]], yse_t[[0, -1]], 'r-', color=color, zorder=10, lw=lw)
    
    def plot_iso(self, band):
        pars_ = self.iso_data[band]
        fig, ax = plt.subplots(figsize=(12, 10), ncols=2, nrows=3, dpi=100)
        ax1, ax2, ax3, ax4, ax5, ax6 = ax.flatten()

        ecc = 1-pars_[1]/pars_[0]
        ax1.plot(pars_[0]*0.262, ecc, 'r.')
        ax1.set_ylabel('ellipticity')
        ax1.set_xlabel('R [arcsec]')
        ax1.set_ylim([0, 1])

        ax2.plot(pars_[0]*0.262, pars_[4]+0.02, 'b.', label='center x')
        ax2.plot(pars_[0]*0.262, pars_[5]-0.02, 'r.', label='center y')
        ax2.axhline(self.center[0]+0.02, c='b', zorder=0)
        ax2.axhline(self.center[1]-0.02, c='r', zorder=0)
        ax2.set_xlabel('R [arcsec]')
        ax2.legend()

        ax3.plot(pars_[0]*0.262, np.rad2deg(pars_[2]), 'b.')
        ax3.axhline(np.rad2deg(self.iso_stat['pa']), c='r', zorder=0)
        ax3.set_ylim([0, 180])
        ax3.set_xlabel('R [arcsec]')
        ax3.set_ylabel('Position angle [deg]')

        ax4.plot(pars_[0]*0.262, np.sqrt(pars_[6]*pars_[7])*0.262, 'b.', label='fit error')
        ax4.plot(pars_[0]*0.262, pars_[10], 'r.', label='mag error')
        # ax4.plot(pars_[0]*0.262, pars_[6]/pars_[0], 'g.', label='fractional fit error', c='lime')
        ax4.legend()
        ax4.set_xlabel('R [arcsec]')

        ax5.plot(pars_[0]*0.262, pars_[3], 'b.')
        ax5.set_ylim([0, 5])
        ax5.set_xlabel('R [arcsec]')
        ax5.set_ylabel('Super ellipse n')

        plt.tight_layout()
        return fig, ax
    
    @staticmethod
    def get_b(n):
        return 2*n - 1/3 + 4/(405*n) + 46/(25515*n**2)

    def transform(self, u0, h, n):
        b = self.get_b(n)
        return u0 + 2.5*b/np.log(10), b**n*h
    
    def back_transform(self, ue, Re, n):
        b = self.get_b(n)
        return ue - 2.5*b/np.log(10), Re/b**n

    @staticmethod
    def bulge(x, ue, Re, n):
        b = BDdecomp.get_b(n)
        return ue + 2.5*b/np.log(10) * ((np.abs(x)/np.abs(Re))**(1/n) - 1)
    
    @staticmethod
    def disk(x, u0, h):
        return u0 + 2.5/np.log(10)*(np.abs(x)/np.abs(h))

    @staticmethod
    def add_mag(m1, m2):
        return -2.5*np.log10(10**(-0.4*m1) + 10**(-0.4*m2))

    @staticmethod
    def fit_combine(x, ue, u0, Re, h, n):
        trunc_radius = 5
        condlist = [(x <= np.abs(Re)*trunc_radius), (x > np.abs(Re)*trunc_radius)]
        choicelist = [BDdecomp.add_mag(BDdecomp.bulge(x, ue, Re, n), BDdecomp.disk(x, u0, h)), BDdecomp.disk(x, u0, h)]
        return np.select(condlist, choicelist)
    
    @staticmethod
    def combine(x, ue, u0, Re, h, n):
        return BDdecomp.add_mag(BDdecomp.bulge(x, ue, Re, n), BDdecomp.disk(x, u0, h))
    
    @staticmethod
    def combine_3(x, ue_bulge, ue_bar, u0, Re_bulge, Re_bar, h, n_bulge, n_bar):
        bar_bulge = BDdecomp.add_mag(BDdecomp.bulge(x, ue_bulge, Re_bulge, n_bulge), BDdecomp.bulge(x, ue_bar, Re_bar, n_bar))
        return BDdecomp.add_mag(bar_bulge, BDdecomp.disk(x, u0, h))

    @staticmethod
    def fit_combine_3(x, ue_bulge, ue_bar, u0, Re_bulge, Re_bar, h, n_bulge, n_bar):
        trunc_r_bulge, trunc_r_bar = 5*np.abs(Re_bulge), 5*np.abs(Re_bar)
        condlist = [(x <= trunc_r_bulge) & (x <= trunc_r_bar), (x <= trunc_r_bulge) & (x > trunc_r_bar), 
                    (x > trunc_r_bulge) & (x <= trunc_r_bar), (x > trunc_r_bulge) & (x > trunc_r_bar)]
        choicelist = [BDdecomp.add_mag(BDdecomp.add_mag(BDdecomp.bulge(x, ue_bulge, Re_bulge, n_bulge), BDdecomp.bulge(x, ue_bar, Re_bar, n_bar)),  BDdecomp.disk(x, u0, h)),
                       BDdecomp.add_mag(BDdecomp.bulge(x, ue_bulge, Re_bulge, n_bulge),  BDdecomp.disk(x, u0, h)),
                       BDdecomp.add_mag(BDdecomp.bulge(x, ue_bar, Re_bar, n_bar),  BDdecomp.disk(x, u0, h)),
                       BDdecomp.disk(x, u0, h)]
        return np.select(condlist, choicelist)

    def stabilize(self, phi, theta, center0, center1, a, b, pa, n):
        vec = center1 - center0
        beta = np.arctan2(vec[1], vec[0])
        v_mag = np.sqrt(vec[0]**2 + vec[1]**2)
        r =  self.super_ellipse(phi, a, b, pa, n)
        return v_mag * np.sin(theta - beta) - r * np.sin((phi - theta))

    def target_angle(self, c_r, theta):
        target_ang = np.zeros(len(c_r))
        theta = theta 
        for i, row_i in enumerate(c_r):
            ai, bi, pai, ni, xci, yci, *errs = row_i
            phi_i = fsolve(self.stabilize, theta, args=(theta, self.center, np.array([xci, yci]), ai, bi, pai, ni))
            target_ang[i] =  self.super_ellipse(phi_i, ai, bi, pai, ni)
        return target_ang

    def plot_gal_iso(self, spokes, zoom=True):
        fig, axis = plt.subplots(figsize=(12, 6), dpi=100, ncols=2)
        theta_arr =  {'g': np.linspace(0, 360-360/spokes, spokes), 'r': np.linspace(180/spokes, 360-180/spokes, spokes)}
        for band, ax in zip(['g', 'r'], axis):
            sky_mag = self.gobj[band].cutout['mag_raw'].copy()
            sky_lim = self.gobj[band].brick['psfdepth']
            sky_mag[np.isnan(sky_mag)] = sky_lim
            ci = np.round(self.center).astype(int)
            mag_c = min(19, np.mean(self.image[band][ci[0]-1:ci[0]+2, ci[1]-1:ci[1]+2]))
            ax.imshow(sky_mag, origin='lower', cmap='gray', vmax=30, vmin=mag_c)
        
            a_lim = self.iso_data[band][0].max()
            min_i = np.argmin(self.iso_data[band][0])
            xc_ref, yc_ref = self.iso_data[band][4][min_i], self.iso_data[band][5][min_i]
            ax.set_xlim([xc_ref-a_lim-10, xc_ref+a_lim+10]) if zoom else None
            ax.set_ylim([yc_ref-a_lim-10, yc_ref+a_lim+10]) if zoom else None

            for theta in theta_arr[band]:
                for row_i in self.iso_data[band].T[:1:-2]:
                    ai, bi, pai, ni, xci, yci, *errs = row_i
                    if theta is not None:
                        phi = fsolve(self.stabilize, np.deg2rad(theta), args=(np.deg2rad(theta), self.center, 
                                                                            np.array([xci, yci]), ai, bi, pai, ni))
                        xp, yp = self.super_ellipse(phi, ai, bi, pai, ni, polar=False)
                        ax.scatter(xp+xci, yp+yci, color='blue', s=8, zorder=15)
                    color= 'lime' if band=='g' else 'red'
                    self.patch_super_ellipse((ai, bi, pai, ni), (xci, yci), ax, color)
        return fig, ax
                
    @staticmethod
    def alpha_colormap():
        cmap = np.zeros([256, 4])
        cmap[:, 3] =  np.linspace(1, 0, 256)
        cmap_red, cmap_green, cmap_blue = cmap.copy(), cmap.copy(), cmap.copy()
        cmap_red.T[0] = 1
        cmap_green.T[1] = 1
        cmap_blue.T[2] = 1
        return  ListedColormap(cmap_red), ListedColormap(cmap_green), ListedColormap(cmap_blue)
    
    def get_meshgrid(self):
        n_size = np.arange(len(self.image['g']))
        xm, ym = np.meshgrid(n_size, n_size)
        rm = np.linalg.norm(np.stack([xm, ym]).T - self.center[::-1], axis=2)

        theta_top = np.arccos(np.dot(np.stack([xm, ym]).T - self.center[::-1], np.array([0,1]))/rm)[round(self.center[1]):]
        theta_bot = np.arccos(np.dot(np.stack([xm, ym]).T - self.center[::-1], np.array([0,-1]))/rm)[:round(self.center[1])]
        thetam = np.vstack([theta_bot, theta_top])
        thetam[np.isnan(thetam)] = 0
        return xm, ym, rm, thetam
    
    @staticmethod
    def error_prop_SE(theta_, a_, b_, pa_, n_, d_a_, d_b_, d_pa_, d_n_):
        theta, a, b, pa, n, d_a, d_b, d_pa, d_n = smp.symbols('theta, a, b, phi, n, da, db, dphi, dn', positive=True, real=True)
        r = (smp.Abs(smp.cos(theta-pa)/a)**n + smp.Abs(smp.sin(theta-pa)/b)**n ) ** (-1/n)
        err_expr = smp.sqrt((r.diff(a)*d_a)**2 + (r.diff(b)*d_b)**2 + (r.diff(pa)*d_pa)**2 + (r.diff(n)*d_n)**2)
        err_func = smp.lambdify([theta, a, b, pa, n, d_a, d_b, d_pa, d_n], err_expr)
        return err_func(theta_, a_, b_, pa_, n_, d_a_, d_b_, d_pa_, d_n_)


    def create_data(self, spokes, center=False):
        angle_targets = {'g': np.linspace(0, 360-360/spokes, spokes), 'r': np.linspace(180/spokes, 360-180/spokes, spokes)}
        x_data_g = np.concatenate([self.target_angle(self.iso_data['g'].T, np.deg2rad(ang_i)) for ang_i in angle_targets['g']]) * 0.262
        x_err_g = np.concatenate([self.error_prop_SE(ang_i, *[self.iso_data['g'][i] for i in [0,1,2,3,6,7,8,9]]) for ang_i in angle_targets['g']]) * 0.262
        y_data_g = np.tile(self.iso_data['g'][11], spokes)
        y_err_g = np.tile(self.iso_data['g'][10], spokes)
        
        x_data_r = np.concatenate([self.target_angle(self.iso_data['r'].T, np.deg2rad(ang_i)) for ang_i in angle_targets['r']]) * 0.262
        x_err_r = np.concatenate([self.error_prop_SE(ang_i, *[self.iso_data['r'][i] for i in [0,1,2,3,6,7,8,9]]) for ang_i in angle_targets['r']]) * 0.262
        y_data_r = np.tile(self.iso_data['r'][11], spokes)
        y_err_r = np.tile(self.iso_data['r'][10], spokes)

        if center: 
            ci = np.round(self.center).astype(int)
            cmag_g = self.image['g'][ci[1]-1:ci[1]+2, ci[0]-1:ci[0]+2]
            cmag_r = self.image['r'][ci[1]-1:ci[1]+2, ci[0]-1:ci[0]+2]
            x_, x_err = 0.262, 0.262/3
            y_g, y_g_err = np.mean(cmag_g), np.std(cmag_g)
            y_r, y_r_err = np.mean(cmag_r), np.std(cmag_r)
            g_len, r_len = len(x_data_g.reshape(spokes, -1)[0]), len(x_data_r.reshape(spokes, -1)[0])

            x_data_g = np.insert(x_data_g.reshape(spokes, -1), g_len, x_, axis=1).flatten()
            x_err_g = np.insert(x_err_g.reshape(spokes, -1), g_len, x_err, axis=1).flatten()
            y_data_g = np.insert(y_data_g.reshape(spokes, -1), g_len, y_g, axis=1).flatten()
            y_err_g = np.insert(y_err_g.reshape(spokes, -1), g_len, y_g_err, axis=1).flatten()

            x_data_r = np.insert(x_data_r.reshape(spokes, -1), r_len, x_, axis=1).flatten()
            x_err_r = np.insert(x_err_r.reshape(spokes, -1), r_len, x_err, axis=1).flatten()
            y_data_r = np.insert(y_data_r.reshape(spokes, -1), r_len, y_r, axis=1).flatten()
            y_err_r = np.insert(y_err_r.reshape(spokes, -1), r_len, y_r_err, axis=1).flatten()

        x_data = np.concatenate([x_data_g, x_data_r])
        y_data = np.concatenate([y_data_g, y_data_r])
        x_err = np.concatenate([x_err_g, x_err_r])
        y_err = np.concatenate([y_err_g, y_err_r])

        return {'g': [x_data_g, y_data_g, x_err_g, y_err_g], 'r': [x_data_r, y_data_r, x_err_r, y_err_r], 'all': [x_data, y_data, x_err, y_err]}


    def fit_functions(self, spokes, psf=True, truncate=True):
        xm, ym, rm, thetam = self.get_meshgrid()
        angle_targets = {'g': np.linspace(0, 360-360/spokes, spokes), 'r': np.linspace(180/spokes, 360-180/spokes, spokes)}

        def psf_convolve(model):
            model_grid_g = model(rm*0.262, thetam, 'g')
            model_grid_r = model(rm*0.262, thetam, 'r')
            model_psf_g = -2.5*np.log10(convolve(10**(-0.4*model_grid_g), self.kernal['g'], mode='same'))
            model_psf_r = -2.5*np.log10(convolve(10**(-0.4*model_grid_r), self.kernal['r'], mode='same'))
            interp_g = RegularGridInterpolator((xm[0]*0.262, ym.T[0]*0.262), model_psf_g.T)
            interp_r = RegularGridInterpolator((xm[0]*0.262, ym.T[0]*0.262), model_psf_r.T)
            return {'g': interp_g, 'r': interp_r}
        
        def model_psf(r, theta, band, psf_interp):
            x, y = r*np.cos(np.deg2rad(theta))+self.center[0]*0.262, r*np.sin(np.deg2rad(theta))+self.center[1]*0.262
            return psf_interp[band]((x, y))
        
        def generate_data(model, data):
            if psf:
                psf_interp = psf_convolve(model)
                data_g = np.concatenate([model_psf(data['g'].reshape(spokes, -1)[i], angle_targets['g'][i], 'g', psf_interp) for i in range(spokes)])
                data_r = np.concatenate([model_psf(data['r'].reshape(spokes, -1)[i], angle_targets['r'][i], 'r', psf_interp) for i in range(spokes)])
            else:
                data_g = np.concatenate([model(data['g'].reshape(spokes, -1)[i], np.deg2rad(angle_targets['g'][i]), 'g') for i in range(spokes)])
                data_r = np.concatenate([model(data['r'].reshape(spokes, -1)[i], np.deg2rad(angle_targets['r'][i]), 'r') for i in range(spokes)])
            return np.concatenate([data_g, data_r])
        
        def disk_2D_model(pars, all_data):
            rp = 2
            u0_g, u0_r = pars[:rp]
            u0 = {'g': u0_g, 'r': u0_r}
            data = {'g': all_data[:len(self.iso_data['g'].T)*spokes], 'r': all_data[len(self.iso_data['g'].T)*spokes:]}
            def model(x, theta, band):
                h = self.super_ellipse(theta, *pars[rp:rp+4])
                return self.disk(x, u0[band], h)
            
            return generate_data(model, data)

        def bulge_2D_model(pars, all_data):
            rp = 3
            ue_g, ue_r, n = pars[:rp]
            ue = {'g': ue_g, 'r': ue_r}
            data = {'g': all_data[:len(self.iso_data['g'].T)*spokes], 'r': all_data[len(self.iso_data['g'].T)*spokes:]}
            def model(x, theta, band):
                Re = self.super_ellipse(theta, *pars[rp:rp+4])
                return self.bulge(x, ue[band], Re, n)
            
            return generate_data(model, data)

        def bulge_disk_2D_model(pars, all_data):
            rp = 5
            ue_g, ue_r, u0_g, u0_r, n = pars[:rp]
            ue = {'g': ue_g, 'r': ue_r}
            u0 = {'g': u0_g, 'r': u0_r}
            data = {'g': all_data[:len(self.iso_data['g'].T)*spokes], 'r': all_data[len(self.iso_data['g'].T)*spokes:]}
            def model(x, theta, band):
                Re = self.super_ellipse(theta, *pars[rp:rp+4])
                h = self.super_ellipse(theta, *pars[rp+4:rp+8])
                if truncate:
                    return self.fit_combine(x, ue[band], u0[band], Re, h, n)
                else:
                    return self.combine(x, ue[band], u0[band], Re, h, n)
            
            return generate_data(model, data)
        
        def bulge_bar_disk_2D_model(pars, all_data):
            rp = 9
            ue_bulge_g, ue_bulge_r, ue_bar_g, ue_bar_r, u0_g, u0_r, n_bulge, n_bar, Re_bulge = pars[:rp]
            ue_bulge = {'g': ue_bulge_g, 'r': ue_bulge_r}
            ue_bar = {'g': ue_bar_g, 'r': ue_bar_r}
            u0 = {'g': u0_g, 'r': u0_r}
            data = {'g': all_data[:len(self.iso_data['g'].T)*spokes], 'r': all_data[len(self.iso_data['g'].T)*spokes:]}
            def model(x, theta, band):
                Re_bar = self.super_ellipse(theta, *pars[rp:rp+4])
                h = self.super_ellipse(theta, *pars[rp+4:rp+8])
                if truncate:
                    return self.fit_combine_3(x, ue_bulge[band], ue_bar[band], u0[band], Re_bulge, Re_bar, h, n_bulge, n_bar)
                else:
                    return self.combine_3(x, ue_bulge[band], ue_bar[band], u0[band], Re_bulge, Re_bar, h, n_bulge, n_bar)
            
            return generate_data(model, data)
        
        return bulge_2D_model, bulge_disk_2D_model, bulge_bar_disk_2D_model, disk_2D_model


    def main_BD(self, spokes=12, mode=0, psf=True):
        bulge_2D_model, bulge_disk_2D_model, bulge_bar_disk_2D_model, disk_2D_model = self.fit_functions(spokes, psf)

        self.decomp_data = self.create_data(spokes, center=False)
        x_data, y_data, x_err, y_err = self.decomp_data['all']
        data = Data(x_data, y_data, we=1/x_err**2, wd=1/y_err**2)

        ci = np.round(self.center).astype(int)
        cmag_g = self.image['g'][ci[1]-1:ci[1]+2, ci[0]-1:ci[0]+2]
        cmag_r = self.image['r'][ci[1]-1:ci[1]+2, ci[0]-1:ci[0]+2]
        dmu = 2.5*self.get_b(2)/np.log(10)
        p0_1 = [np.mean(cmag_g)+dmu, np.mean(cmag_r)+dmu, 2, np.mean(self.iso_data['g'][0]*0.262), np.mean(self.iso_data['g'][1]*0.262),  self.iso_stat['pa'], 2]
        p0_0 = [np.mean(cmag_g), np.mean(cmag_r),  np.mean(self.iso_data['g'][0]*0.262), np.mean(self.iso_data['g'][1]*0.262), self.iso_stat['pa'], 2]

        odr_1 = ODR(data, Model(bulge_2D_model), beta0=p0_1, maxit=10)
        odr_1.set_job(fit_type=mode)
        output_1 = odr_1.run()
        self.decomp[0] = [[*output_1.beta], [*output_1.sd_beta]]

        odr_0 = ODR(data, Model(disk_2D_model), beta0=p0_0, maxit=10)
        odr_0.set_job(fit_type=mode)
        output_0 = odr_0.run()
        self.decomp[3] = [[*output_0.beta], [*output_0.sd_beta]]
        
        u0g, u0r, ha, hb, pa_d, c_d = self.decomp[3][0]
        dmu = 2.5*self.get_b(1)/np.log(10)
        ind_g = np.argmin(np.abs(self.iso_data['g'][11] - (np.mean(cmag_g)+dmu)))
        ai, bi, pai, ci, *rest = self.iso_data['g'].T[ind_g]
        p0_2 =  [np.mean(cmag_g)+dmu, np.mean(cmag_r)+dmu, u0g, u0r, 1, max(abs(ai), abs(bi))*0.262, min(abs(ai), abs(bi))*0.262, pai, ci, ha, hb, pa_d, c_d]
        odr_2 = ODR(data, Model(bulge_disk_2D_model), beta0=p0_2, maxit=10)
        odr_2.set_job(fit_type=mode)
        output_2 = odr_2.run()
        self.decomp[1] = [[*output_2.beta], [*output_2.sd_beta]]

    
    def bulge_bar_disk_decomp(self, spokes=12, mode=0, psf=True):
        bulge_2D_model, bulge_disk_2D_model, bulge_bar_disk_2D_model, disk_2D_model = self.fit_functions(spokes, psf)  
        def init_bar(ue_bulge, u0_disk, n_bulge, n_bar):
            b_bulge = self.get_b(n_bulge)
            u0_bulge = ue_bulge - 2.5*b_bulge/np.log(10)
            u0_bar = (u0_bulge + u0_disk)/2
            b_bar = self.get_b(n_bar)
            return u0_bar + 2.5*b_bar/np.log(10)
        
        if self.decomp[1][0] != 0:
            self.decomp_data = self.create_data(spokes, center=False)
            x_data, y_data, x_err, y_err = self.decomp_data['all']
            data = Data(x_data, y_data, we=1/x_err**2, wd=1/y_err**2)

            ue_g, ue_r, u0_g, u0_r, n, Re_a, Re_b, Re_pa, Re_n, h_a, h_b, h_pa, h_n = self.decomp[1][0]
            ue_bar_g = init_bar(ue_g, u0_g, n, 0.5)
            ue_bar_r = init_bar(ue_r, u0_r, n, 0.5)
            
            p0_3 = [ue_g, ue_r, ue_bar_g, ue_bar_r, u0_g, u0_r, n, 0.5, Re_b, Re_a, Re_b, Re_pa, 3, h_a, h_b, h_pa, h_n]
            odr_3 = ODR(data, Model(bulge_bar_disk_2D_model), beta0=p0_3, maxit=10)
            odr_3.set_job(fit_type=mode)
            output_3 = odr_3.run()
            self.decomp[2] = [[*output_3.beta], [*output_3.sd_beta]]
    

    def load_main_BD(self, spokes=12, center=False):
        self.decomp_data = self.create_data(spokes, center)
        bd_data_1 = df_bd1[df_bd1['sn_name'] == self.name]
        bd_data_2 = df_bd2[df_bd2['sn_name'] == self.name]
        bd_data_3 = df_bd3[df_bd3['sn_name'] == self.name]
        bd_data_4 = df_bd0[df_bd0['sn_name'] == self.name]
        self.decomp[0] = [[*bd_data_1.values[0][1:8]], [*bd_data_1.values[0][8:]]] if np.any(bd_data_1) else None
        self.decomp[1] = [[*bd_data_2.values[0][1:14]], [*bd_data_2.values[0][14:]]] if np.any(bd_data_2) else None
        self.decomp[2] = [[*bd_data_3.values[0][1:18]], [*bd_data_3.values[0][18:]]] if np.any(bd_data_3) else None
        self.decomp[3] = [[*bd_data_4.values[0][1:7]], [*bd_data_4.values[0][7:]]] if np.any(bd_data_4) else None

    def load_pars(self):
        
        def get_shape(i):
            if i == 0:
                if self.decomp[0][0][0] == 0:
                    return [0,0,0,0,0,0]
                else:
                    ue_g_1, ue_r_1, n_1, Re_a_1, Re_b_1, Re_pa_1, Re_n_1 = self.decomp[0][0]
                    a, b = max(abs(Re_a_1), abs(Re_b_1)), min(abs(Re_a_1), abs(Re_b_1))
                    return [a, 1-b/a, ue_g_1-ue_r_1, n_1, Re_n_1, Re_pa_1%(np.pi)]
                  
            elif i == 1:
                if self.decomp[1][0][0] == 0:
                    return [[0,0,0,0,0,0,0], [0,0,0,0,0,0]]
                else:
                    ue_g_2, ue_r_2, u0_g_2, u0_r_2, n_2, Re_a_2, Re_b_2, Re_pa_2, Re_n_2, h_a_2, h_b_2, h_pa_2, h_n_2 = self.decomp[1][0]
                    a1, b1 =  max(abs(Re_a_2), abs(Re_b_2)), min(abs(Re_a_2), abs(Re_b_2))
                    a2, b2 =  max(abs(h_a_2), abs(h_b_2)), min(abs(h_a_2), abs(h_b_2))
                    return [[a1, 1-b1/a1, ue_g_2, ue_r_2, n_2, Re_n_2, Re_pa_2%(np.pi)], [a2, 1-b2/a2, u0_g_2, u0_r_2, h_n_2, h_pa_2%(np.pi)]]

            elif i == 2:
                if self.decomp[2][0][0] == 0:
                    return [[0,0,0,0], [0,0,0,0,0,0], [0,0,0,0,0,0,0]]
                else:
                    ue_g_bul_3, ue_r_bul_3, ue_g_bar_3, ue_r_bar_3, u0_g_3, u0_r_3, n_bul_3, n_bar_3, Re_bul_3, Re_a_3, Re_b_3, Re_pa_3, Re_n_3, h_a_3, h_b_3, h_pa_3, h_n_3 = self.decomp[2][0]
                    a1, b1 =  max(abs(Re_a_3), abs(Re_b_3)), min(abs(Re_a_3), abs(Re_b_3))
                    a2, b2 =  max(abs(h_a_3), abs(h_b_3)), min(abs(h_a_3), abs(h_b_3))
                    return [[Re_bul_3, ue_g_bul_3, ue_r_bul_3, n_bul_3], [a2, 1-b2/a2, u0_g_3, u0_r_3, h_n_3, h_pa_3%(np.pi)], [a1, 1-b1/a1, ue_g_bar_3, ue_r_bar_3, n_bar_3, Re_n_3, Re_pa_3%(np.pi)]]
            
            elif i == 3:
                if self.decomp[3][0][0] == 0:
                    return [0,0,0,0,0]
                else:
                    ue_g_4, ue_r_4, h_a_4, h_b_4, h_pa_4, h_n_4 = self.decomp[3][0]
                    a, b =  max(abs(h_a_4), abs(h_b_4)), min(abs(h_a_4), abs(h_b_4))
                    return [a, 1-b/a, ue_g_4-ue_r_4, h_n_4, h_pa_4%(np.pi)]

 
        bulge_2D_model, bulge_disk_2D_model, bulge_bar_disk_2D_model, disk_2D_model = self.fit_functions(spokes=12, psf=True, truncate=True)
        self.decomp_data = self.create_data(spokes=12, center=False)
        x_data, y_data, x_err, y_err = self.decomp_data['all']
        if len(x_data) == 0:
            return np.zeros(54)

        def RSS(out, model):
            return np.sum((model(out, x_data) - y_data)**2) / len(x_data)

        RSS_1 = RSS(self.decomp[0][0],  bulge_2D_model) if (self.decomp[0][0][0] != 0) else 1
        RSS_2 = RSS(self.decomp[1][0],  bulge_disk_2D_model) if (self.decomp[1][0][0] != 0) else 1
        RSS_3 = RSS(self.decomp[2][0],  bulge_bar_disk_2D_model) if (self.decomp[2][0][0] != 0) else 1
        RSS_4 = RSS(self.decomp[3][0],  disk_2D_model) if (self.decomp[3][0][0] != 0) else 1

        RSS_arr = [RSS_1, RSS_2, RSS_3, RSS_4]
        err_mean = [x_err.mean(), y_err.mean()]
        n_iso = [self.iso_data['g'].shape[1], self.iso_data['r'].shape[1]]

        fit_0 = get_shape(0)
        fit_1 = get_shape(1)
        fit_2 = get_shape(2)
        fit_3 = get_shape(3)

        return [self.name, *RSS_arr, *err_mean, *fit_0, *fit_1[0], *fit_1[1], *fit_2[0], *fit_2[1], *fit_2[2], *fit_3, x_data.min(), x_data.max(), y_data.min(), y_data.max(), *n_iso]

    @staticmethod
    def bd_ratio(ue, u0, n, Re, h):
        b = 1.9992*n - 0.3271
        Ie, I0 = 10**(-0.4*ue), 10**(-0.4*u0)
        if (I0 != 0) and (h!=0):
            return n*gamma(2*n)*np.exp(b)/b**(2*n) * (Re/h)**2 * (Ie/I0)
        else:
            return 0

    def classify_gal(self, verbose=False):
        keys = ['sn_name','RSS_0','RSS_1','RSS_2','RSS_3','xerr','yerr',   'a_0','e_0','c_0','n_0','se_0','pa_0',
                'a1_1','e1_1','ug1_1','ur1_1','n_1','se1_1','pa1_1',    'a2_1','e2_1','ug2_1','ur2_1','se2_1','pa2_1',
                'a1_2','ug1_2','ur1_2','n1_2',    'a2_2','e2_2','ug2_2','ur2_2','se2_2','pa2_2',    'a3_2','e3_2','ug3_2','ur3_2','n3_2','se3_2','pa3_2',
                'a_3','e_3','c_3','se_3','pa_3',    'x_min','x_max','y_min','y_max','n_iso_g','n_iso_r']
        df_g = pd.DataFrame(columns=keys)
        df_g.loc[0] = self.load_pars()
        galaxy_possible = ['bulge', 'bulge+disk', 'bulge+bar+disk', 'disk']

        z = self.gobj['g'].gal['z']
        self.kcorr = {'Ell': K_corr(z=z, template=df_E), 'Bulge': K_corr(z=z, template=df_E), 'S0': K_corr(z=z, template=df_s0),
                      'Sab': K_corr(z=z, template=df_sab), 'Sc': K_corr(z=z, template=df_sc)}
        
        def rss_check(labels):
            for i, gal_type in enumerate(['bulge', 'bulge+disk', 'bulge+bar+disk', 'disk']):
                rss_lim = 0.03
                print(gal_type, df_g[f'RSS_{i}'].values) if verbose else None
                if (gal_type in labels) and (df_g[f'RSS_{i}'].values > rss_lim):
                    labels.remove(gal_type)
            return labels

        def elliptical_check(labels, ell, col, SE_n, sersic_n):
            col_k = col - np.subtract(*self.kcorr['Ell'])
            condition = (ell < 0.6) and (col_k > 0.5) and (col_k < 2) and (SE_n > 1.8) and (SE_n < 2.2) and (self.decomp[0][0] != 0) 
            if not condition: # se, ecc rework
                labels.remove('bulge')
            return labels
        
        def disk_check(labels, ell, col, SE_n):
            col_k = col - np.subtract(*self.kcorr['Sc'])
            condition = (ell < 0.55) and (col_k > 0) and (col_k < 0.7) and (SE_n > 1) and (SE_n < 3) and (self.decomp[3][0] != 0) 
            if not condition:
                labels.remove('disk')
            return labels
        
        def disk_subtype(BD_color, BD_ratio):
            print(np.round(BD_color, 3), np.round(BD_ratio, 3)) if verbose else None
            if (BD_color > 0) & (BD_color < 0.15) & (BD_ratio > 0.05) & (BD_ratio < 0.6):
                return 'S0'
            elif (BD_color >= 0.15) & (BD_color < 0.5) & (BD_ratio > 0.01) & (BD_ratio < 0.5):
                return 'Sab'
            elif (BD_color >= 0.5) & (BD_color < 1.6) & (BD_ratio > 0.01) & (BD_ratio < 0.2):
                return 'Sc'
            else:
                return 'none'
        
        def bulge_disk_check(labels, bulge_pars, disk_pars):
            Re, b_ell, b_ug, b_ur, b_n, b_SE = bulge_pars
            h, d_ell, d_ug, d_ur, d_SE = disk_pars

            if self.decomp[1][0] == 0:
                labels.remove('bulge+disk')
                return labels

            BD_g = self.bd_ratio(b_ug, d_ug, b_n, Re, h)
            b_col, d_col = b_ug-b_ur, d_ug-d_ur
            kcorr_type = disk_subtype(BD_color=b_col-d_col, BD_ratio=BD_g)
            self.b_sub_type = kcorr_type
            if kcorr_type == 'none':
                labels.remove('bulge+disk')
                return labels
            
            kcorr_bulge, kcorr_disk = self.kcorr['Bulge'], self.kcorr[kcorr_type]
            b_kcol = b_col-np.subtract(*kcorr_bulge)
            d_kcol = d_col-np.subtract(*kcorr_disk)

            bulge_condition = (b_ell < 0.4) and (b_kcol > 0.5) and (b_kcol < 2) and (b_SE > 1.5) and (b_SE < 3)
            disk_condition = (d_ell < 0.55) and (d_kcol > 0) and (d_kcol < 0.8) and (d_SE > 1.5) and (d_SE < 3)
            condition = bulge_condition and disk_condition
            if not condition:
                labels.remove('bulge+disk')
            return labels
        
        def bulge_bar_disk_check(labels, bulge_pars, disk_pars, bar_pars):
            Re_bulge, bulge_ug, bulge_ur, bulge_n = bulge_pars
            h, d_ell, d_ug, d_ur, d_SE = disk_pars
            Re_bar, bar_ell, bar_ug, bar_ur, bar_n, bar_SE = bar_pars
            
            if self.decomp[2][0] == 0:
                labels.remove('bulge+bar+disk')
                return labels

            BD_g = self.bd_ratio(bulge_ug, d_ug, bulge_n, Re_bulge, h)
            bulge_col, bar_col, d_col = bulge_ug-bulge_ur, bar_ug-bar_ur, d_ug-d_ur
            kcorr_type = disk_subtype(BD_color=bulge_col-d_col, BD_ratio=BD_g)
            self.bb_sub_type = kcorr_type
            if kcorr_type == 'none':
                labels.remove('bulge+bar+disk')
                return labels
            
            kcorr_bulge_bar, kcorr_disk = self.kcorr['Bulge'], self.kcorr[kcorr_type]
            bulge_kcol = bulge_col-np.subtract(*kcorr_bulge_bar)
            bar_kcol = bar_col-np.subtract(*kcorr_bulge_bar)
            d_kcol = d_col-np.subtract(*kcorr_disk)

            b_bulge = 1.9992*bulge_n - 0.3271
            b_bar = 1.9992*bar_n - 0.3271
            u0_bulge = bulge_ug - 2.5*b_bulge/np.log(10)
            u0_bar = bar_ug - 2.5*b_bar/np.log(10)
            bulge_condition = (bulge_kcol > 0.5) and (bulge_kcol < 2) and (u0_bulge < u0_bar) and (Re_bulge < Re_bar)
            bar_condition = (bar_ell > 0.3) and (bar_n < 0.75) and (bar_kcol > d_kcol-0.1) and (bar_kcol < bulge_kcol+0.5) and (bar_SE > 1) and (bar_SE < 5) # rework bar color
            disk_condition = (d_ell < 0.55) and (d_kcol > 0) and (d_kcol < 0.8) and (d_SE > 1.5) and (d_SE < 3)
            condition = bulge_condition and bar_condition and disk_condition
            if not condition:
                labels.remove('bulge+bar+disk')
            return labels


        galaxy_possible = elliptical_check(galaxy_possible, *df_g[['e_0', 'c_0', 'se_0','n_0']].values[0])
        print(galaxy_possible) if verbose else None
        galaxy_possible = disk_check(galaxy_possible, *df_g[['e_3', 'c_3', 'se_3']].values[0])
        print(galaxy_possible) if verbose else None
        galaxy_possible = bulge_disk_check(galaxy_possible, df_g[['a1_1', 'e1_1', 'ug1_1', 'ur1_1', 'n_1','se1_1']].values[0], df_g[['a2_1','e2_1','ug2_1','ur2_1','se2_1']].values[0])
        print(galaxy_possible) if verbose else None
        galaxy_possible = bulge_bar_disk_check(galaxy_possible, df_g[['a1_2','ug1_2','ur1_2','n1_2',]].values[0], df_g[['a2_2','e2_2','ug2_2','ur2_2','se2_2']].values[0], df_g[['a3_2','e3_2','ug3_2','ur3_2','n3_2','se3_2']].values[0])
        print(galaxy_possible) if verbose else None
        galaxy_rss_cut = rss_check(galaxy_possible.copy())
        print(galaxy_rss_cut) if verbose else None
        

        if len(galaxy_possible) == 0:
            return 'unclear'
        elif len(galaxy_possible) > 0:
            cond1 = (len(galaxy_rss_cut) == 0)
            cond2 =  (abs(df_g['n_iso_g'].values[0]-df_g['n_iso_r'].values[0]) > 10)
            cond3 = (self.decomp_data['g'][1].max() < 23)
            cond4 = (self.decomp_data['r'][1].max() < 23)
            if cond1 or cond2 or cond3 or cond4:
                return 'bad_fit'
            elif len(galaxy_rss_cut) == 1:
                if galaxy_rss_cut[0] == 'disk':
                    z = self.gobj['g'].gal['z']
                    wcs = self.gobj['g'].cutout['wcs']
                    sn_loc = SkyCoord(*self.gobj['g'].gal['sn'], unit='deg')
                    host_center = wcs.pixel_to_world(*self.center)
                    arcsec2kpc = Planck18.arcsec_per_kpc_proper(z).value
                    separation = host_center.separation(sn_loc).arcsec/arcsec2kpc
                    if (separation < self.decomp_data['g'][0].min()) or (separation < self.decomp_data['r'][0].min()):
                        return 'unclear'
                    else:
                        return 'disk'
                elif galaxy_rss_cut[0] == 'bulge':
                    if (df_g['n_0'].values[0] < 1):
                        return 'core-sersic'
                    else:
                        return 'bulge'
                else:
                    return galaxy_rss_cut[0]
            else:
                if ('disk' in galaxy_rss_cut) and ('bulge+disk' in galaxy_rss_cut) and ('bulge+bar+disk' in galaxy_rss_cut):
                    return 'bulge+bar+disk'
                elif ('bulge' in galaxy_rss_cut) and ('bulge+disk' in galaxy_rss_cut):
                    if  (df_g[f'RSS_0'].values < df_g[f'RSS_1'].values) or (self.b_sub_type == 'S0'):
                        return 'bulge'
                    elif  (self.b_sub_type == 'Sab'):
                        return 'E-S0'
                    else:
                        return 'bulge+disk'
                elif ('bulge' in galaxy_rss_cut) and ('bulge+bar+disk' in galaxy_rss_cut):
                    if  (df_g[f'RSS_0'].values < df_g[f'RSS_2'].values) or (self.bb_sub_type == 'S0'):
                        return 'bulge'
                    elif  (self.bb_sub_type == 'Sab'):
                        return 'E-S0'
                    else:
                        return 'bulge+bar+disk'
                elif ('bulge' in galaxy_rss_cut) and ('disk' in galaxy_rss_cut):
                    return 'E-S0'
                elif ('disk' in galaxy_rss_cut) and ('bulge+disk' in galaxy_rss_cut):
                    return 'bulge+disk'
                elif ('bulge+disk' in galaxy_rss_cut) and ('bulge+bar+disk' in galaxy_rss_cut):
                    return 'bulge+bar+disk'
                elif ('disk' in galaxy_rss_cut) and ('bulge+bar+disk' in galaxy_rss_cut):
                    return 'bulge+bar+disk'
               
                else:
                    return 'unclear'
                
    @staticmethod
    def total_mag_lum(u, R, n, disk=False):
        b = 1.9992*n - 0.3271
        ue = u + 2.5*b/np.log(10) if disk else u
        Re = b**n * R if disk else R
        f = n*np.exp(b)/b**(2*n) * gamma(2*n)

        mag = ue - 2.5*np.log10(f) - 2.5*np.log10(2*np.pi*Re**2)
        lum = np.pi * np.exp(b)*10**(-0.4*ue) * (Re/b**n)**2 * gamma(2*n + 1)
        return mag, lum


    def galaxy_pars(self):
        galaxy_type = self.classify_gal()
        gal_i = np.where(np.array(['bulge', 'bulge+disk', 'bulge+bar+disk', 'disk', 'E-S0', 'core-sersic', 'unclear', 'bad_fit']) == galaxy_type)[0][0]

        z = self.gobj['g'].gal['z']
        z_source = self.gobj['g'].gal['z_source']
        A_dim = 10*np.log10(1+z)
        wcs = self.gobj['g'].cutout['wcs']
        sn_loc = SkyCoord(*self.gobj['g'].gal['sn'], unit='deg')
        host_center = wcs.pixel_to_world(*self.center)

        arcsec2kpc = Planck18.arcsec_per_kpc_proper(z).value
        separation = host_center.separation(sn_loc).arcsec/arcsec2kpc
        sn_deg = np.array([sn_loc.ra.deg, sn_loc.dec.deg])
        host_deg = np.array([host_center.ra.deg, host_center.dec.deg])
        sn_vec = sn_deg - host_deg
        theta = np.pi - np.arctan(sn_vec[1]/sn_vec[0]) # is this still correct?

        if self.decomp[0][0][0] != 0:
            if gal_i == 3:
                model_i = 0
            elif gal_i >= 4:
                model_i = 1
            else:
                model_i = gal_i+1

            sn_local_g = self.SB_profile(separation*arcsec2kpc, theta, band='g', model='combine', n_model=model_i, px=False, corr=True)
            sn_local_r = self.SB_profile(separation*arcsec2kpc, theta, band='r', model='combine', n_model=model_i, px=False, corr=True)
            try:
                R25_g_a,R25_g_b,pa_g,n_g  = self.SB_isophote(25, 'g', n_model=model_i, model='combine', corr=True)[0]
                R25_r_a,R25_r_b,pa_r,n_r  = self.SB_isophote(25, 'r', n_model=model_i, model='combine', corr=True)[0]
                R25_g, R25_r = R25_g_a*0.262/arcsec2kpc, R25_r_a*0.262/arcsec2kpc
                R25_sn_g = self.super_ellipse(theta, R25_g_a*0.262/arcsec2kpc, R25_g_b*0.262/arcsec2kpc, pa_g, n_g)
                R25_sn_r = self.super_ellipse(theta, R25_r_a*0.262/arcsec2kpc, R25_r_b*0.262/arcsec2kpc, pa_r, n_r)
            except RuntimeError:
                R25_g, R25_r, R25_sn_g, R25_sn_r = 0, 0, 0, 0
        else:
            sn_local_g, sn_local_r, R25_g, R25_r, R25_sn_g, R25_sn_r = 0, 0, 0, 0, 0, 0

        def extract_bulge_pars(fit_params):
            if len(fit_params) == 0:
                return np.zeros(14)
            else:
                pars, errs, kcorr = fit_params
                ue_g, ue_r, n, Re_a, Re_b, Re_pa, Re_n = pars
                ue_g_err, ue_r_err, n_err, Re_a_err, Re_b_err, Re_pa_err, Re_n_err = errs
                Re_a, Re_b = max(abs(Re_a), abs(Re_b)), min(abs(Re_a), abs(Re_b))

                Bulge_ue_g, Bulge_ue_r, Bulge_n = ue_g-(kcorr[0]+A_dim), ue_r-(kcorr[1]+A_dim), n
                Bulge_ue_g_err, Bulge_ue_r_err, Bulge_n_err = ue_g_err, ue_r_err, n_err
                Bulge_Re, Bulge_ecc, Bulge_pa, Bulge_SE = Re_a/(arcsec2kpc), 1-Re_b/Re_a, Re_pa % (np.pi), Re_n
                Bulge_Re_err, Bulge_ecc_err, Bulge_pa_err, Bulge_SE_err = np.sqrt(Re_a_err*Re_b_err)/arcsec2kpc, abs(Re_a_err/Re_a), Re_pa_err, Re_n_err
                return Bulge_ue_g, Bulge_ue_r, Bulge_n, Bulge_ue_g_err, Bulge_ue_r_err, Bulge_n_err, Bulge_Re, Bulge_ecc, Bulge_pa, Bulge_SE, Bulge_Re_err, Bulge_ecc_err, Bulge_pa_err, Bulge_SE_err

        def extract_disk_pars(fit_params):
            if len(fit_params) == 0:
                return np.zeros(12)
            else:
                pars, errs, kcorr = fit_params
                u0_g, u0_r, h_a, h_b, h_pa, h_n = pars
                u0_g_err, u0_r_err, h_a_err, h_b_err, h_pa_err, h_n_err = errs
                h_a, h_b = max(abs(h_a), abs(h_b)), min(abs(h_a), abs(h_b))
                Disk_u0_g, Disk_u0_r = u0_g-(kcorr[0]+A_dim), u0_r-(kcorr[1]+A_dim)
                Disk_u0_g_err, Disk_u0_r_err = u0_g_err, u0_r_err
                Disk_h, Disk_ecc, Disk_pa, Disk_SE = h_a/(arcsec2kpc), 1-h_b/h_a, h_pa % (np.pi), h_n
                Disk_h_err, Disk_ecc_err, Disk_pa_err, Disk_SE_err = np.sqrt(h_a_err*h_b_err)/arcsec2kpc, abs(h_a_err/h_a), h_pa_err, h_n_err
                return Disk_u0_g, Disk_u0_r, Disk_u0_g_err, Disk_u0_r_err, Disk_h, Disk_ecc, Disk_pa, Disk_SE, Disk_h_err, Disk_ecc_err, Disk_pa_err, Disk_SE_err

        if self.decomp[0][0][0] == 0:
            bulge_pars = extract_bulge_pars([])
            disk_pars = extract_disk_pars([])
            bar_pars = extract_bulge_pars([])
            sn_component = 'none'
            sn_local_g_err, sn_local_r_err = 0, 0
            bulge_disk_ratio, gal_Re, gal_ecc, gal_pa = 0, 1, 0, 0
            bulge_mag_g, bulge_mag_r, bulge_lum_g, bulge_lum_r = 0, 0, 0, 0
            bar_mag_g, bar_mag_r, bar_lum_g, bar_lum_r = 0, 0, 0, 0
            disk_mag_g, disk_mag_r, disk_lum_g, disk_lum_r = 0, 0, 0, 0

        elif (gal_i == 0) or (gal_i >= 4):
            fit_data, fit_errs = self.decomp[0]
            kcorr = self.kcorr['Ell']
            bulge_pars = extract_bulge_pars([fit_data, fit_errs, kcorr])
            disk_pars = extract_disk_pars([])
            bar_pars = extract_bulge_pars([])
            sn_component = 'elliptical' if (gal_i == 0) else galaxy_type
            sn_local_g_err = bulge_pars[3]
            sn_local_r_err = bulge_pars[4]
            bulge_disk_ratio = 0
            gal_Re, gal_ecc, gal_pa = bulge_pars[6], bulge_pars[7], bulge_pars[8]
            bulge_mag_g, bulge_lum_g = self.total_mag_lum(bulge_pars[0], bulge_pars[6]*np.sqrt(1-bulge_pars[7]), bulge_pars[2], disk=False)
            bulge_mag_r, bulge_lum_r = self.total_mag_lum(bulge_pars[1], bulge_pars[6]*np.sqrt(1-bulge_pars[7]), bulge_pars[2], disk=False)
            bar_mag_g, bar_mag_r, bar_lum_g, bar_lum_r = 0, 0, 0, 0
            disk_mag_g, disk_mag_r, disk_lum_g, disk_lum_r = 0, 0, 0, 0
           
            
        elif gal_i == 1:
            fit_data, fit_errs = self.decomp[1]
            ue_g, ue_r, u0_g, u0_r, n, Re_a, Re_b, Re_pa, Re_n, h_a, h_b, h_pa, h_n = fit_data
            ue_g_err, ue_r_err, u0_g_err, u0_r_err, n_err, Re_a_err, Re_b_err, Re_pa_err, Re_n_err, h_a_err, h_b_err, h_pa_err, h_n_err = fit_errs
            kcorr_bulge, kcorr_disk = self.kcorr['Bulge'], self.kcorr[self.b_sub_type]
            bulge_pars = extract_bulge_pars([[ue_g,ue_r,n,Re_a,Re_b,Re_pa,Re_n], [ue_g_err,ue_r_err,n_err,Re_a_err,Re_b_err,Re_pa_err,Re_n_err], kcorr_bulge])
            disk_pars = extract_disk_pars([[u0_g,u0_r,h_a,h_b,h_pa,h_n], [u0_g_err,u0_r_err,h_a_err,h_b_err,h_pa_err,h_n_err], kcorr_disk])
            bar_pars = extract_bulge_pars([])
            bulge_disk_ratio = self.bd_ratio(bulge_pars[0], disk_pars[0], bulge_pars[2], bulge_pars[6], disk_pars[4])
            gal_Re, gal_ecc, gal_pa = 1.678*disk_pars[4], disk_pars[5], disk_pars[6]

            sn_bulge = self.SB_profile(separation*arcsec2kpc, theta, band='g', model='bulge', n_model=gal_i+1, px=False, corr=True)
            sn_disk = self.SB_profile(separation*arcsec2kpc, theta, band='g', model='disk', n_model=gal_i+1, px=False, corr=True)
            if (sn_bulge - sn_disk < 1):
                sn_component = 'bulge' 
                sn_local_g_err = bulge_pars[3]
                sn_local_r_err = bulge_pars[4]
            else:
                sn_component = 'disk' 
                sn_local_g_err = disk_pars[2]
                sn_local_r_err = disk_pars[3]
            bulge_mag_g, bulge_lum_g = self.total_mag_lum(bulge_pars[0], bulge_pars[6]*np.sqrt(1-bulge_pars[7]), bulge_pars[2], disk=False)
            bulge_mag_r, bulge_lum_r = self.total_mag_lum(bulge_pars[1], bulge_pars[6]*np.sqrt(1-bulge_pars[7]), bulge_pars[2], disk=False)
            bar_mag_g, bar_mag_r, bar_lum_g, bar_lum_r = 0, 0, 0, 0
            disk_mag_g, disk_lum_g = self.total_mag_lum(disk_pars[0], disk_pars[4]*np.sqrt(1-disk_pars[5]), 1, disk=True)
            disk_mag_r, disk_lum_r = self.total_mag_lum(disk_pars[1], disk_pars[4]*np.sqrt(1-disk_pars[5]), 1, disk=True)
            
        elif gal_i == 2:
            fit_data, fit_errs = self.decomp[2]
            ue_bulge_g, ue_bulge_r, ue_bar_g, ue_bar_r, u0_g, u0_r, n_bulge, n_bar, Re_bulge, Re_bar_a, Re_bar_b, Re_bar_pa, Re_bar_n, h_a, h_b, h_pa, h_n = fit_data
            ue_bulge_g_err, ue_bulge_r_err, ue_bar_g_err, ue_bar_r_err, u0_g_err, u0_r_err, n_bulge_err, n_bar_err, Re_bulge_err, Re_bar_a_err, Re_bar_b_err, Re_bar_pa_err, Re_bar_n_err, h_a_err, h_b_err, h_pa_err, h_n_err = fit_errs
            bulge_input, bulge_input_err = [ue_bulge_g, ue_bulge_r, n_bulge, Re_bulge, Re_bulge, 0, 2], [ue_bulge_g_err, ue_bulge_r_err, n_bulge_err, Re_bulge_err, Re_bulge_err, 0.01, 0.001]
            disk_input, disk_input_err = [u0_g, u0_r, h_a, h_b, h_pa, h_n], [u0_g_err, u0_r_err, h_a_err, h_b_err, h_pa_err, h_n_err]
            bar_input, bar_input_err = [ue_bar_g, ue_bar_r, n_bar, Re_bar_a, Re_bar_b, Re_bar_pa, Re_bar_n], [ue_bar_g_err, ue_bar_r_err, n_bar_err, Re_bar_a_err, Re_bar_b_err, Re_bar_pa_err, Re_bar_n_err]
            kcorr_bulge, kcorr_disk = self.kcorr['Bulge'], self.kcorr[self.bb_sub_type]
            bulge_pars = extract_bulge_pars([bulge_input, bulge_input_err, kcorr_bulge])
            disk_pars = extract_disk_pars([disk_input, disk_input_err, kcorr_disk])
            bar_pars = extract_bulge_pars([bar_input, bar_input_err, kcorr_bulge])

            bulge_disk_ratio = self.bd_ratio(bulge_pars[0], disk_pars[0], bulge_pars[2], bulge_pars[6], disk_pars[4])
            gal_Re, gal_ecc, gal_pa = 1.678*disk_pars[4], disk_pars[5], disk_pars[6]

            sn_bulge = self.SB_profile(separation*arcsec2kpc, theta, band='g', model='bulge', n_model=gal_i+1, px=False, corr=True)
            sn_bar = self.SB_profile(separation*arcsec2kpc, theta, band='g', model='bar', n_model=gal_i+1, px=False, corr=True)
            sn_disk = self.SB_profile(separation*arcsec2kpc, theta, band='g', model='disk', n_model=gal_i+1, px=False, corr=True)
            if (sn_bar < sn_bulge) & (sn_bar < sn_disk):
                sn_component = 'bar'
                sn_local_g_err = bulge_pars[3]
                sn_local_r_err = bulge_pars[4]
            elif (sn_bulge - sn_disk < 1):
                sn_component = 'bulge' 
                sn_local_g_err = bulge_pars[3]
                sn_local_r_err = bulge_pars[4]
            else:
                sn_component = 'disk'
                sn_local_g_err = disk_pars[2]
                sn_local_r_err = disk_pars[3]
            bulge_mag_g, bulge_lum_g = self.total_mag_lum(bulge_pars[0], bulge_pars[6]*np.sqrt(1-bulge_pars[7]), bulge_pars[2], disk=False)
            bulge_mag_r, bulge_lum_r = self.total_mag_lum(bulge_pars[1], bulge_pars[6]*np.sqrt(1-bulge_pars[7]), bulge_pars[2], disk=False)
            bar_mag_g, bar_lum_g = self.total_mag_lum(bar_pars[0], bar_pars[6]*np.sqrt(1-bar_pars[7]), bar_pars[2], disk=False)
            bar_mag_r, bar_lum_r = self.total_mag_lum(bar_pars[1], bar_pars[6]*np.sqrt(1-bar_pars[7]), bar_pars[2], disk=False)
            disk_mag_g, disk_lum_g = self.total_mag_lum(disk_pars[0], disk_pars[4]*np.sqrt(1-disk_pars[5]), 1, disk=True)
            disk_mag_r, disk_lum_r = self.total_mag_lum(disk_pars[1], disk_pars[4]*np.sqrt(1-disk_pars[5]), 1, disk=True)

        elif gal_i == 3:
            fit_data, fit_errs = self.decomp[3]
            kcorr = self.kcorr['Sc']
            bulge_pars = extract_bulge_pars([])
            disk_pars = extract_disk_pars([fit_data, fit_errs, kcorr])
            bar_pars = extract_bulge_pars([])
            sn_component = 'disk'
            sn_local_g_err = disk_pars[2]
            sn_local_r_err = disk_pars[3]
            bulge_disk_ratio = 0
            gal_Re, gal_ecc, gal_pa = 1.678*disk_pars[4], disk_pars[5], disk_pars[6]
            bulge_mag_g, bulge_mag_r, bulge_lum_g, bulge_lum_r = 0, 0, 0, 0
            bar_mag_g, bar_mag_r, bar_lum_g, bar_lum_r = 0, 0, 0, 0
            disk_mag_g, disk_lum_g = self.total_mag_lum(disk_pars[0], disk_pars[4]*np.sqrt(1-disk_pars[5]), 1, disk=True)
            disk_mag_r, disk_lum_r = self.total_mag_lum(disk_pars[1], disk_pars[4]*np.sqrt(1-disk_pars[5]), 1, disk=True)
           
        
        rotation_matrix = np.array([[np.cos(gal_pa), np.sin(gal_pa)], [-np.sin(gal_pa), np.cos(gal_pa)]])
        sn_radial, sn_height = (rotation_matrix @ (sn_deg - host_deg)) * 3600/arcsec2kpc
        
        dr2_pars = df_salt.loc[self.name][['x1', 'c', 'peak_mag_ztfg', 'peak_mag_ztfr', 'lccoverage_flag', 'x1_err', 'c_err', 'frac_fitted', 'lcquality_flag', 'fitprob']]
        sn_class = df_class.loc[self.name]

        if np.any(si_umut[si_umut['ztfname'] == self.name]):
            si_pars = si_umut[si_umut['ztfname'] == self.name][['V_Sil_6355_median', 'FWHMA_Sil_6355_median', 'pEW_Sil_6355_median_trap','V_sil_6355_err_final_high',  'FWHMA_err_high_6355_final', 'pEW_err_trap_high_6355_final',
                                                                 'V_Sil_5972_median', 'FWHMA_Sil_5972_median', 'pEW_Sil_5972_median_trap','V_sil_5972_err_final_high', 'FWHMA_err_high_5972_final', 'pEW_err_trap_high_5972_final',
                                                                 'pEW_ratio', 'pEW_ratio_err']].values[0]
        else:
            si_pars = np.zeros(14)
        
        if self.name in df_mass.index:
            host_mass, mass_error = df_mass[['mass', 'mass_err']].loc[self.name]
            smsd = np.log10(10**host_mass/(2*np.pi*gal_Re**2))
            tau = 1.12 * smsd - 8.6
            tau_list = disk_dust['tau'].unique()
            proj_h, proj_u0, dust_h, dust_u0_g, dust_u0_r = get_corrections(gal_ecc, tau_list[np.argmin(np.abs(tau-tau_list))])
        else:
            host_mass, mass_error, smsd, tau = 0, 0, 0, 0
            proj_h, proj_u0, dust_h, dust_u0_g, dust_u0_r = 0, 0, 0, 0, 0
        # psf colour too
        self.galaxy = [self.name, z, z_source, *sn_deg, *host_deg, galaxy_type, sn_component, separation, sn_radial, sn_height, sn_local_g, sn_local_r, sn_local_g_err, sn_local_r_err, 
                       R25_g, R25_r,R25_sn_g, R25_sn_r, *dr2_pars, *sn_class, *si_pars, *bulge_pars, *disk_pars, *bar_pars, bulge_disk_ratio, host_mass, mass_error, smsd, tau,
                       proj_h, proj_u0, dust_h, dust_u0_g, dust_u0_r, gal_Re, gal_ecc, gal_pa, bulge_mag_g, bulge_mag_r, bulge_lum_g, bulge_lum_r, bar_mag_g, bar_mag_r, bar_lum_g, bar_lum_r,
                       disk_mag_g, disk_mag_r, disk_lum_g, disk_lum_r]
                                                      

    def plot_func(self, psf):
        xm, ym, rm, thetam = self.get_meshgrid()

        def psf_convolve(model):
            model_grid_g = model(rm*0.262, thetam, 'g')
            model_grid_r = model(rm*0.262, thetam, 'r')
            model_psf_g = -2.5*np.log10(convolve(10**(-0.4*model_grid_g), self.kernal['g'], mode='same'))
            model_psf_r = -2.5*np.log10(convolve(10**(-0.4*model_grid_r), self.kernal['r'], mode='same'))
            interp_g = RegularGridInterpolator((xm[0]*0.262, ym.T[0]*0.262), model_psf_g.T)
            interp_r = RegularGridInterpolator((xm[0]*0.262, ym.T[0]*0.262), model_psf_r.T)
            return {'g': interp_g, 'r': interp_r}
        
        def model_psf(r, theta, band, psf_interp):
            x, y = r*np.cos(theta)+self.center[0]*0.262, r*np.sin(theta)+self.center[1]*0.262
            return psf_interp[band]((x, y))
        
        def generate_data(model, x_data, theta, band):
            if psf:
                psf_interp = psf_convolve(model)
                return model_psf(x_data, theta, band, psf_interp)
            else:
                return model(x_data, theta, band)
        
        def disk_2D_model(x_data, pars, theta, band):
            rp = 2
            u0_g, u0_r = pars[:rp]
            u0 = {'g': u0_g, 'r': u0_r}
            def model(x, theta, band):
                h = self.super_ellipse(theta, *pars[rp:rp+4])
                return self.disk(x, u0[band], h)
            return generate_data(model, x_data, theta, band)

        def bulge_2D_model(x_data, pars, theta, band):
            rp = 3
            ue_g, ue_r, n = pars[:rp]
            ue = {'g': ue_g, 'r': ue_r}
            def model(x, theta, band):
                Re = self.super_ellipse(theta, *pars[rp:rp+4])
                return self.bulge(x, ue[band], Re, n)
            return generate_data(model, x_data, theta, band)

        def bulge_disk_2D_model(x_data, pars, theta, band):
            rp = 5
            ue_g, ue_r, u0_g, u0_r, n = pars[:rp]
            ue = {'g': ue_g, 'r': ue_r}
            u0 = {'g': u0_g, 'r': u0_r}
            def model(x, theta, band):
                Re = self.super_ellipse(theta, *pars[rp:rp+4])
                h = self.super_ellipse(theta, *pars[rp+4:rp+8])
                return self.fit_combine(x, ue[band], u0[band], Re, h, n)
            return generate_data(model, x_data, theta, band)

        def bulge_bar_disk_2D_model(x_data, pars, theta, band):
            rp = 9
            ue_bulge_g, ue_bulge_r, ue_bar_g, ue_bar_r, u0_g, u0_r, n_bulge, n_bar, Re_bulge = pars[:rp]
            ue_bulge = {'g': ue_bulge_g, 'r': ue_bulge_r}
            ue_bar = {'g': ue_bar_g, 'r': ue_bar_r}
            u0 = {'g': u0_g, 'r': u0_r}
            def model(x, theta, band):
                Re_bar = self.super_ellipse(theta, *pars[rp:rp+4])
                h = self.super_ellipse(theta, *pars[rp+4:rp+8])
                return self.fit_combine_3(x, ue_bulge[band], ue_bar[band], u0[band], Re_bulge, Re_bar, h, n_bulge, n_bar)
            return generate_data(model, x_data, theta, band)

        return  disk_2D_model, bulge_2D_model, bulge_disk_2D_model, bulge_bar_disk_2D_model

        
    def plot_spokes(self, sigma=3, n_model=2, psf=True, center=True):
        spokes=12
        rp = [3, 5, 9, 2][n_model-1]
        disk_2D_model, bulge_2D_model, bulge_disk_2D_model, bulge_bar_disk_2D_model = self.plot_func(psf)  
        self.decomp_data = self.create_data(spokes, center=center)
        fig, axis = plt.subplots(figsize=(18, 10), ncols=2, nrows=3, dpi=100)

        wcs = self.gobj['g'].cutout['wcs']
        sn_loc = SkyCoord(*self.gobj['g'].gal['sn'], unit='deg')
        host_center = wcs.pixel_to_world(*self.center)
        separation = host_center.separation(sn_loc).arcsec
        sn_deg = np.array([sn_loc.ra.deg, sn_loc.dec.deg])
        host_deg = np.array([host_center.ra.deg, host_center.dec.deg])
        sn_vec = sn_deg - host_deg
        theta_sn = np.arctan(sn_vec[1]/sn_vec[0])

        for i in range(6):
            band = 'g' if i%2 == 0 else 'r'
            bd_data = self.decomp_data[band]
            x_data, y_data = bd_data[0].reshape(spokes, -1)[::int(spokes/6)], bd_data[1].reshape(spokes, -1)[::int(spokes/6)]
            x_err, y_err = bd_data[2].reshape(spokes, -1)[::int(spokes/6)], bd_data[3].reshape(spokes, -1)[::int(spokes/6)]
            ax = axis.flatten()[i]
            theta = {'g': np.linspace(0, 360-360/spokes, spokes), 'r': np.linspace(180/spokes, 360-180/spokes, spokes)}[band][::2]
            ax.errorbar(x_data[i], y_data[i],yerr=sigma*y_err[i], xerr=sigma*x_err[i], fmt='k.', zorder=0, label=fr'{sigma} sigma,  $\theta =${theta[i]:.0f}$^\circ$')
            ax.set_ylim([y_data.min()-0.5, 25.5])
            ax.invert_yaxis()
            ax.axvline(separation, color='k', label=f'SN {np.rad2deg(theta_sn) % (360):.2f} deg')

        x_ax = np.linspace(0, x_data.max()+1, 100)
        if n_model == 3:
            ue_bulge_g, ue_bulge_r, ue_bar_g, ue_bar_r, u0_g, u0_r, n_bulge, n_bar, Re_bulge = self.decomp[n_model-1][0][:rp]
            for i in range(6):
                band = 'g' if i%2 == 0 else 'r'
                ue_bulge = {'g': ue_bulge_g, 'r': ue_bulge_r}[band]
                ue_bar = {'g': ue_bar_g, 'r': ue_bar_r}[band]
                u0_disk = {'g': u0_g, 'r': u0_r}[band]
                theta = {'g': np.linspace(0, 360-360/spokes, spokes), 'r': np.linspace(180/spokes, 360-180/spokes, spokes)}[band][::2]
                ax = axis.flatten()[i]
                Re_bar = self.super_ellipse(np.deg2rad(theta[i]), *self.decomp[n_model-1][0][rp:rp+4])
                h_disk = self.super_ellipse(np.deg2rad(theta[i]), *self.decomp[n_model-1][0][rp+4:rp+8])
                ax.plot(x_ax, bulge_bar_disk_2D_model(x_ax, self.decomp[n_model-1][0], np.deg2rad(theta[i]), band), 'r-', label=f'combined_{band}')
                x_bulge = np.linspace(0, self.trunc_bulge*abs(Re_bulge), 100)
                x_bar = np.linspace(0, self.trunc_bar*abs(Re_bar), 100)
                ax.plot(x_bulge, self.bulge(x_bulge, ue_bulge, Re_bulge, n_bulge), 'g--', label='bulge')
                ax.plot(x_bar, self.bulge(x_bar, ue_bar, Re_bar, n_bar), 'g--', label='bar', color='lime')
                ax.plot(x_ax, self.disk(x_ax, u0_disk, h_disk), 'b--', label='disk')
                ax.legend()

        elif n_model == 2:
            ue_g, ue_r, u0_g, u0_r, n_sersic = self.decomp[n_model-1][0][:rp]
            
            for i in range(6):
                band = 'g' if i%2 == 0 else 'r'
                ue_bulge = {'g': ue_g, 'r': ue_r}[band]
                u0_disk = {'g': u0_g, 'r': u0_r}[band]
                theta = {'g': np.linspace(0, 360-360/spokes, spokes), 'r': np.linspace(180/spokes, 360-180/spokes, spokes)}[band][::2]
                ax = axis.flatten()[i]
                Re_bulge = self.super_ellipse(np.deg2rad(theta[i]), *self.decomp[n_model-1][0][rp:rp+4])
                h_disk = self.super_ellipse(np.deg2rad(theta[i]), *self.decomp[n_model-1][0][rp+4:rp+8])    
                ax.plot(x_ax, bulge_disk_2D_model(x_ax, self.decomp[n_model-1][0], np.deg2rad(theta[i]), band), 'r-', label=f'combined_{band}')
                x_bulge = np.linspace(0, self.trunc_bulge*abs(Re_bulge), 100)
                ax.plot(x_bulge, self.bulge(x_bulge, ue_bulge, Re_bulge, n_sersic), 'g--', label='bulge')
                ax.plot(x_ax, self.disk(x_ax, u0_disk, h_disk), 'b--', label='disk')
                ax.legend()

        elif n_model == 1:
            ue_g, ue_r, n_sersic = self.decomp[n_model-1][0][:rp]
           
            for i in range(6):
                band = 'g' if i%2 == 0 else 'r'
                ue_bulge = {'g': ue_g, 'r': ue_r}[band]
                theta = {'g': np.linspace(0, 360-360/spokes, spokes), 'r': np.linspace(180/spokes, 360-180/spokes, spokes)}[band][::2]
                ax = axis.flatten()[i]
                ax.plot(x_ax, bulge_2D_model(x_ax, self.decomp[n_model-1][0], np.deg2rad(theta[i]), band), 'r-', label=f'bulge_{band}')   
                ax.legend()  
        
        elif n_model == 0:
            u0_g, u0_r = self.decomp[n_model-1][0][:rp]
           
            for i in range(6):
                band = 'g' if i%2 == 0 else 'r'
                u0_disk = {'g': u0_g, 'r': u0_r}[band]
                theta = {'g': np.linspace(0, 360-360/spokes, spokes), 'r': np.linspace(180/spokes, 360-180/spokes, spokes)}[band][::2]
                ax = axis.flatten()[i]
                ax.plot(x_ax, disk_2D_model(x_ax, self.decomp[n_model-1][0], np.deg2rad(theta[i]), band), 'r-', label=f'disk_{band}')   
                ax.legend() 

        plt.tight_layout()


    def SB_profile(self, r, theta, band, model='all', n_model=2, px=True, corr=False):
        z = self.gobj['g'].gal['z']
        A_dim = 10*np.log10(1+z)
        rp = [3, 5, 9, 2][n_model-1]
        arcsec2px = np.array([0.262, 0.262, 1, 1]) if px else np.ones(4)
        if n_model == 3:
            ue_bulge_g, ue_bulge_r, ue_bar_g, ue_bar_r, u0_g, u0_r, n_bulge, n_bar, Re_bulge = self.decomp[n_model-1][0][:rp]
            Re_bar = self.super_ellipse(theta, *self.decomp[n_model-1][0][rp:rp+4]/arcsec2px)
            h_disk = self.super_ellipse(theta, *self.decomp[n_model-1][0][rp+4:rp+8]/arcsec2px)

            if corr:
                kcorr_bulge_bar, kcorr_disk = self.kcorr['Bulge'], self.kcorr[self.bb_sub_type]
                ue_bulge_g -= (kcorr_bulge_bar[0]+A_dim)
                ue_bulge_r -= (kcorr_bulge_bar[1]+A_dim)
                ue_bar_g -= (kcorr_bulge_bar[0]+A_dim)
                ue_bar_r -= (kcorr_bulge_bar[1]+A_dim)
                u0_g -= (kcorr_disk[0]+A_dim)
                u0_r -= (kcorr_disk[1]+A_dim)

            ue_bulge = {'g': ue_bulge_g, 'r': ue_bulge_r}[band]
            ue_bar = {'g': ue_bar_g, 'r': ue_bar_r}[band]
            u0_disk = {'g': u0_g, 'r': u0_r}[band]
            
            div = 0.262 if px else 1
            if model == 'bulge':
                return self.bulge(r, ue_bulge, Re_bulge/div, n_bulge)
            elif model == 'bar':
                return self.bulge(r, ue_bar, Re_bar, n_bar)
            elif model == 'disk':
                return self.disk(r, u0_disk, h_disk)
            else:
                return self.combine_3(r, ue_bulge, ue_bar, u0_disk, Re_bulge/div, Re_bar, h_disk, n_bulge, n_bar)
            
        elif n_model == 2:
            ue_g, ue_r, u0_g, u0_r, n_sersic = self.decomp[n_model-1][0][:rp]
            
            Re_bulge = self.super_ellipse(theta, *self.decomp[n_model-1][0][rp:rp+4]/arcsec2px)
            h_disk = self.super_ellipse(theta, *self.decomp[n_model-1][0][rp+4:rp+8]/arcsec2px)
            if corr:
                kcorr_bulge, kcorr_disk = self.kcorr['Bulge'], self.kcorr[self.b_sub_type]
                ue_g -= (kcorr_bulge[0]+A_dim)
                ue_r -= (kcorr_bulge[1]+A_dim)
                u0_g -= (kcorr_disk[0]+A_dim)
                u0_r -= (kcorr_disk[1]+A_dim)
            
            ue_bulge = {'g': ue_g, 'r': ue_r}[band]
            u0_disk = {'g': u0_g, 'r': u0_r}[band]

            if model == 'bulge':
                return self.bulge(r, ue_bulge, Re_bulge, n_sersic)
            elif model == 'disk':
                return self.disk(r, u0_disk, h_disk)
            else:
                 return self.combine(r, ue_bulge, u0_disk, Re_bulge, h_disk, n_sersic)
            
        elif n_model == 1:
            ue_g, ue_r, n_sersic = self.decomp[n_model-1][0][:rp]
            if corr:
                kcorr = self.kcorr['Ell']
                ue_g -= (kcorr[0]+A_dim)
                ue_r -= (kcorr[1]+A_dim)

            ue_bulge = {'g': ue_g, 'r': ue_r}[band]
            Re_bulge = self.super_ellipse(theta, *self.decomp[n_model-1][0][rp:rp+4]/arcsec2px)
            return self.bulge(r, ue_bulge, Re_bulge, n_sersic)
        
        elif n_model == 0:
            u0_g, u0_r = self.decomp[n_model-1][0][:rp]
            h_disk = self.super_ellipse(theta, *self.decomp[n_model-1][0][rp:rp+4]/arcsec2px)
            if corr:
                kcorr = self.kcorr['Sc']
                u0_g -= (kcorr[0]+A_dim)
                u0_r -= (kcorr[1]+A_dim)
 
            u0_disk = {'g': u0_g, 'r': u0_r}[band]
            return self.disk(r, u0_disk, h_disk)
        
    
    def SB_isophote(self, isophote, band, n_model, model='both', corr=False): 
        def radial(isophote, theta):
            sb_profile = lambda r: self.SB_profile(r, theta, band=band, n_model=n_model, model=model, corr=corr) - isophote
            return fsolve(sb_profile, 1)

        theta = np.linspace(0, 2*np.pi, 50)
        r = np.vectorize(radial)(isophote, theta)

        out_pars = curve_fit(self.super_ellipse, theta, r, p0=[r.max()+0.1, r.min()-0.1, self.iso_stat['pa'],  2], maxfev=5000)
        check = np.mean(self.SB_profile(r, theta, band=band, n_model=n_model, model=model, corr=corr)) - isophote
        a, b, pa, n = out_pars[0]
        return [[max(abs(a), abs(b)), min(abs(a), abs(b)), pa, n], np.sqrt(np.diag(out_pars[1])), np.round(check, 3)] 


    def plot_SB_profile(self, band, isophote=False, n_model=2, corr=False, psf=True):
        xm, ym, rm, thetam = self.get_meshgrid()

        bulge_arr = self.SB_profile(rm, thetam, model='bulge', n_model=n_model, band=band,  corr=corr)
        bar_arr = self.SB_profile(rm, thetam, model='bar', n_model=n_model, band=band, corr=corr)
        disk_arr = self.SB_profile(rm, thetam, model='disk', n_model=n_model, band=band, corr=corr)
        comb_arr = self.SB_profile(rm, thetam, model='all', n_model=n_model, band=band, corr=corr)

        if psf:
            bulge_arr = -2.5*np.log10(convolve(10**(-0.4*bulge_arr), self.kernal[band], mode='same'))
            bar_arr = -2.5*np.log10(convolve(10**(-0.4*bar_arr), self.kernal[band], mode='same'))
            disk_arr = -2.5*np.log10(convolve(10**(-0.4*disk_arr), self.kernal[band], mode='same'))
            comb_arr = -2.5*np.log10(convolve(10**(-0.4*comb_arr), self.kernal[band], mode='same'))

        reds, greens, blues = self.alpha_colormap()

        def sub_mag(m1, m2):
            return -2.5*np.log10(np.abs(10**(-0.4*m1) - 10**(-0.4*m2)))
    
        fig, (ax1, ax2, ax3) = plt.subplots(figsize=(18, 6), ncols=3, dpi=100)
        sky_mag = self.gobj[band].cutout['mag_raw'].copy()
        sky_lim = self.gobj[band].brick['psfdepth']
        sky_mag[np.isnan(sky_mag)] = sky_lim
        ci = np.round(self.center).astype(int)
        mag_c = min(19, np.mean(self.image[band][ci[0]-1:ci[0]+2, ci[1]-1:ci[1]+2]))
        ax1.imshow(sky_mag, origin='lower', cmap='gray', vmax=30, vmin=mag_c)

        wcs = self.gobj[band].cutout['wcs']
        (sn_ra, sn_dec) = self.gobj[band].gal['sn']
        c = SkyCoord(ra=sn_ra*u.degree, dec=sn_dec*u.degree, frame='icrs')
        sn_px = wcs.world_to_pixel(c)

        ax2.imshow(disk_arr, cmap=blues, origin='lower', vmax=25) if (n_model in [0, 2, 3]) else None
        ax2.imshow(bar_arr, cmap=greens, origin='lower', vmax=25) if n_model == 3 else None
        ax2.imshow(bulge_arr, cmap=reds, origin='lower', vmax=25) if n_model != 0 else None
        
        ax2.scatter(sn_px[0], sn_px[1], c='cyan', s=40, lw=2, ec='cyan', zorder=10, marker='x', label='SNe Ia')

        ax3.imshow(sub_mag(sky_mag, comb_arr), origin='lower', cmap='gray', vmax=30, vmin=mag_c)
        plt.tight_layout()


        rp = [3, 5, 9, 2][n_model-1]
        arcsec2px = np.array([0.262, 0.262, 1, 1])
        if n_model == 3:
            Re_bulge = self.decomp[n_model-1][0][8]/0.262
            self.patch_super_ellipse([Re_bulge, Re_bulge, 0, 2], self.center, ax2, 'darkred', label='Bulge $R_e$')
            self.patch_super_ellipse(self.decomp[n_model-1][0][rp:rp+4]/arcsec2px, self.center, ax2, 'green', label='Bar $R_e$')

            disk_a, disk_b, disk_pa, disk_n = self.decomp[n_model-1][0][rp+4:rp+8]
            u0_g, u0_r = self.decomp[n_model-1][0][4:6]
            u0_disk = {'g': u0_g, 'r': u0_r}[band]
            ue_disk, (disk_a, disk_b) = self.transform(u0_disk, np.array([disk_a, disk_b]), n=1)
            self.patch_super_ellipse(np.array([disk_a, disk_b, disk_pa, disk_n])/arcsec2px, self.center, ax2, 'blue', label='Disk $R_e$')

        elif n_model == 2:
            self.patch_super_ellipse(self.decomp[n_model-1][0][rp:rp+4]/arcsec2px, self.center, ax2, 'darkred', label='Bulge $R_e$')
            u0_g, u0_r = self.decomp[n_model-1][0][2:rp-1]
            u0_disk = {'g': u0_g, 'r': u0_r}[band]
            disk_a, disk_b, disk_pa, disk_n = self.decomp[n_model-1][0][rp+4:rp+8]
            ue_disk, (disk_a, disk_b) = self.transform(u0_disk, np.array([disk_a, disk_b]), n=1)
            self.patch_super_ellipse(np.array([disk_a, disk_b, disk_pa, disk_n])/arcsec2px, self.center, ax2, 'blue', label='Disk $R_e$')

        elif n_model == 1:
            self.patch_super_ellipse(self.decomp[n_model-1][0][rp:rp+4]/arcsec2px, self.center, ax2, 'darkred', label='Bulge $R_e$')
        
        elif n_model == 0:
            disk_a, disk_b, disk_pa, disk_n = self.decomp[n_model-1][0][rp:rp+4]
            u0_g, u0_r = self.decomp[n_model-1][0][:rp]
            u0_disk = {'g': u0_g, 'r': u0_r}[band]
            ue_disk, (disk_a, disk_b) = self.transform(u0_disk, np.array([disk_a, disk_b]), n=1)
            self.patch_super_ellipse(np.array([disk_a, disk_b, disk_pa, disk_n])/arcsec2px, self.center, ax2, 'blue', label='Disk $R_e$')
        
        if isophote:
            wi = 0.1
            mag_i = np.where((self.image[band].T > isophote-wi) & (self.image[band].T < isophote+wi))
            ax1.plot(mag_i[0], mag_i[1], 'g.', c='green', ms=2, zorder=5, label=f'{isophote} mag/arcsec$^2$')

            pars, errs, check = self.SB_isophote(isophote, band, n_model)
            self.patch_super_ellipse(pars, self.center, ax1, 'lime') if check == 0 else None

        ax2.legend(framealpha=1)
           


def get_corrections(x, tau):
    # x = 1 - cos(i) = 1-b/a = ecc
    def BV_g(B, V):
        return V + 0.6 * (B-V)#  - 0.12

    def BV_r(B, V):
        return V - 0.42 * (B-V) # + 0.11
    
    B_disk_proj = disk_proj[disk_proj['Band'] == 'V']
    disk_proj_h_pars = B_disk_proj['Ri_R0'].values
    disk_proj_u0_pars = B_disk_proj['delta_SB0_'].values
    disk_proj_h = np.poly1d(disk_proj_h_pars[::-1])(x)
    disk_proj_u0 = np.poly1d(disk_proj_u0_pars[::-1])(x)

    B_disk_dust = disk_dust[(disk_dust['Band'] == 'B') & (disk_dust['tau'] == tau)]
    V_disk_dust = disk_dust[(disk_dust['Band'] == 'V') & (disk_dust['tau'] == tau)]
    disk_dust_u0_B_pars = B_disk_dust['SBapp_SBi'].values
    disk_dust_u0_V_pars = V_disk_dust['SBapp_SBi'].values
    disk_dust_h_pars = V_disk_dust['Rapp_Ri'].values
    disk_dust_h = np.poly1d(disk_dust_h_pars[::-1])(x)
    disk_dust_u0_B = np.poly1d(disk_dust_u0_B_pars[::-1])(x)
    disk_dust_u0_V = np.poly1d(disk_dust_u0_V_pars[::-1])(x)
    disk_dust_u0_g = BV_g(disk_dust_u0_B, disk_dust_u0_V)
    disk_dust_u0_r = BV_r(disk_dust_u0_B, disk_dust_u0_V)

    return disk_proj_h, disk_proj_u0, disk_dust_h, disk_dust_u0_g, disk_dust_u0_r
        


def K_corr(z, template):
    filter_g, filter_r = fset['gDEC'], fset['rDEC']
    filter_g.zp, filter_r.zp  = 22.5, 22.5

    k_g, _ = kcorr.K(template.wl.values, template.fl.values, filter_g, filter_g, z=z)
    k_r, _ = kcorr.K(template.wl.values, template.fl.values, filter_r, filter_r, z=z)
    return [k_g, k_r]