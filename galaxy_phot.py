import bs4
import lxml
import ztfidr
import requests
import sympy as smp
import numpy as np
import pandas as pd
from io import BytesIO
from extinction import fm07
from astropy.io import fits
from astropy.wcs import WCS
from snpy import kcorr, fset
from astropy import units as u
from astropy.table import Table
import matplotlib.pyplot as plt
from scipy.integrate import quad
from skimage import measure, draw
from scipy.signal import convolve2d
# from astroquery.ipac.ned import Ned
from astroquery.vizier import Vizier
from astropy.cosmology import Planck18
from matplotlib.patches import Ellipse
from scipy.interpolate import interp1d, RegularGridInterpolator
from scipy.odr import Model, Data, ODR
from skimage.measure import EllipseModel
from astropy.coordinates import SkyCoord
from scipy.ndimage import gaussian_filter
from scipy.special import gamma, gammainc
from matplotlib.colors import ListedColormap
from skimage.measure import label as ConnectRegion
from astroquery.exceptions import RemoteServiceError
from scipy.optimize import curve_fit, fsolve, minimize
from astropy.convolution import convolve_fft, Gaussian2DKernel

sample = ztfidr.get_sample()
host_data = ztfidr.io.get_host_data()
Vizier.ROW_LIMIT = 1000
disk_proj = Vizier.get_catalogs('J/A+A/553/A80/tablea1')[0].to_pandas()
bulge_proj = Vizier.get_catalogs('J/A+A/553/A80/tablea4')[0].to_pandas()
disk_dust = Vizier.get_catalogs('J/A+A/553/A80/dexp')[0].to_pandas()
bulge_dust = Vizier.get_catalogs('J/A+A/553/A80/ebser')[0].to_pandas()

df_update = pd.read_csv('dr2/tables/.dataset_creation/host_prop/host_match/ztfdr2_matched_hosts.csv', index_col=0)
df_gp = pd.read_csv('csv_files/ZTFidr_all_gri_kcor_salt_gp_fitted_clipped_z_0.3.csv', index_col=0)
df10 = Table.read('fits_files/survey-bricks-dr10-south.fits.gz', format='fits')
df9 = Table.read('fits_files/survey-bricks-dr9-north.fits.gz', format='fits')
df_iso = pd.read_csv('csv_files/galaxy_isophotes.csv', index_col=0)
df_bd1 = pd.read_csv('csv_files/one_decomp.csv', index_col=0)
df_bd2 = pd.read_csv('csv_files/two_decomp.csv', index_col=0)
df_bd3 = pd.read_csv('csv_files/three_decomp.csv', index_col=0)
df_bd0 = pd.read_csv('csv_files/disk_decomp.csv', index_col=0)

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
            pass
            # query = Vizier.query_object(host_name, catalog='J/AJ/90/1681')
            # sc = SkyCoord(ra=query[0]['_RA.icrs'][0], dec=query[0]['_DE.icrs'][0],  unit=(u.hourangle, u.deg))
            # self.gal = {'host': [sc.ra.deg, sc.dec.deg], 'z': 0.004, 'sn': [sc.ra.deg, sc.dec.deg],
            #             'A': 0, 'z_source': 'None'}
        
        if catalog == 'sdss':
            pass
            # self.gal = {'host': [qs[0][host_name]['_RA'], qs[0][host_name]['_DE']], 'z': qs[0][host_name]['z'], 
            #             'sn': [qs[0][host_name]['_RA'], qs[0][host_name]['_DE']], 'A': 0, 'z_source': 'None'}
            # print(self.gal['z']) if self.verbose else None

    def init_dr2(self, sn_name):
        self.host_name = 'DR2'
        self.sn_name = sn_name

        z = sample.data['redshift'][sn_name]
        host_ra, host_dec = df_update[['ra', 'dec']].loc[sn_name]
        if np.isnan(host_ra) or np.isnan(host_dec):
            host_ra, host_dec = host_data[['host_ra', 'host_dec']].loc[sn_name]
            
        mwebv, mwr_v, sn_ra, sn_dec, z_source = sample.data[['mwebv', 'mwr_v', 'ra', 
                                                                'dec', 'source']].loc[sn_name]
        if host_dec == -80:
            host_ra, host_dec = sn_ra, sn_dec
            print('no host') if self.verbose else None
        print(sn_name, host_ra, host_dec, z, z_source) if self.verbose else None
        self.gal = {'host': [host_ra, host_dec], 'z': z, 'sn': [sn_ra, sn_dec],
                     'A': mwebv * mwr_v, 'z_source': z_source}
           
    def get_cutout(self, size, band, scale=0.262):
        ra, dec = self.gal['host']

        def legacy_url(survey):
            layer = 'ls-dr10' if survey == 'legacy' else 'sdss'
            service = 'https://www.legacysurvey.org/viewer/'
            return f'{service}fits-cutout?ra={ra}&dec={dec}&layer={layer}&pixscale={scale}&bands={band}&size={size}&subimage'

        url = legacy_url('legacy')
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
        flux = flux - self.brick['sky']
        return mag_soft(flux, zp) if soft else mag_(flux, zp)

    @staticmethod
    def calc_aperature(size, redshift, scale):
        if size == 'z':
            if np.sign(redshift) == -1:
                return 500
            else:
                aperature = 0.07 
                dist = Planck18.luminosity_distance(redshift)
                return min(int(np.rad2deg(aperature/dist.value)*(3600/scale)),  500)
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
            raise FileNotFoundError
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
        if len(attach) > 1:
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
        
    def get_image(self, source, size, band, scale):
        path = f'dr2_fits_{band}/{self.sn_name}.fits'
        output_size = self.calc_aperature(size, self.gal['z'], scale)
        if source == 'query':
            fits_data = self.get_cutout(output_size, band, scale)
        elif source == 'save':
            fits_data = fits.open(path)
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
    def __init__(self, target_name, verbose, mask, source, size='z', catalog='ztf'):
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

        if len(self.gobj['g'].cutout) == 0:
            print('no image') if verbose else None
            return

        self.image = {'g': self.gobj['g'].cutout['mag'], 'r': self.gobj['r'].cutout['mag']}
        self.invvar = {'g': self.gobj['g'].cutout['invvar'], 'r': self.gobj['r'].cutout['invvar']} 
        self.flux = {'g': self.gobj['g'].cutout['flux'], 'r': self.gobj['r'].cutout['flux']} 
        self.mask = mask
        self.center = (np.array(self.image['g'].shape))/2

    def plot_fit(self, isophotes, band, width=0.1, mask=False, zoom=False):
        fig, (ax1, ax2) = plt.subplots(figsize=(12, 6), dpi=100, ncols=2)
        mag, wcs = self.image[band], self.gobj[band].cutout['wcs']
        ax2.imshow(mag, cmap='gray', origin='lower')

        (sn_ra, sn_dec) = self.gobj[band].gal['sn']
        c = SkyCoord(ra=sn_ra*u.degree, dec=sn_dec*u.degree, frame='icrs')
        sn_px = wcs.world_to_pixel(c)
        ax2.scatter(sn_px[0], sn_px[1], c='cyan', s=40, lw=2, fc='None', ec='cyan')

        if np.any(mask):
            for (ra, dec, r) in mask:
                coords = SkyCoord(ra=ra*u.degree, dec=dec*u.degree, frame='icrs')
                m_px = wcs.world_to_pixel(coords)
                m_patch = Ellipse((m_px[0], m_px[1]), 2*r, 2*r, 0, edgecolor='black', facecolor='black', zorder=10, alpha=0.5)
                ax1.add_patch(m_patch)
  
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
        
    def prep_pixels(self, isophote, window, band):
        kernal3 = np.array([[ 0,  1,  1],
                            [ -1,  0,  1],
                            [-1,  -1,  0,]])

        kernal_ = 1/(kernal3 + isophote)
        kernal_unit = kernal_ / np.sum(kernal_)
        convolve_1 = convolve2d(self.image[band], kernal_unit, mode='same')
        convolve_2 = convolve2d(convolve_1, kernal_unit[::-1], mode='same')

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

    def extract_regions(self, contour, mask, band):
        def get_pixels(region, connect): return np.stack(np.where(connect == region))

        binary_image = np.zeros_like(self.image[band], dtype=np.uint8)
        binary_image[contour[0], contour[1]] = 1
        connect_ = ConnectRegion(binary_image, connectivity=2, background=0)
        region_count = np.asarray(np.unique(connect_, return_counts=True)).T[1:].T
        galaxy_region = max(region_count.T, key=lambda x: x[1])[0]
        # galaxy_region = min(region_count.T, key=lambda x: np.sum(np.mean(get_pixels(x[0], connect_).T , axis=0) - self.center)**2)[0]

        if np.any(mask):
            for (ra, dec, r) in mask:
                coords = SkyCoord(ra=ra*u.degree, dec=dec*u.degree, frame='icrs')
                m_px = self.gobj[band].cutout['wcs'].world_to_pixel(coords)
                ell_px = draw.ellipse(m_px[0], m_px[1], r, r, rotation=0)
                mask = np.ones_like(self.image[band])
                px_cutoff = (ell_px[0] < self.image[band].shape[0]) & (ell_px[1] < self.image[band].shape[0])
                mask[ell_px[0][px_cutoff], ell_px[1][px_cutoff]] = 0
                connect_ = connect_ * mask

        return get_pixels(galaxy_region, connect_).T         
           
    def contour_fit(self, isophote, mask, band):
        
        def fit_px(window, return_err):
            all_pixels = self.prep_pixels(isophote, window, band)
            try:
                connect_all = self.extract_regions(all_pixels, mask, band)
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
            except (RuntimeError, ValueError):
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
        result = minimize(fit_px, 0.2, args=(True), bounds=[(0.1, 0.5)], method='Powell', tol=0.1)
        self.contours[band][isophote] = fit_px(result.x, return_err=False)

    def main_run(self):
        step = 0.2
        for band in ['g', 'r']:
            max_iso = min(round(self.gobj[band].brick['psfdepth'], 1), 25)
            max_iso = 24 if max_iso == 0 else max_iso
            for iso in np.arange(max_iso, 16, -step):
                iso = np.round(iso, 1)
                print(f'{band}-{iso}') if (iso%1==0.0 and self.verbose) else None
                self.contour_fit(iso, mask=[], band=band)
                if len(self.contours[band][iso][1][1]) == 0:
                    del self.contours[band][iso]
                    if iso < 23:
                        break
        
    def load_main_run(self):
        iso_data = df_iso[df_iso['sn_name'] == self.name]
        if iso_data.iloc[0]['band'] != 'error':    
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
        self.mask = gd.mask
        self.decomp = [[np.zeros(7), np.zeros(7)], [np.zeros(13), np.zeros(13)], [np.zeros(17), np.zeros(17)], [np.zeros(6), np.zeros(6)]]
        mags_g, iso_data_g = self.extract_data('g')
        mags_r, iso_data_r = self.extract_data('r')
        self.mags = {'g': mags_g, 'r': mags_r}
        self.iso_data = {'g': iso_data_g, 'r': iso_data_r}
        if (len(iso_data_g) > 0) and  (len(iso_data_r) > 0):
            self.center, self.iso_stat = self.contour_stats()

    def extract_data(self, band):
        key_targs = list(self.contours[band].keys())
        if len(key_targs) == 0:
            return [], []
        pars_ = np.array([self.contours[band][iso_key][0] for iso_key in key_targs if len(self.contours[band][iso_key][0]) > 0]).T

        offsets = np.sqrt((pars_[4]-self.center[0])**2 + (pars_[5]-self.center[1])**2)
        cuts = np.where((offsets < 5))

        mags = np.array(key_targs)[cuts]
        pars_ = pars_.T[cuts].T.reshape((12, -1))
        if len(pars_) == 0:
            return [], []

        return mags, pars_
    
    def contour_stats(self):
        centerx = np.median(np.concatenate([self.iso_data['g'][4], self.iso_data['g'][4]]))
        centery = np.median(np.concatenate([self.iso_data['g'][5], self.iso_data['g'][5]]))
        pa = np.median(np.concatenate([self.iso_data['g'][2], self.iso_data['g'][2]]))

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

    def patch_super_ellipse(self, pars, center, ax, color, label=None):
        t_r = np.arange(-np.pi/2, np.pi/2, 0.01)+ pars[2]
        xse, yse = self.super_ellipse(t_r, *pars, polar=False) 
        xse_t = np.concatenate([xse, -xse])+ center[0]
        yse_t = np.concatenate([yse, -yse])+ center[1]
        ax.plot(xse_t, yse_t, 'r-', color=color, zorder=6, label=label)
        ax.plot(xse_t[[0, -1]], yse_t[[0, -1]], 'r-', color=color, zorder=10)
    
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
        def func(b, n): return 1 - 2*gammainc(2*n[0], b)
        return fsolve(func, 1.9992*n - 0.3271, args=[n])

    def transform(self, u0, h, n):
        b = np.vectorize(self.get_b)(n)
        return u0 + 2.5*b/np.log(10), b**n*h
    
    def back_transform(self, ue, Re, n):
        b = np.vectorize(self.get_b)(n)
        return ue - 2.5*b/np.log(10), Re/b**n

    @staticmethod
    def bulge(x, ue, Re, n):
        # b = np.vectorize(BDdecomp.get_b)(n)
        b = 1.9992*n - 0.3271
        return ue + 2.5*b/np.log(10) * ((np.abs(x)/Re)**(1/n) - 1)
    
    @staticmethod
    def disk(x, u0, h):
        return u0 + 2.5/np.log(10)*(np.abs(x)/h)

    @staticmethod
    def add_mag(m1, m2):
        return -2.5*np.log10(10**(-0.4*m1) + 10**(-0.4*m2))

    @staticmethod
    def combine(x, ue, u0, Re, h, n):
        return BDdecomp.add_mag(BDdecomp.bulge(x, ue, Re, n), BDdecomp.disk(x, u0, h))
    
    @staticmethod
    def combine_3(x, ue_bulge, ue_bar, u0, Re_bulge, Re_bar, h, n_bulge, n_bar):
        bar_bulge = BDdecomp.add_mag(BDdecomp.bulge(x, ue_bulge, Re_bulge, n_bulge), BDdecomp.bulge(x, ue_bar, Re_bar, n_bar))
        return BDdecomp.add_mag(bar_bulge, BDdecomp.disk(x, u0, h))
       
    def stabilize(self, phi, theta, center0, center1, a, b, pa, n):
        vec = center1 - center0
        beta = np.arctan2(vec[1], vec[0])
        v_mag = np.sqrt(vec[0]**2 + vec[1]**2)
        r =  self.super_ellipse(phi, a, b, pa, n)
        return v_mag * np.sin(theta - beta) - r * np.sin((phi - theta))

    def target_angle(self, c_r, theta):
        target_ang = np.zeros(len(c_r))
        for i, row_i in enumerate(c_r):
            ai, bi, pai, ni, xci, yci, *errs = row_i
            phi_i = fsolve(self.stabilize, theta, args=(theta, self.center, np.array([xci, yci]), ai, bi, pai, ni))
            target_ang[i] =  self.super_ellipse(phi_i, ai, bi, pai, ni)
        return target_ang

    def plot_gal_iso(self, spokes, zoom=True):
        fig, axis = plt.subplots(figsize=(12, 6), dpi=100, ncols=2)
        theta_arr =  {'g': np.linspace(0, 360-360/spokes, spokes), 'r': np.linspace(180/spokes, 360-180/spokes, spokes)}
        for band, ax in zip(['g', 'r'], axis):
            ax.imshow(self.image[band], cmap='gray', origin='lower')
        
            a_lim = self.iso_data[band][0].max()
            min_i = np.argmin(self.iso_data[band][0])
            xc_ref, yc_ref = self.iso_data[band][4][min_i], self.iso_data[band][5][min_i]
            ax.set_xlim([xc_ref-a_lim-10, xc_ref+a_lim+10]) if zoom else None
            ax.set_ylim([yc_ref-a_lim-10, yc_ref+a_lim+10]) if zoom else None

            for theta in theta_arr[band]:
                for row_i in self.iso_data[band].T[::-2]:
                    ai, bi, pai, ni, xci, yci, *errs = row_i
                    if theta is not None:
                        phi = fsolve(self.stabilize, np.deg2rad(theta), args=(np.deg2rad(theta), self.center, 
                                                                            np.array([xci, yci]), ai, bi, pai, ni))
                        xp, yp = self.super_ellipse(phi, ai, bi, pai, ni, polar=False)
                        ax.scatter(xp+xci, yp+yci, color='blue', s=10, zorder=15)

                    self.patch_super_ellipse((ai, bi, pai, ni), (xci, yci), ax, band)
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
        rm = np.linalg.norm(np.stack([xm, ym]).T - self.center, axis=2)

        theta_top = np.arccos(np.dot(np.stack([xm, ym]).T - self.center, np.array([0,1]))/rm)[len(n_size)//2:]
        theta_bot = np.arccos(np.dot(np.stack([xm, ym]).T - self.center, np.array([0,-1]))/rm)[:len(n_size)//2]
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

    def get_init_pars(self):
        r_major = self.iso_data['g'][0]*0.262
        r_minor = self.iso_data['g'][1]*0.262
        mags = self.iso_data['g'][11]
        mag_err = self.iso_data['g'][10]
        pa = self.iso_stat['pa']
        max_r = np.mean(self.center)*0.262

        c1_a = curve_fit(self.bulge, r_major, mags, sigma=mag_err, p0=[20, max_r/3, 2], bounds=[[16, 0.1, 0.1], [25, max_r, 8]])[0]
        c1_b = curve_fit(self.bulge, r_minor, mags, sigma=mag_err, p0=[20, max_r/3, 2], bounds=[[16, 0.1, 0.1], [25, max_r, 8]])[0]
        p0_1 = [(c1_a[0]+c1_b[0])/2, (c1_a[0]+c1_b[0])/2-1, (c1_a[2]+c1_b[2])/2, c1_a[1], c1_b[1], pa, 2]

        c2_a = curve_fit(self.combine, r_major, mags, sigma=mag_err, p0=[21, 20, max_r/4, max_r/2, 1], bounds=[[16, 16, 0.5, 1, 0.5], [24, 23, max_r, max_r, 3]])[0]
        c2_b = curve_fit(self.combine, r_minor, mags, sigma=mag_err,  p0=[21, 20, max_r/4, max_r/2, 1], bounds=[[16, 16, 0.5, 1, 0.5], [24, 23, max_r, max_r, 3]])[0]
        p0_2 = [(c2_a[0]+c2_b[0])/2, (c2_a[0]+c2_b[0])/2-1, (c2_a[1]+c2_b[1])/2, (c2_a[1]+c2_b[1])/2+1, (c2_a[4]+c2_b[4])/2, 
                c2_a[2], c2_b[2], pa, 2, c2_a[3], c2_b[3], pa, 2]
        return p0_1, p0_2
    
    def create_data(self, spokes, plot=False):
        angle_targets = {'g': np.linspace(0, 360-360/spokes, spokes), 'r': np.linspace(180/spokes, 360-180/spokes, spokes)}
        x_data_g = np.concatenate([self.target_angle(self.iso_data['g'].T, np.deg2rad(ang_i)) for ang_i in angle_targets['g']]) * 0.262
        x_err_g = np.concatenate([self.error_prop_SE(ang_i, *[self.iso_data['g'][i] for i in [0,1,2,3,6,7,8,9]]) for ang_i in angle_targets['g']]) * 0.262
        y_data_g = np.tile(self.iso_data['g'][11], spokes)
        y_err_g = np.tile(self.iso_data['g'][10], spokes)
        
        x_data_r = np.concatenate([self.target_angle(self.iso_data['r'].T, np.deg2rad(ang_i)) for ang_i in angle_targets['r']]) * 0.262
        x_err_r = np.concatenate([self.error_prop_SE(ang_i, *[self.iso_data['r'][i] for i in [0,1,2,3,6,7,8,9]]) for ang_i in angle_targets['r']]) * 0.262
        y_data_r = np.tile(self.iso_data['r'][11], spokes)
        y_err_r = np.tile(self.iso_data['r'][10], spokes)

        if plot:
            ci = np.round(self.center).astype(int)
            cmag_g = self.image['g'][ci[0]-1:ci[0]+2, ci[1]-1:ci[1]+2]
            cmag_r = self.image['r'][ci[0]-1:ci[0]+2, ci[1]-1:ci[1]+2]
            x_, x_err = 0.262, 0.262/3
            y_g, y_g_err = np.min(cmag_g), np.std(cmag_g)
            y_r, y_r_err = np.min(cmag_r), np.std(cmag_r)
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


    def fit_functions(self, spokes):
        xm, ym, rm, thetam = self.get_meshgrid()
        std_g = self.gobj['g'].brick['psfsize']/(2*np.sqrt(2*np.log(2)))
        std_r = self.gobj['r'].brick['psfsize']/(2*np.sqrt(2*np.log(2)))
        kernal_g = Gaussian2DKernel(x_stddev=std_g, y_stddev=std_g)
        kernal_r = Gaussian2DKernel(x_stddev=std_r, y_stddev=std_r)
        angle_targets = {'g': np.linspace(0, 360-360/spokes, spokes), 'r': np.linspace(180/spokes, 360-180/spokes, spokes)}

        def psf_convolve(model):
            model_grid_g = model(rm*0.262, thetam, 'g')
            model_grid_r = model(rm*0.262, thetam, 'r')
            model_psf_g = -2.5*np.log10(convolve_fft(10**(-0.4*model_grid_g), kernal_g))
            model_psf_r = -2.5*np.log10(convolve_fft(10**(-0.4*model_grid_r), kernal_r))
            interp_g = RegularGridInterpolator((xm[0]*0.262, ym.T[0]*0.262), model_psf_g.T)
            interp_r = RegularGridInterpolator((xm[0]*0.262, ym.T[0]*0.262), model_psf_r.T)
            return {'g': interp_g, 'r': interp_r}
        
        def model_psf(r, theta, band, psf_interp):
            x, y = r*np.cos(np.deg2rad(theta))+self.center[0]*0.262, r*np.sin(np.deg2rad(theta))+self.center[1]*0.262
            return psf_interp[band]((x, y))
        
        def disk_2D_model(pars, all_data):
            rp = 2
            u0_g, u0_r = pars[:rp]
            u0 = {'g': u0_g, 'r': u0_r}
            data = {'g': all_data[:len(self.iso_data['g'].T)*spokes], 'r': all_data[len(self.iso_data['g'].T)*spokes:]}
            def model(x, theta, band):
                h = self.super_ellipse(theta, *pars[rp:rp+4])
                return self.disk(x, u0[band], h)
            
            psf_interp = psf_convolve(model)
            data_g = np.concatenate([model_psf(data['g'].reshape(spokes, -1)[i], angle_targets['g'][i], 'g', psf_interp) for i in range(spokes)])
            data_r = np.concatenate([model_psf(data['r'].reshape(spokes, -1)[i], angle_targets['r'][i], 'r', psf_interp) for i in range(spokes)])
            return np.concatenate([data_g, data_r])

        def bulge_2D_model(pars, all_data):
            rp = 3
            ue_g, ue_r, n = pars[:rp]
            ue = {'g': ue_g, 'r': ue_r}
            data = {'g': all_data[:len(self.iso_data['g'].T)*spokes], 'r': all_data[len(self.iso_data['g'].T)*spokes:]}
            def model(x, theta, band):
                Re = self.super_ellipse(theta, *pars[rp:rp+4])
                return self.bulge(x, ue[band], Re, n)
            
            psf_interp = psf_convolve(model)
            data_g = np.concatenate([model_psf(data['g'].reshape(spokes, -1)[i], angle_targets['g'][i], 'g', psf_interp) for i in range(spokes)])
            data_r = np.concatenate([model_psf(data['r'].reshape(spokes, -1)[i], angle_targets['r'][i], 'r', psf_interp) for i in range(spokes)])
            return np.concatenate([data_g, data_r])

        def bulge_disk_2D_model(pars, all_data):
            rp = 5
            ue_g, ue_r, u0_g, u0_r, n = pars[:rp]
            ue = {'g': ue_g, 'r': ue_r}
            u0 = {'g': u0_g, 'r': u0_r}
            data = {'g': all_data[:len(self.iso_data['g'].T)*spokes], 'r': all_data[len(self.iso_data['g'].T)*spokes:]}
            def model(x, theta, band):
                Re = self.super_ellipse(theta, *pars[rp:rp+4])
                h = self.super_ellipse(theta, *pars[rp+4:rp+8])
                return self.combine(x, ue[band], u0[band], Re, h, n)
            
            psf_interp = psf_convolve(model)
            data_g = np.concatenate([model_psf(data['g'].reshape(spokes, -1)[i], angle_targets['g'][i], 'g', psf_interp) for i in range(spokes)])
            data_r = np.concatenate([model_psf(data['r'].reshape(spokes, -1)[i], angle_targets['r'][i], 'r', psf_interp) for i in range(spokes)])
            return np.concatenate([data_g, data_r])
        
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
                return self.combine_3(x, ue_bulge[band], ue_bar[band], u0[band], Re_bulge, Re_bar, h, n_bulge, n_bar)
            
            psf_interp = psf_convolve(model)            
            data_g = np.concatenate([model_psf(data['g'].reshape(spokes, -1)[i], angle_targets['g'][i], 'g', psf_interp) for i in range(spokes)])
            data_r = np.concatenate([model_psf(data['r'].reshape(spokes, -1)[i], angle_targets['r'][i], 'r', psf_interp) for i in range(spokes)])
            return np.concatenate([data_g, data_r])

        return bulge_2D_model, bulge_disk_2D_model, bulge_bar_disk_2D_model, disk_2D_model


    def main_BD(self, spokes=12, mode=0, verbose=True, disk=1, init_2=[]):
        bulge_2D_model, bulge_disk_2D_model, bulge_bar_disk_2D_model, disk_2D_model = self.fit_functions(spokes)

        self.decomp_data = self.create_data(spokes)
        x_data, y_data, x_err, y_err = self.decomp_data['all']
        data = Data(x_data, y_data, we=1/x_err**2, wd=1/y_err**2)

        p0_1, p0_2 = self.get_init_pars()
        odr_1 = ODR(data, Model(bulge_2D_model), beta0=p0_1, maxit=10)
        odr_1.set_job(fit_type=mode)
        output_1 = odr_1.run()
        self.decomp[0] = [[*output_1.beta], [*output_1.sd_beta]]
        
        if output_1.beta[2] > disk:
            p0_2 = init_2 if init_2 else p0_2
            odr_2 = ODR(data, Model(bulge_disk_2D_model), beta0=p0_2, maxit=10)
            odr_2.set_job(fit_type=mode)
            output_2 = odr_2.run()
            self.decomp[1] = [[*output_2.beta], [*output_2.sd_beta]]

    
    def bulge_bar_disk_decomp(self, spokes=12, mode=0, verbose=False):
        bulge_2D_model, bulge_disk_2D_model, bulge_bar_disk_2D_model, disk_2D_model = self.fit_functions(spokes)  
        
        if self.decomp[1][0] != 0:
            self.decomp_data = self.create_data(spokes)
            x_data, y_data, x_err, y_err = self.decomp_data['all']
            data = Data(x_data, y_data, we=1/x_err**2, wd=1/y_err**2)
            ue_g, ue_r, u0_g, u0_r, n, Re_a, Re_b, Re_pa, Re_n, h_a, h_b, h_pa, h_n = self.decomp[1][0]
            p0_3 = [ue_g-0.5, ue_r-0.5, ue_g+1, ue_r+1, u0_g, u0_r, n, 0.3, Re_b, Re_a, Re_b, Re_pa, Re_n, h_a, h_b, h_pa, h_n]
            odr_3 = ODR(data, Model(bulge_bar_disk_2D_model), beta0=p0_3, maxit=10)
            odr_3.set_job(fit_type=mode)
            output_3 = odr_3.run()
            self.decomp[2] = [output_3.beta, output_3.sd_beta]
    
    def disk_decomp(self, spokes=12, mode=0, verbose=False):
        bulge_2D_model, bulge_disk_2D_model, bulge_bar_disk_2D_model, disk_2D_model = self.fit_functions(spokes)  
        self.decomp_data = self.create_data(spokes)
        x_data, y_data, x_err, y_err = self.decomp_data['all']
        data = Data(x_data, y_data, we=1/x_err**2, wd=1/y_err**2)

        ci = np.round(self.center).astype(int)
        cmag_g = self.image['g'][ci[0]-1:ci[0]+2, ci[1]-1:ci[1]+2]
        cmag_r = self.image['r'][ci[0]-1:ci[0]+2, ci[1]-1:ci[1]+2]
        p0_0 = [np.min(cmag_g), np.min(cmag_r),  np.mean(self.iso_data['g'][0]*0.262), np.mean(self.iso_data['g'][1]*0.262), self.iso_stat['pa'], 2]
        odr_0 = ODR(data, Model(disk_2D_model), beta0=p0_0, maxit=10)
        odr_0.set_job(fit_type=mode)
        output_0 = odr_0.run()

        BIC_4 = self.BIC(output_0.beta, x_data, y_data, disk_2D_model)
        self.decomp[3] = [output_0.beta, output_0.sd_beta]

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

 
        bulge_2D_model, bulge_disk_2D_model, bulge_bar_disk_2D_model, disk_2D_model = self.fit_functions(spokes=12)
        x_data, y_data, x_err, y_err = self.decomp_data['all']

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
        return n*gamma(2*n)*np.exp(b)/b**(2*n) * (Re/h)**2 * (Ie/I0)
    
    @staticmethod
    def bb_ratio(ue1, ue2, n1, n2, Re1, Re2):
        Ie1, Ie2 = 10**(-0.4*ue1), 10**(-0.4*ue2)
        return gamma(2*n1 + 1)/gamma(2*n2 + 1) * (Re1/Re2)**2 * (Ie1/Ie2)

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
                rss_lim = 0.025 if i!=3 else 0.03
                if (gal_type in labels) and (df_g[f'RSS_{i}'].values > rss_lim):
                    labels.remove(gal_type)
            return labels

        def elliptical_check(labels, ell, col, SE_n):
            col_k = col - np.subtract(*self.kcorr['Ell'])
            condition = (ell < 0.7) and (col_k > 0.5) and (col_k < 2) and (SE_n > 1.8) and (SE_n < 2.2) and (self.decomp[0][0] != 0) 
            if not condition:
                labels.remove('bulge')
            return labels
        
        def disk_check(labels, ell, col, SE_n):
            col_k = col - np.subtract(*self.kcorr['Sab'])
            self.d_corr = get_corrections(ell, 1, trad=10, tau=0.5)
            condition = (ell < 0.6) and (col_k > 0) and (col_k < 0.5) and (SE_n > 1) and (SE_n < 3) and (self.decomp[3][0] != 0) 
            if not condition:
                labels.remove('disk')
            return labels
        
        def disk_subtype(BD_color, BD_ratio):
            print(np.round(BD_color, 3), np.round(BD_ratio, 3)) if verbose else None
            if (BD_color > 0) & (BD_color < 0.1) & (BD_ratio > 0.15) & (BD_ratio < 0.6):
                return 'S0'
            elif (BD_color >= 0.1) & (BD_color < 0.45) & (BD_ratio > 0.01) & (BD_ratio < 0.5):
                return 'Sab'
            elif (BD_color >= 0.45) & (BD_color < 1.6) & (BD_ratio > 0.001) & (BD_ratio < 0.15):
                return 'Sc'
            else:
                return 'none'
        
        def bulge_disk_check(labels, bulge_pars, disk_pars):
            Re, b_ell, b_ug, b_ur, b_n, b_SE = bulge_pars
            h, d_ell, d_ug, d_ur, d_SE = disk_pars

            if self.decomp[1][0] == 0:
                labels.remove('bulge+disk')
                return labels

            corrs = get_corrections(d_ell, b_n, trad=10, tau=0.5)
            h_corr = corrs['disk_proj'][0] * corrs['disk_dust'][0]
            u0g_corr = corrs['disk_proj'][1] + corrs['disk_dust'][1]
            u0r_corr = corrs['disk_proj'][1] + corrs['disk_dust'][2]
            Re_corr = corrs['bulge_proj'][0] * corrs['bulge_dust'][0]
            n_corr = corrs['bulge_proj'][1] + corrs['bulge_dust'][1]
            self.bd_corr = corrs

            BD_g = self.bd_ratio(b_ug, d_ug-u0g_corr, b_n-n_corr, Re/Re_corr, h/h_corr)
            b_col, d_col = b_ug-b_ur, d_ug-d_ur
            kcorr_type = disk_subtype(BD_color=b_col-d_col, BD_ratio=BD_g)
            self.b_sub_type = kcorr_type
            if kcorr_type == 'none':
                labels.remove('bulge+disk')
                return labels
            
            kcorr_bulge, kcorr_disk = self.kcorr['Bulge'], self.kcorr[kcorr_type]
            b_kcol = b_col-np.subtract(*kcorr_bulge)
            d_kcol = d_col-np.subtract(*kcorr_disk)

            bulge_condition = (b_ell < 0.6) and (b_kcol > 0.6) & (b_kcol < 2.1) and (b_SE > 1.5) and (b_SE < 2.5)
            disk_condition = (d_ell < 0.6) and (d_kcol > 0) and (d_kcol < 0.8) and (d_SE > 1.5) and (d_SE < 2.5)
            condition = bulge_condition and disk_condition
            if not condition:
                labels.remove('bulge+disk')
            return labels
        
        def bulge_bar_disk_check(labels, bulge_pars, disk_pars, bar_pars):
            Re_bulge, bulge_ug, bulge_ur, bulge_n = bulge_pars
            h, d_ell, d_ug, d_ur, d_SE = disk_pars
            Re_bar, bar_ug, bar_ur, bar_n, bar_SE = bar_pars
            
            if self.decomp[2][0] == 0:
                labels.remove('bulge+bar+disk')
                return labels

            corrs = get_corrections(d_ell, bulge_n, trad=10, tau=0.5)
            h_corr = corrs['disk_proj'][0] * corrs['disk_dust'][0]
            u0g_corr = corrs['disk_proj'][1] + corrs['disk_dust'][1]
            u0r_corr = corrs['disk_proj'][1] + corrs['disk_dust'][2]
            Re_corr = corrs['bulge_proj'][0] * corrs['bulge_dust'][0]
            n_corr = corrs['bulge_proj'][1] + corrs['bulge_dust'][1]
            self.bbd_corr = corrs

            BD_g = self.bd_ratio(bulge_ug, d_ug-u0g_corr, bulge_n-n_corr, Re_bulge/Re_corr, h/h_corr)
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

            bulge_condition = (bulge_kcol > 0.6) and (bulge_kcol < 2.1) and (u0_bulge < u0_bar) and (Re_bulge < Re_bar)
            bar_condition = (bar_n < 0.75) and (bar_kcol > d_kcol-0.1) and (bar_kcol < bulge_kcol+0.5) and (bar_SE > 1) and (bar_SE < 6)
            disk_condition = (d_ell < 0.6) and (d_kcol > 0) and (d_kcol < 0.85) and (d_SE > 1.5) and (d_SE < 2.5)
            condition = bulge_condition and bar_condition and disk_condition
            if not condition:
                labels.remove('bulge+bar+disk')
            return labels


        galaxy_possible = elliptical_check(galaxy_possible, *df_g[['e_0', 'c_0', 'se_0']].values[0])
        print(galaxy_possible) if verbose else None
        galaxy_possible = disk_check(galaxy_possible, *df_g[['e_3', 'c_3', 'se_3']].values[0])
        print(galaxy_possible) if verbose else None
        galaxy_possible = bulge_disk_check(galaxy_possible, df_g[['a1_1', 'e1_1', 'ug1_1', 'ur1_1', 'n_1','se1_1']].values[0], df_g[['a2_1','e2_1','ug2_1','ur2_1','se2_1']].values[0])
        print(galaxy_possible) if verbose else None
        galaxy_possible = bulge_bar_disk_check(galaxy_possible, df_g[['a1_2','ug1_2','ur1_2','n1_2',]].values[0], df_g[['a2_2','e2_2','ug2_2','ur2_2','se2_2']].values[0], df_g[['a3_2','ug3_2','ur3_2','n3_2','se3_2']].values[0])
        print(galaxy_possible) if verbose else None
        galaxy_rss_cut = rss_check(galaxy_possible.copy())
        print(galaxy_rss_cut) if verbose else None
        
        if len(galaxy_possible) == 0:
            return 'unclear'
        elif len(galaxy_possible) > 0:
            if (len(galaxy_rss_cut) == 0) or (abs(df_g['n_iso_g'].values[0]-df_g['n_iso_r'].values[0]) > 10):
                return 'bad_fit'
            elif len(galaxy_rss_cut) == 1:
                return galaxy_rss_cut[0]
            else:
                if 'bulge' in galaxy_rss_cut:
                    return 'E-S0'
                elif 'disk' in galaxy_rss_cut:
                    return 'bulge+disk'
                else:
                    return 'bulge+bar+disk'


    def galaxy_pars(self):
        galaxy_type = self.classify_gal()
        gal_i = np.where(np.array(['Bulge', 'Bulge+Disk', 'Bulge+Bar+Disk', 'Disk', 'E-S0', 'unclear', 'bad_fit']) == galaxy_type)[0][0]

        z = self.gobj['g'].gal['z']
        wcs = self.gobj['g'].cutout['wcs']
        sn_loc = SkyCoord(*self.gobj['g'].gal['sn'], unit='deg')
        host_center = wcs.pixel_to_world(*self.center)

        arcsec2kpc = Planck18.arcsec_per_kpc_proper(z).value
        separation = host_center.separation(sn_loc).arcsec/arcsec2kpc
        sn_deg = np.array([sn_loc.ra.deg, sn_loc.dec.deg])
        host_deg = np.array([host_center.ra.deg, host_center.dec.deg])
        sn_vec = sn_deg - host_deg
        theta = np.pi - np.arctan(sn_vec[1]/sn_vec[0])

        if gal_i < 4:
            model_i = gal_i+1 if gal_i!=3 else 0
            sn_local_g = self.SB_profile(separation*arcsec2kpc, theta, band='g', model='combine', n_model=model_i, px=False, corr=True)
            sn_local_r = self.SB_profile(separation*arcsec2kpc, theta, band='r', model='combine', n_model=model_i, px=False, corr=True)
            try:
                R25_g = self.SB_isophote(25, 'g', n_model=model_i, model='combine', corr=True)[0][0]*0.262/arcsec2kpc
                R25_r = self.SB_isophote(25, 'r', n_model=model_i, model='combine', corr=True)[0][0]*0.262/arcsec2kpc
            except RuntimeError:
                R25_g, R25_r = 0, 0
        else:
            sn_local_g, sn_local_r, R25_g, R25_r = 0, 0, 0, 0

        fit_data, fit_errs = self.decomp[gal_i]
        if gal_i == 0 :
            ue_g, ue_r, n, Re_a, Re_b, Re_pa, Re_n = fit_data
            Re_a, Re_b = max(abs(Re_a), abs(Re_b)), min(abs(Re_a), abs(Re_b))
            gal_pa, gal_ecc, gal_se = Re_pa % (np.pi), 1-Re_b/Re_a, Re_n
            sn_component = 'elliptical'
            sersic_n = n
            bd_g, bb_g = 0, 0
            
        elif gal_i == 1:
            ue_g, ue_r, u0_g, u0_r, n, Re_a, Re_b, Re_pa, Re_n, h_a, h_b, h_pa, h_n = fit_data
            Re_a, Re_b = max(abs(Re_a), abs(Re_b)), min(abs(Re_a), abs(Re_b))
            h_a, h_b = max(abs(h_a), abs(h_b)), min(abs(h_a), abs(h_b))
            gal_pa, gal_ecc, gal_se = h_pa % (np.pi), 1-h_b/h_a, h_n
            sn_bulge = self.SB_profile(separation*arcsec2kpc, theta, band='g', model='bulge', n_model=gal_i+1, px=False, corr=True)
            sn_disk = self.SB_profile(separation*arcsec2kpc, theta, band='g', model='disk', n_model=gal_i+1, px=False, corr=True)
            if (sn_bulge - sn_disk < 1):
                sn_component = 'bulge' 
            else:
                sn_component = 'disk' 
            sersic_n = n
            bd_g, bb_g = self.bd_ratio(ue_g, u0_g, n, Re_a, h_a), 0

        elif gal_i == 2:
            ue_bulge_g, ue_bulge_r, ue_bar_g, ue_bar_r, u0_g, u0_r, n_bulge, n_bar, Re_bulge, Re_bar_a, Re_bar_b, Re_bar_pa, Re_bar_n, h_a, h_b, h_pa, h_n = fit_data
            Re_bar_a, Re_bar_b = max(abs(Re_bar_a), abs(Re_bar_b)), min(abs(Re_bar_a), abs(Re_bar_b))
            h_a, h_b = max(abs(h_a), abs(h_b)), min(abs(h_a), abs(h_b))
            gal_pa, gal_ecc, gal_se = h_pa % (np.pi), 1-h_b/h_a, h_n
            sersic_n = n_bulge

            sn_bulge = self.SB_profile(separation*arcsec2kpc, theta, band='g', model='bulge', n_model=gal_i+1, px=False, corr=True)
            sn_bar = self.SB_profile(separation*arcsec2kpc, theta, band='g', model='bar', n_model=gal_i+1, px=False, corr=True)
            sn_disk = self.SB_profile(separation*arcsec2kpc, theta, band='g', model='disk', n_model=gal_i+1, px=False, corr=True)
            if (sn_bar < sn_bulge) & (sn_bar < sn_disk):
                sn_component = 'bar'
            elif (sn_bulge - sn_disk < 1):
                sn_component = 'bulge' 
            else:
                sn_component = 'disk'

            bd_g = self.bd_ratio(ue_bulge_g, u0_g, n_bulge, Re_bulge, h_a)
            bb_g = self.bb_ratio(ue_bulge_g, ue_bar_g, n_bulge, n_bar, Re_bulge, Re_bar_a * np.sqrt(Re_bar_b/Re_bar_a))

        if gal_i == 3:
            ue_g, ue_r, h_a, h_b, h_pa, h_n = fit_data
            h_a, h_b = max(abs(h_a), abs(h_b)), min(abs(h_a), abs(h_b))
            gal_pa, gal_ecc, gal_se = h_pa % (np.pi), 1-h_b/h_a, h_n
            sn_component = 'disk'
            bd_g, bb_g = 0, 0
        
        rotation_matrix = np.array([[np.cos(gal_pa), np.sin(gal_pa)], [-np.sin(gal_pa), np.cos(gal_pa)]])
        sn_radial, sn_height = (rotation_matrix @ (sn_deg - host_deg)) * 3600/arcsec2kpc
        
        x1_salt, c_salt, sntype, abs_g_salt, abs_r_salt, lc_flag = sample.data.loc[self.name][['x1', 'c', 'classification', 'peak_mag_ztfg', 'peak_mag_ztfr', 'lccoverage_flag']]
        
        if np.any(df_gp[df_gp['ztfname'] == self.name]):
            gp_pars = df_gp[df_gp['ztfname'] == self.name][['abs_mag_peak_g', 'abs_mag_peak_r', 'dm_p15_g', 'dm_p15_r', 'dm_m10_g', 'dm_m10_r', 
                                                        'gr_peak_color', 'gr_p15_color', 'tr_tg']].values[0]
            gp_errs = df_gp[df_gp['ztfname'] == self.name][['abs_mag_peak_err_g', 'abs_mag_peak_err_r', 'dm_p15_err_g', 'dm_p15_err_r', 'dm_m10_err_g', 'dm_m10_err_r', 
                                                            'gr_peak_color_err', 'gr_p15_color_err']].values[0]
        else:
            gp_pars, gp_errs = np.zeros(9), np.zeros(8)

        self.galaxy = [self.name, z, *sn_deg, *host_deg, galaxy_type, gal_pa, gal_ecc, gal_se, separation, sersic_n, sn_radial, sn_height, sn_component, sn_local_g, sn_local_r, 
                       R25_g, R25_r,  bd_ratio, x1_salt, c_salt, sntype, abs_g_salt, abs_r_salt, lc_flag, *gp_pars, *gp_errs]


    def plot_func(self):
        xm, ym, rm, thetam = self.get_meshgrid()
        std_g = self.gobj['g'].brick['psfsize']/(2*np.sqrt(2*np.log(2)))
        std_r = self.gobj['r'].brick['psfsize']/(2*np.sqrt(2*np.log(2)))
        kernal_g = Gaussian2DKernel(x_stddev=std_g, y_stddev=std_g)
        kernal_r = Gaussian2DKernel(x_stddev=std_r, y_stddev=std_r)

        def psf_convolve(model):
            model_grid_g = model(rm*0.262, thetam, 'g')
            model_grid_r = model(rm*0.262, thetam, 'r')
            model_psf_g = -2.5*np.log10(convolve_fft(10**(-0.4*model_grid_g), kernal_g))
            model_psf_r = -2.5*np.log10(convolve_fft(10**(-0.4*model_grid_r), kernal_r))
            interp_g = RegularGridInterpolator((xm[0]*0.262, ym.T[0]*0.262), model_psf_g.T)
            interp_r = RegularGridInterpolator((xm[0]*0.262, ym.T[0]*0.262), model_psf_r.T)
            return {'g': interp_g, 'r': interp_r}
        
        def model_psf(r, theta, band, psf_interp):
            x, y = r*np.cos(theta)+self.center[0]*0.262, r*np.sin(theta)+self.center[1]*0.262
            return psf_interp[band]((x, y))
        
        def disk_2D_model(x_data, pars, theta, band):
            rp = 2
            u0_g, u0_r = pars[:rp]
            u0 = {'g': u0_g, 'r': u0_r}
            def model(x, theta, band):
                h = self.super_ellipse(theta, *pars[rp:rp+4])
                return self.disk(x, u0[band], h)
            
            psf_interp = psf_convolve(model)
            return model_psf(x_data, theta, band, psf_interp)

        def bulge_2D_model(x_data, pars, theta, band):
            rp = 3
            ue_g, ue_r, n = pars[:rp]
            ue = {'g': ue_g, 'r': ue_r}
            def model(x, theta, band):
                Re = self.super_ellipse(theta, *pars[rp:rp+4])
                return self.bulge(x, ue[band], Re, n)
            
            psf_interp = psf_convolve(model)
            return model_psf(x_data, theta, band, psf_interp)

        def bulge_disk_2D_model(x_data, pars, theta, band):
            rp = 5
            ue_g, ue_r, u0_g, u0_r, n = pars[:rp]
            ue = {'g': ue_g, 'r': ue_r}
            u0 = {'g': u0_g, 'r': u0_r}
            def model(x, theta, band):
                Re = self.super_ellipse(theta, *pars[rp:rp+4])
                h = self.super_ellipse(theta, *pars[rp+4:rp+8])
                return self.combine(x, ue[band], u0[band], Re, h, n)
            
            psf_interp = psf_convolve(model)
            return model_psf(x_data, theta, band, psf_interp)
        
        def bulge_bar_disk_2D_model(x_data, pars, theta, band):
            rp = 9
            ue_bulge_g, ue_bulge_r, ue_bar_g, ue_bar_r, u0_g, u0_r, n_bulge, n_bar, Re_bulge = pars[:rp]
            ue_bulge = {'g': ue_bulge_g, 'r': ue_bulge_r}
            ue_bar = {'g': ue_bar_g, 'r': ue_bar_r}
            u0 = {'g': u0_g, 'r': u0_r}
            def model(x, theta, band):
                Re_bar = self.super_ellipse(theta, *pars[rp:rp+4])
                h = self.super_ellipse(theta, *pars[rp+4:rp+8])
                return self.combine_3(x, ue_bulge[band], ue_bar[band], u0[band], Re_bulge, Re_bar, h, n_bulge, n_bar)
            
            psf_interp = psf_convolve(model)
            return model_psf(x_data, theta, band, psf_interp)
        return  disk_2D_model, bulge_2D_model, bulge_disk_2D_model, bulge_bar_disk_2D_model

        
    def plot_spokes(self, spokes=12, sigma=3, n_model=2):
        rp = [3, 5, 9, 2][n_model-1]
        disk_2D_model, bulge_2D_model, bulge_disk_2D_model, bulge_bar_disk_2D_model = self.plot_func()  
        fig, axis = plt.subplots(figsize=(18, 10), ncols=2, nrows=3, dpi=100)

        z = self.gobj['g'].gal['z']
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
                ax.plot(x_ax, self.bulge(x_ax, ue_bulge, Re_bulge, n_bulge), 'g--', label='bulge')
                ax.plot(x_ax, self.bulge(x_ax, ue_bar, Re_bar, n_bar), 'g--', label='bar', color='lime')
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
                ax.plot(x_ax, self.bulge(x_ax, ue_bulge, Re_bulge, n_sersic), 'g--', label='bulge')
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
        rp = [3, 5, 9, 2][n_model-1]
        arcsec2px = np.array([0.262, 0.262, 1, 1]) if px else np.ones(4)
        if n_model == 3:
            ue_bulge_g, ue_bulge_r, ue_bar_g, ue_bar_r, u0_g, u0_r, n_bulge, n_bar, Re_bulge = self.decomp[n_model-1][0][:rp]
            Re_bar = self.super_ellipse(theta, *self.decomp[n_model-1][0][rp:rp+4]/arcsec2px)
            h_disk = self.super_ellipse(theta, *self.decomp[n_model-1][0][rp+4:rp+8]/arcsec2px)

            if corr:
                kcorr_bulge_bar, kcorr_disk = self.kcorr['Bulge'], self.kcorr[self.bb_sub_type]
                corrs = self.bbd_corr
                h_corr = corrs['disk_proj'][0] * corrs['disk_dust'][0]
                u0g_corr = corrs['disk_proj'][1] + corrs['disk_dust'][1]
                u0r_corr = corrs['disk_proj'][1] + corrs['disk_dust'][2]
                Re_corr = corrs['bulge_proj'][0] * corrs['bulge_dust'][0]
                n_corr = corrs['bulge_proj'][1] + corrs['bulge_dust'][1]

                ue_bulge_g -= kcorr_bulge_bar[0]
                ue_bulge_r -= kcorr_bulge_bar[1]
                ue_bar_g -= kcorr_bulge_bar[0]
                ue_bar_r -= kcorr_bulge_bar[1]
                u0_g -= (kcorr_disk[0]+u0g_corr)
                u0_r -= (kcorr_disk[1]+u0r_corr)
                h_disk /= h_corr
                Re_bulge /= Re_corr
                Re_bar /= Re_corr
                n_bulge -= n_corr

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
                corrs = self.bd_corr
                h_corr = corrs['disk_proj'][0] * corrs['disk_dust'][0]
                u0g_corr = corrs['disk_proj'][1] + corrs['disk_dust'][1]
                u0r_corr = corrs['disk_proj'][1] + corrs['disk_dust'][2]
                Re_corr = corrs['bulge_proj'][0] * corrs['bulge_dust'][0]
                n_corr = corrs['bulge_proj'][1] + corrs['bulge_dust'][1]

                ue_g -= kcorr_bulge[0]
                ue_r -= kcorr_bulge[1]
                u0_g -= (kcorr_disk[0]+u0g_corr)
                u0_r -= (kcorr_disk[1]+u0r_corr)
                h_disk /= h_corr
                Re_bulge /= Re_corr
                n_sersic -= n_corr
            
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
                ue_g -= kcorr[0]
                ue_r -= kcorr[1]

            ue_bulge = {'g': ue_g, 'r': ue_r}[band]
            Re_bulge = self.super_ellipse(theta, *self.decomp[n_model-1][0][rp:rp+4]/arcsec2px)
            return self.bulge(r, ue_bulge, Re_bulge, n_sersic)
        
        elif n_model == 0:
            u0_g, u0_r = self.decomp[n_model-1][0][:rp]
            h_disk = self.super_ellipse(theta, *self.decomp[n_model-1][0][rp:rp+4]/arcsec2px)
            if corr:
                kcorr = self.kcorr['Sab']
                corrs = self.d_corr
                h_corr = corrs['disk_proj'][0] * corrs['disk_dust'][0]
                u0g_corr = corrs['disk_proj'][1] + corrs['disk_dust'][1]
                u0r_corr = corrs['disk_proj'][1] + corrs['disk_dust'][2]

                u0_g -= (kcorr[0]+u0g_corr)
                u0_r -= (kcorr[1]+u0r_corr)
                h_disk /= h_corr
 
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
        return [[max(a, b), min(a, b), pa, n], np.sqrt(np.diag(out_pars[1])), np.round(check, 3)] 

    def plot_SB_profile(self, band, isophote=False, subtract=False, n_model=2, corr=False):
        xm, ym, rm, thetam = self.get_meshgrid()

        bulge_arr = self.SB_profile(rm, thetam, model='bulge', n_model=n_model, band=band,  corr=corr)
        bar_arr = self.SB_profile(rm, thetam, model='bar', n_model=n_model, band=band, corr=corr)
        disk_arr = self.SB_profile(rm, thetam, model='disk', n_model=n_model, band=band, corr=corr)
        both_arr = self.SB_profile(rm, thetam, model='all', n_model=n_model, band=band, corr=corr)

        std = self.gobj[band].brick['psfsize']/(2*np.sqrt(2*np.log(2)))
        kernal = Gaussian2DKernel(x_stddev=std, y_stddev=std)
        both_arr = -2.5*np.log10(convolve_fft(10**(-0.4*both_arr), kernal))

        reds, greens, blues = self.alpha_colormap()

        if not subtract:
            fig, ax = self.gobj[band].plot()
            rp = [3, 5, 9, 0][n_model-1]
            arcsec2px = np.array([0.262, 0.262, 1, 1])
            if n_model == 3:
                Re_bulge = self.decomp[n_model-1][0][8]/0.262
                self.patch_super_ellipse([Re_bulge, Re_bulge, 0, 2], self.center, ax, 'red', label='Bulge $R_e$')
                self.patch_super_ellipse(self.decomp[n_model-1][0][rp:rp+4]/arcsec2px, self.center, ax, 'green', label='Bar $R_e$')

                disk_a, disk_b, disk_pa, disk_n = self.decomp[n_model-1][0][rp+4:rp+8]
                u0_g, u0_r = self.decomp[n_model-1][0][4:6]
                u0_disk = {'g': u0_g, 'r': u0_r}[band]
                ue_disk, (disk_a, disk_b) = self.transform(u0_disk, [disk_a, disk_b], n=1)
                self.patch_super_ellipse(np.array([disk_a, disk_b, disk_pa, disk_n])/arcsec2px, self.center, ax, 'blue', label='Disk $R_e$')
                ax.imshow(disk_arr, cmap=blues, vmax=26)
                ax.imshow(bar_arr, cmap=greens, vmax=26)
                ax.imshow(bulge_arr, cmap=reds, vmax=26)

            elif n_model == 2:
                self.patch_super_ellipse(self.decomp[n_model-1][0][rp:rp+4]/arcsec2px, self.center, ax, 'red', label='Bulge $R_e$')
                u0_g, u0_r = self.decomp[n_model-1][0][2:rp-1]
                u0_disk = {'g': u0_g, 'r': u0_r}[band]
                disk_a, disk_b, disk_pa, disk_n = self.decomp[n_model-1][0][rp+4:rp+8]
                ue_disk, (disk_a, disk_b) = self.transform(u0_disk, [disk_a, disk_b], n=1)
                self.patch_super_ellipse(np.array([disk_a, disk_b, disk_pa, disk_n])/arcsec2px, self.center, ax, 'blue', label='Disk $R_e$')
                ax.imshow(disk_arr, cmap=blues, vmax=26)
                ax.imshow(bulge_arr, cmap=reds, vmax=26)

            elif n_model == 1:
                self.patch_super_ellipse(self.decomp[n_model-1][0][rp:rp+4]/arcsec2px, self.center, ax, 'red', label='Bulge $R_e$')
                ax.imshow(bulge_arr, cmap=reds, vmax=26)
            
            elif n_model == 0:
                disk_a, disk_b, disk_pa, disk_n = self.decomp[n_model-1][0][rp:rp+4]
                u0_g, u0_r = self.decomp[n_model-1][0][:rp]
                u0_disk = {'g': u0_g, 'r': u0_r}[band]
                ue_disk, (disk_a, disk_b) = self.transform(u0_disk, [disk_a, disk_b], n=1)
                self.patch_super_ellipse(np.array([disk_a, disk_b, disk_pa, disk_n])/arcsec2px, self.center, ax, 'blue', label='Disk $R_e$')
                ax.imshow(disk_arr, cmap=blues, vmax=26)
            
            if isophote:
                wi = 0.1
                mag_i = np.where((self.image[band].T > isophote-wi) & (self.image[band].T < isophote+wi))
                ax.plot(mag_i[0], mag_i[1], 'g.', c='green', ms=2, zorder=5, label=f'{isophote} mag/arcsec$^2$')

                pars, errs, check = self.SB_isophote(isophote, band, n_model)
                self.patch_super_ellipse(pars, self.center, ax, 'lime') if check == 0 else None

            ax.legend(framealpha=1, fontsize=7, loc=1)

        if subtract:
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
            # disk_arr[int(sn_px[1])-10:int(sn_px[1])+10,int(sn_px[0])-10:int(sn_px[0])+10] = 25
            ax2.imshow(disk_arr, cmap=blues, origin='lower', vmax=25) if (n_model in [0, 2, 3]) else None
            ax2.imshow(bar_arr, cmap=greens, origin='lower', vmax=25) if n_model == 3 else None
            ax2.imshow(bulge_arr, cmap=reds, origin='lower', vmax=25) if n_model != 0 else None
            
            ax2.scatter(sn_px[0], sn_px[1], c='cyan', s=40, lw=2, fc='None', ec='cyan', zorder=10)

            ax3.imshow(sub_mag(sky_mag, both_arr), origin='lower', cmap='gray', vmax=30, vmin=mag_c)
            plt.tight_layout()


def get_corrections(x, n, tau=0.5, trad=3):
    # x = 1 - cos(i) = 1-b/a 
    def BV_g(B, V):
        return V + 0.6 * (B-V)#  - 0.12

    def BV_r(B, V):
        return V - 0.42 * (B-V) # + 0.11
    
    B_disk_proj = disk_proj[disk_proj['Band'] == 'V']
    disk_proj_h_pars = B_disk_proj['Ri_R0'].values
    disk_proj_u0_pars = B_disk_proj['delta_SB0_'].values
    disk_proj_h = np.poly1d(disk_proj_h_pars[::-1])(x)
    disk_proj_u0 = np.poly1d(disk_proj_u0_pars[::-1])(x)

    sersic_proj_range = bulge_proj[bulge_proj['trad'] == trad]['n0'].values
    bulge_proj_Re_pars =  bulge_proj[bulge_proj['trad'] == trad]['Ri_R0'].values
    bulge_proj_n_pars =  bulge_proj[bulge_proj['trad'] == trad]['ni'].values
    bulge_proj_Re_interp = np.polyfit(sersic_proj_range, bulge_proj_Re_pars, deg=2)
    bulge_proj_n_interp = np.polyfit(bulge_proj_n_pars, sersic_proj_range, deg=2)
    bulge_proj_Re = np.poly1d(bulge_proj_Re_interp)(n)
    bulge_proj_n = n - np.poly1d(bulge_proj_n_interp)(n)

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

    V_bulge_dust = bulge_dust[(bulge_dust['Band'] == 'V') & (bulge_dust['tau'] == tau)]
    bulge_dust_Re_pars = V_bulge_dust['Rapp_Ri'].values
    bulge_dust_n_pars = V_bulge_dust['napp'].values
    bulge_dust_Re = np.poly1d(bulge_dust_Re_pars[::-1])(x)
    bulge_dust_n = np.poly1d(bulge_dust_n_pars[::-1])(x) - 1
    
    return {'disk_proj': [disk_proj_h, disk_proj_u0],
            'disk_dust': [disk_dust_h, disk_dust_u0_g, disk_dust_u0_r],
            'bulge_proj': [bulge_proj_Re, bulge_proj_n],
            'bulge_dust': [bulge_dust_Re, bulge_dust_n]}


def K_corr(z, template):
    filter_g, filter_r = fset['gDEC'], fset['rDEC']
    filter_g.zp, filter_r.zp  = 22.5, 22.5

    k_g, _ = kcorr.K(template.wl.values, template.fl.values, filter_g, filter_g, z=z)
    k_r, _ = kcorr.K(template.wl.values, template.fl.values, filter_r, filter_r, z=z)
    return [k_g, k_r]