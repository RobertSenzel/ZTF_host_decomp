import bs4
import lxml
import ztfidr
import requests
import numpy as np
import pandas as pd
from io import BytesIO
from extinction import fm07
from astropy.io import fits
from astropy.wcs import WCS
from astropy import units as u
from astropy.table import Table
import matplotlib.pyplot as plt
from scipy.integrate import quad
from skimage import measure, draw
from scipy.signal import convolve2d
from astroquery.ipac.ned import Ned
from astroquery.vizier import Vizier
from astropy.cosmology import Planck18
from matplotlib.patches import Ellipse
from scipy.interpolate import interp1d
from scipy.odr import Model, Data, ODR
from skimage.measure import EllipseModel
from astropy.coordinates import SkyCoord
from scipy.ndimage import gaussian_filter
from scipy.special import gamma, gammainc
from scipy.optimize import curve_fit, fsolve, minimize
from matplotlib.colors import ListedColormap
from skimage.measure import label as ConnectRegion
from astroquery.exceptions import RemoteServiceError
from astropy.convolution import convolve_fft, Gaussian2DKernel

sample = ztfidr.get_sample()
host_data = ztfidr.io.get_host_data()
Vizier.ROW_LIMIT = 1000
qs = Vizier.get_catalogs('J/ApJS/196/11/table2')
df_update = pd.read_csv('csv_files/ztfdr2_matched_hosts.csv', index_col=0)
df10 = Table.read('fits_files/survey-bricks-dr10-south.fits.gz', format='fits')
df9 = Table.read('fits_files/survey-bricks-dr9-north.fits.gz', format='fits')


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
        
        if catalog == 'sdss':
            self.gal = {'host': [qs[0][host_name]['_RA'], qs[0][host_name]['_DE']], 'z': qs[0][host_name]['z'], 
                        'sn': [qs[0][host_name]['_RA'], qs[0][host_name]['_DE']], 'A': 0, 'z_source': 'None'}
            print(self.gal['z']) if self.verbose else None


    def init_dr2(self, sn_name):
        self.host_name = 'DR2'
        self.sn_name = sn_name

        dz = (sample.data['redshift'] - df_update['z']).dropna()
        delta_z = dz[dz!=0]
        ztf_z = sample.data['redshift']
        ztf_z.loc[delta_z.index] = df_update['z'].loc[delta_z.index]
        z = ztf_z.loc[sn_name]

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
        

    def get_coords(self, host_name, sn_name):
        pass
        # def NED(host_query):
        #     ra, dec =  host_query['RA'][0], host_query['DEC'][0]
        #     print(host_name, sn_name, ra, dec, host_query['Redshift'][0]) if self.verbose else None
        #     redshift = host_query['Redshift']
        #     redshift = [-1] if (len(redshift) == 0) or (type(redshift[0]) == np.ma.core.MaskedConstant) else redshift
        #     return {'host': [ra, dec], 'z': float(redshift[0]), 'sn': [sn_query['RA'][0], sn_query['DEC'][0]]}

        # def Hyper_Leda():
        #     host_query = Vizier.query_object(host_name, catalog=['HyperLeda'])[0][0]
        #     loc = SkyCoord(host_query['RAJ2000'], host_query['DEJ2000'], unit=(u.hourangle, u.deg))
        #     ra, dec = loc.ra.deg, loc.dec.deg
        #     print(host_name, sn_name, ra, dec, 'HL') if self.verbose else None
        #     return {'host': [ra, dec], 'z': -1, 'sn': [sn_query['RA'][0], sn_query['DEC'][0]]}

        # sn_query = Ned.query_object(sn_name)
        # if len(sn_query['RA']) == 0:
        #     raise TypeError
        # try:
        #     host_query = Ned.query_object(host_name)
        # except RemoteServiceError:
        #     return Hyper_Leda()
        # if (len(host_query['RA']) == 0) or (len(host_query['DEC']) == 0):
        #    return Hyper_Leda()
        # else:
        #    return NED(host_query)
   

    def get_cutout(self, survey, output_size, band, scale=0.262):
        ra, dec = self.gal['host']

        def legacy_url(survey):
            layer = 'ls-dr10' if survey == 'legacy' else 'sdss'
            service = 'https://www.legacysurvey.org/viewer/'
            return f'{service}fits-cutout?ra={ra}&dec={dec}&layer={layer}&pixscale={scale}&bands={band}&size={output_size}&subimage'

        def ps1_url():
            size = round(output_size*scale/0.25)
            service = 'https://ps1images.stsci.edu/cgi-bin/'
            file_url = f'{service}ps1filenames.py?ra={ra}&dec={dec}&filters={band}'
            table = Table.read(file_url, format='ascii')
            if len(table['filename']) == 0:
                return 'no file'
            else:
                return f"{service}fitscut.cgi?ra={ra}&dec={dec}&size={size}&format=fits&output_size={output_size}&red={table['filename'][0]}"
        
        if survey == 'auto':
            url = legacy_url('legacy')
            res = requests.get(url, timeout=10)
            self.survey = 'legacy'
            if len(res.content) > 1000:
                return fits.open(BytesIO(res.content))
            else:
                url = ps1_url()
                self.survey = 'ps1'
                if url == 'no file':
                    url = legacy_url('sdss')
                    self.survey = 'sdss'
                res = requests.get(url, timeout=10)
                if len(res.content) > 1000:
                    return fits.open(BytesIO(res.content))
                else:
                    raise TypeError
        elif (survey == 'ps1'):
            url = ps1_url()
            res = requests.get(url)
            self.survey = 'ps1'
            return fits.open(BytesIO(res.content))
        else:
            url = legacy_url(survey)
            res = requests.get(url)
            self.survey = survey
            return fits.open(BytesIO(res.content))
        

    def hyper_leda(self):
        pass
        # ra, dec = self.gal['host']
        # co = SkyCoord(ra=ra, dec=dec, unit=(u.deg, u.deg))
        # result_table = Vizier.query_region(co, radius=0.1 * u.arcmin, catalog='HyperLeda')
        # if (len(result_table) == 0):
        #     self.gal['type'] = 'none'
        #     self.gal['t'] = [-9, 20]
        #     self.gal['incl'] = -1
        #     self.gal['source'] = 'Hyperleda'
        # else:
        #     gal_name = result_table[0]['PGC'][0]
        #     url = f'https://leda.univ-lyon1.fr/ledacat.cgi?PGC{gal_name}&ob=ra'
        #     out = requests.get(url)
        #     soup = bs4.BeautifulSoup(out.content, 'lxml')
        #     table_source = soup.find_all('table')[5]
        #     table = pd.DataFrame(columns=['Parameter', 'Value', 'Unit', 'Description'])
        #     for j in table_source.find_all('tr')[1:]:
        #         row_data = j.find_all('td')
        #         row = [i.text for i in row_data]
        #         table.loc[len(table)] = row

        #     dict_ = dict(zip(table['Parameter'].values, table['Value'].values))
        #     tcode = dict_.get('t', '-9 \pm 20').split()
        #     t = float(tcode[0])
        #     t_err = float(tcode[-1]) if len(tcode) == 3 else 20

        #     self.gal['type'] = dict_.get('type', 'none').strip()
        #     self.gal['source'] = 'hyperleda'
        #     self.gal['t'] = [t, t_err]
        #     self.gal['incl'] = float(dict_.get('incl', '-1').strip())


    def get_galaxy_params(self, source='hl', args=[0, 0, 6]):
        pass
        # def manual(t, gal_type, source):
        #     a, b, t = self.iso.get('a', [args[0]])[0], self.iso.get('b', [args[1]])[0], args[2]
        #     self.gal['type'] = gal_type
        #     self.gal['t'] = [t, 20]
        #     self.gal['incl'] = self.calc_incl(t, a/b)
        #     self.gal['source'] = source

        # if source == 'hl':
        #     self.hyper_leda()
        # elif source == 'calc':
        #     manual(t=args[2], gal_type='none', source='calculated')
        # elif source == 'auto':
        #     self.hyper_leda()
        #     if self.gal['incl'] != -1:
        #         pass
        #     elif (self.gal['incl'] == -1) & (self.gal['t'][0] != -9):
        #         manual(t=self.gal['t'][0], gal_type=self.gal['type'], source='hl_calculated')
        #     else:
        #         manual(t=args[2], gal_type='none', source='calculated')

        # print(self.gal['type'], self.gal['t'], self.gal['incl'], self.gal['source'] ) if self.verbose else None
    
    
    def flux2mag(self, flux, survey, band, scale, soft, exptime=1):
        # ztf_filters = {'g': 4746.48, 'r': 6366.38, 'i': 7829.03}
        ps1_filters = {'g': 4810.16 , 'r': 6155.47, 'i': 7503.03, 'z': 8668.36, 'y': 9613.60}
        sdss_filters = {'g': 4671.78, 'r': 6141.12, 'i': 7457.89, 'z': 8922.78, 'u': 3608.04}
        leg_filters = {'g': 4769.90, 'r': 6370.44 ,'i': 7774.30, 'z': 9154.88, 'y': 9886.45, 'u': 3856.88} # DECam
        eff_wl = {'legacy': leg_filters[band], 'ps1': ps1_filters[band], 'sdss': sdss_filters[band]}

        A_ = fm07(np.array([eff_wl[survey]]), self.gal['A'])

        def mag_soft(flux, zp):
            b = {'g': 0.9e-10, 'r': 1.2e-10, 'i': 1.8e-10, 'z': 7.4e-10}
            return zp + -2.5/np.log(10) * (np.arcsinh(flux/(2*b[band]*scale**2)) + np.log(b[band])) - A_
        
        def mag_(flux, zp):
            return zp - 2.5*np.log10(flux/scale**2) - A_
        
        zp_ = {'legacy': 22.5, 'sdss': 22.5, 'ps1': 25 + 2.5*np.log10(exptime)}
        self.brick['sky_mag'] = float(mag_soft(self.brick['sky'], zp_[survey]))

        flux = flux - self.brick['sky']
        return mag_soft(flux, zp_[survey]) if soft else mag_(flux, zp_[survey])

    
    @staticmethod
    def calc_aperature(size, redshift, scale):
        if size == 'z':
            if np.sign(redshift) == -1:
                return 800
            else:
                aperature = 0.07 # estimate_radius()
                dist = Planck18.luminosity_distance(redshift)
                return min(int(np.rad2deg(aperature/dist.value)*(3600/scale)),  800)
        else:
            return size


    def get_image(self, source, survey, output_, band, scale, soft=True):
        folder_name = 'fits' if survey == 'auto' else survey
        path = f'dr2_{folder_name}_{band}/{self.sn_name}.fits'
        if source == 'query':
            output_size = self.calc_aperature(output_, self.gal['z'], scale)
            fits_ = self.get_cutout(survey, output_size, band, scale)
            fits_data = fits_
        elif source == 'save':
            fits_ = fits.open(path)
            fits_data = fits_
        elif source == 'query_save':
            output_size = self.calc_aperature(output_, self.gal['z'], scale)
            fits_ = self.get_cutout(survey, output_size, band, scale)
            fits_data = fits_
            fits_data.writeto(path, overwrite=True)
            fits_.close()
            return
        else:
            print('invalid source')
            return

        
        flux, invvar, wcs = fits_data[1].data, fits_data[2].data, WCS(fits_data[1].header)
        brick_name = fits_data[1].header['brick']
        brick_data = {'dr10': df10[df10['brickname'] == brick_name], 'dr9': df9[df9['brickname'] == brick_name]}
        brick_dr = 'dr10' if len(brick_data['dr10']) == 1 else 'dr9'
        df_brick = brick_data[brick_dr]
        
        self.brick = {'brickname': brick_name, 'psfsize': df_brick[f'psfsize_{band}'][0], 'psfdepth': df_brick[f'psfdepth_{band}'][0], 'galdepth': df_brick[f'galdepth_{band}'][0],
                      'sky': df_brick[f'cosky_{band}'][0]}

        exptime = fits_data.header['exptime'] if self.survey == 'ps1' else 1
        mag = self.flux2mag(flux, self.survey, band, scale, soft, exptime=exptime)
        mag_raw = self.flux2mag(flux, self.survey, band, scale, soft=False, exptime=exptime)
        self.cutout = {'flux': flux, 'mag': mag, 'mag_raw': mag_raw, 'invvar': invvar, 'wcs': wcs, 'scale': scale, 'band': band}
        
        print(brick_dr, *mag.shape) if self.verbose else None
        fits_.close()

        
    def plot(self):
        wcs, mag, scale = self.cutout['wcs'], self.cutout['mag'], self.cutout['scale']
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
        self.gobj = {'g': HostGal(verbose=verbose), 'r': HostGal(verbose=verbose)}
        if catalog == 'ztf':
            self.gobj['g'].init_dr2(target_name)
            self.gobj['r'].init_dr2(target_name)
        else:
            self.gobj['g'].init_query(target_name, catalog)
            self.gobj['r'].init_query(target_name, catalog)
        self.gobj['g'].get_image(source=source, survey='auto', output_=size, band='g', scale=0.262)
        self.gobj['r'].get_image(source=source, survey='auto', output_=size, band='r', scale=0.262)

        self.contours = {'g': {}, 'r': {}}
        self.image = {'g': self.gobj['g'].cutout['mag'], 'r': self.gobj['r'].cutout['mag']}
        self.invvar = {'g': self.gobj['g'].cutout['invvar'], 'r': self.gobj['r'].cutout['invvar']} 
        self.flux = {'g': self.gobj['g'].cutout['flux'], 'r': self.gobj['r'].cutout['flux']} 
        self.mask = mask
        self.center =  np.array([size/2, size/2])

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
                ax1.scatter(px_all.T[0], px_all.T[1], s=2, marker='o',zorder=0, color='red', label=f'{iso_i:.1f} removed')
           
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
        # if fix_center:
        #     xc, yc = self.center
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

        if np.any(mask):
            for (ra, dec, r) in mask:
                coords = SkyCoord(ra=ra*u.degree, dec=dec*u.degree, frame='icrs')
                m_px = self.gobj.cutout['wcs'].world_to_pixel(coords)
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
            for iso in np.arange(round(self.gobj[band].brick['psfdepth'], 1), 16, -step):
                iso = np.round(iso, 1)
                print(f'{band}-{iso}') if (iso%1==0.0 and self.verbose) else None
                self.contour_fit(iso, mask=[], band=band)
                if len(self.contours[band][iso][1][1]) == 0:
                    del self.contours[band][iso]
                    break
        



class BDdecomp:
    def __init__(self, host_name, gd):
        self.host_name = host_name
        self.contours = gd.contours
        self.image = gd.image
        self.center = gd.center
        self.gobj = gd.gobj
        self.mask = gd.mask
        mags_g, iso_data_g = self.extract_data('g')
        mags_r, iso_data_r = self.extract_data('r')
        self.mags = {'g': mags_g, 'r': mags_r}
        self.iso_data = {'g': iso_data_g, 'r': iso_data_r}

    def extract_data(self, band):
        key_targs = list(self.contours[band].keys())
        pars_ = np.array([self.contours[band][iso_key][0] for iso_key in key_targs if len(self.contours[band][iso_key][0]) > 0]).T
        
        offsets = np.sqrt((pars_[4]-self.center[0])**2 + (pars_[5]-self.center[1])**2)
        cuts = np.where((offsets < 5))

        mags = np.array(key_targs)[cuts]
        pars_ = pars_.T[cuts].T.reshape((12, -1))
        if len(pars_) == 0:
            raise ValueError
        return mags, pars_
    
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

        ax2.plot(pars_[0]*0.262, np.round(pars_[4])+0.02, 'b.', label='center x')
        ax2.plot(pars_[0]*0.262, np.round(pars_[5])-0.02, 'r.', label='center y')
        ax2.set_xlabel('R [arcsec]')
        ax2.legend()

        ax3.plot(pars_[0]*0.262, np.rad2deg(pars_[2]), 'b.')
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
    
    @staticmethod
    def psf(r):
        sigma = 1.5/(2*np.sqrt(2*np.log(2))) # self.gobj[band].brick['psfsize']
        return 1/(sigma*np.sqrt(2*np.pi))*np.exp(-(1/2)*(r/sigma)**2)

    @staticmethod
    @np.vectorize
    def convolve_psf1(r, model):
        def integrand(x):
            return model(x)*BDdecomp.psf(r-x)
        integral = quad(integrand, -np.inf, np.inf)
        if integral[0] < integral[1]:
            return model(r)
        else:
            return integral[0]
    
    @staticmethod
    def convolve_psf(r, model):
        r_test = np.linspace(-50, 50, 1000)
        psf_conv = convolve_fft(model(r_test), BDdecomp.psf(r_test))
        return interp1d(r_test, psf_conv)(r)
    
    @staticmethod
    def one_component(r, ue, Re, n):
        model_flux = lambda x: 10**(-0.4*BDdecomp.bulge(x, 0, Re, n))
        return -2.5*np.log10(10**(-0.4*ue)*BDdecomp.convolve_psf(r, model_flux))
    
    @staticmethod
    def one_component_disk(r, u0, h):
        model_flux = lambda x: 10**(-0.4*BDdecomp.disk(x, 0, h))
        return -2.5*np.log10(10**(-0.4*u0)*BDdecomp.convolve_psf(r, model_flux))
    
    @staticmethod
    def two_component(r, ue, u0, Re, h, n):
        model_flux_1 = lambda x: 10**(-0.4*BDdecomp.bulge(x, 0, Re, n))
        model_flux_2 = lambda x: 10**(-0.4*BDdecomp.disk(x, 0, h))
        return -2.5*np.log10(10**(-0.4*ue)*BDdecomp.convolve_psf(r, model_flux_1) + 10**(-0.4*u0)*BDdecomp.convolve_psf(r, model_flux_2))
    
    @staticmethod
    def three_component(r, ue_bulge, ue_bar, u0, Re_bulge, Re_bar, h, n_bulge, n_bar):
        model_flux_1 = lambda x: 10**(-0.4*BDdecomp.bulge(x, 0, Re_bulge, n_bulge))
        model_flux_2 = lambda x: 10**(-0.4*BDdecomp.disk(x, 0, h))
        model_flux_3 = lambda x: 10**(-0.4*BDdecomp.bulge(x, 0, Re_bar, n_bar))
        return -2.5*np.log10(10**(-0.4*ue_bulge)*BDdecomp.convolve_psf(r, model_flux_1) + 10**(-0.4*u0)*BDdecomp.convolve_psf(r, model_flux_2) + 10**(-0.4*ue_bar)*BDdecomp.convolve_psf(r, model_flux_3))

    
    @staticmethod
    def BIC(out, x_data, y_data, model):
        n = len(x_data)
        k = len(out)
        RSS = np.sum((model(out, x_data) - y_data)**2)
        return n * np.log(RSS/n) + k * np.log(n), RSS, k
    
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
        return target_ang * 0.262

    def plot_gal_iso(self, spokes):
        fig, axis = plt.subplots(figsize=(12, 6), dpi=100, ncols=2)
        theta_arr =  {'g': np.linspace(0, 360-360/spokes, spokes), 'r': np.linspace(180/spokes, 360-180/spokes, spokes)}
        for band, ax in zip(['g', 'r'], axis):
            ax.imshow(self.image[band], cmap='gray', origin='lower')
        
            a_lim = self.iso_data[band][0].max()
            min_i = np.argmin(self.iso_data[band][0])
            xc_ref, yc_ref = self.iso_data[band][4][min_i], self.iso_data[band][5][min_i]
            ax.set_xlim([xc_ref-a_lim-10, xc_ref+a_lim+10])
            ax.set_ylim([yc_ref-a_lim-10, yc_ref+a_lim+10])

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
    
    def main_BD(self, spokes=12, mode=2, verbose=True, bar=False):

        def bulge_2D_model(pars, all_data):
            rp = 3
            ue_g, ue_r, n = pars[:rp]
            ue = {'g': ue_g, 'r': ue_r}
            data = {'g': all_data[:len(self.iso_data['g'].T)*spokes], 'r': all_data[len(self.iso_data['g'].T)*spokes:]}
            def decompostion(x, theta, band):
                Re = self.super_ellipse(np.deg2rad(theta), *pars[rp:rp+4])
                return self.one_component(x, ue[band], Re, n)
            data_g = np.concatenate([decompostion(data['g'].reshape(spokes, -1)[i], angle_targets['g'][i], 'g') for i in range(spokes)])
            data_r = np.concatenate([decompostion(data['r'].reshape(spokes, -1)[i], angle_targets['r'][i], 'r') for i in range(spokes)])
            return np.concatenate([data_g, data_r])

        def bulge_disk_2D_model(pars, all_data):
            rp = 5
            ue_g, ue_r, u0_g, u0_r, n = pars[:rp]
            ue = {'g': ue_g, 'r': ue_r}
            u0 = {'g': u0_g, 'r': u0_r}
            data = {'g': all_data[:len(self.iso_data['g'].T)*spokes], 'r': all_data[len(self.iso_data['g'].T)*spokes:]}
            def decompostion(x, theta, band):
                Re = self.super_ellipse(np.deg2rad(theta), *pars[rp:rp+4])
                h = self.super_ellipse(np.deg2rad(theta), *pars[rp+4:rp+8])
                return self.two_component(x, ue[band], u0[band], Re, h, n)
            data_g = np.concatenate([decompostion(data['g'].reshape(spokes, -1)[i], angle_targets['g'][i], 'g') for i in range(spokes)])
            data_r = np.concatenate([decompostion(data['r'].reshape(spokes, -1)[i], angle_targets['r'][i], 'r') for i in range(spokes)])
            return np.concatenate([data_g, data_r])
        
        def bulge_bar_disk_2D_model(pars, all_data):
            rp = 8
            ue_bulge_g, ue_bulge_r, ue_bar_g, ue_bar_r, u0_g, u0_r, n_bulge, n_bar = pars[:rp]
            ue_bulge = {'g': ue_bulge_g, 'r': ue_bulge_r}
            ue_bar = {'g': ue_bar_g, 'r': ue_bar_r}
            u0 = {'g': u0_g, 'r': u0_r}
            data = {'g': all_data[:len(self.iso_data['g'].T)*spokes], 'r': all_data[len(self.iso_data['g'].T)*spokes:]}
            def decompostion(x, theta, band):
                # b1, b2, b3, b4  = pars[rp:rp+4]
                # c1, c2, c3, c4 = pars[rp+4:rp+8]
                # r_mut = np.sqrt(b2*c2)
                Re_bulge = self.super_ellipse(np.deg2rad(theta), *pars[rp:rp+4])
                Re_bar = self.super_ellipse(np.deg2rad(theta), *pars[rp+4:rp+8])
                h = self.super_ellipse(np.deg2rad(theta), *pars[rp+8:rp+12])
                return self.three_component(x, ue_bulge[band], ue_bar[band], u0[band], Re_bulge, Re_bar, h, n_bulge, n_bar)
            data_g = np.concatenate([decompostion(data['g'].reshape(spokes, -1)[i], angle_targets['g'][i], 'g') for i in range(spokes)])
            data_r = np.concatenate([decompostion(data['r'].reshape(spokes, -1)[i], angle_targets['r'][i], 'r') for i in range(spokes)])
            return np.concatenate([data_g, data_r])

        angle_targets = {'g': np.linspace(0, 360-360/spokes, spokes), 'r': np.linspace(180/spokes, 360-180/spokes, spokes)}
        x_data_g = np.concatenate([self.target_angle(self.iso_data['g'].T, np.deg2rad(ang_i)) for ang_i in angle_targets['g']])
        y_data_g = np.tile(self.iso_data['g'][11], spokes)
        x_err_g = np.sqrt(np.tile(self.iso_data['g'][6], spokes)*np.tile(self.iso_data['g'][7], spokes))*0.262
        y_err_g = np.tile(self.iso_data['g'][10], spokes)

        x_data_r = np.concatenate([self.target_angle(self.iso_data['r'].T, np.deg2rad(ang_i)) for ang_i in angle_targets['r']])
        y_data_r = np.tile(self.iso_data['r'][11], spokes)
        x_err_r = np.sqrt(np.tile(self.iso_data['r'][6], spokes)*np.tile(self.iso_data['r'][7], spokes))*0.262
        y_err_r = np.tile(self.iso_data['r'][10], spokes)

        x_data = np.concatenate([x_data_g, x_data_r])
        y_data = np.concatenate([y_data_g, y_data_r])
        x_err = np.concatenate([x_err_g, x_err_r])
        y_err = np.concatenate([y_err_g, y_err_r])
        data = Data(x_data, y_data, we=1/x_err**2, wd=1/y_err**2)
        self.decomp_data = {'g': [x_data_g, y_data_g, x_err_g, y_err_g], 'r': [x_data_r, y_data_r, x_err_r, y_err_r]}

        p0_1 = [23, 23, 3,  10, 8, 1, 2]
        odr_1 = ODR(data, Model(bulge_2D_model), beta0=p0_1, maxit=3000)
        odr_1.set_job(fit_type=mode)
        output_1 = odr_1.run()
        BIC_1 = self.BIC(output_1.beta, x_data, y_data, bulge_2D_model)
        print(f'One componenet, BIC: {BIC_1[0]:.2f}, RSS: {BIC_1[1]:.2f}, fitted parameters: {BIC_1[2]}') if verbose else None
        
        p0_2 = [21, 20, 21, 20, 3,    3, 2, 3, 2,    5, 4, 3, 2]
        odr_2 = ODR(data, Model(bulge_disk_2D_model), beta0=p0_2, maxit=3000)
        odr_2.set_job(fit_type=mode)
        output_2 = odr_2.run()
        BIC_2 = self.BIC(output_2.beta, x_data, y_data, bulge_disk_2D_model)
        print(f'Two componenet, BIC: {BIC_2[0]:.2f}, RSS: {BIC_2[1]:.2f}, fitted parameters: {BIC_2[2]}') if verbose else None

        if bar:
            p0_3 = [18,19, 21,20, 21.5, 21, 0.8, 0.2,  1,1,1.8,2,  3,1,1.8,4,  5,4,2.6,1.5]
            odr_3 = ODR(data, Model(bulge_bar_disk_2D_model), beta0=p0_3, maxit=3000)
            odr_3.set_job(fit_type=mode)
            output_3 = odr_3.run()

            # b2  = output_3.beta[6]
            # c2 = output_3.beta[10]
            # r_mut = np.sqrt(b2*c2)
            # output_3.beta[6] = r_mut
            # output_3.beta[10] = r_mut
            BIC_3 = self.BIC(output_3.beta, x_data, y_data, bulge_bar_disk_2D_model)
            print(f'Three componenet, BIC: {BIC_3[0]:.2f}, RSS: {BIC_3[1]:.2f}, fitted parameters: {BIC_3[2]}')
            self.decomp = [[output_1.beta, output_1.sd_beta], [output_2.beta, output_2.sd_beta], [output_3.beta, output_3.sd_beta], [BIC_1, BIC_2, BIC_3, np.argmin([BIC_1[0], BIC_2[0], BIC_3[0]])]]
            result = ['single Bulge preferred', 'Bulge + Disk preferred', 'Bulge + Bar + Disk preferred'][self.decomp[3][3]]
            print(result) if verbose else None
        else:
            self.decomp = [[output_1.beta, output_1.sd_beta], [output_2.beta, output_2.sd_beta], [BIC_1, BIC_2, np.argmin([BIC_1[0], BIC_2[0]])]]
            result = ['single Bulge preferred', 'Bulge + Disk preferred'][self.decomp[3][3]]
            print(result) if verbose else None
        

    def plot_spokes(self, band, spokes=12, sigma=3, n_model=2):
       
        theta = {'g': np.linspace(0, 360-360/spokes, spokes), 'r': np.linspace(180/spokes, 360-180/spokes, spokes)}[band][::2]
        bd_data = self.decomp_data[band]
        x_data, y_data = bd_data[0].reshape(spokes, -1)[::int(spokes/6)], bd_data[1].reshape(spokes, -1)[::int(spokes/6)]
        x_err, y_err = bd_data[2].reshape(spokes, -1)[::int(spokes/6)], bd_data[3].reshape(spokes, -1)[::int(spokes/6)]

        rp = [3, 5, 8][n_model-1]
        fig, axis = plt.subplots(figsize=(18, 10), ncols=2, nrows=3, dpi=100)
        for i in range(6):
            ax = axis.flatten()[i]
            x_ax = np.linspace(x_data.min()-0.2, x_data.max()+0.2, 100)
            ax.errorbar(x_data[i], y_data[i],yerr=sigma*y_err[i], xerr=sigma*x_err[i], fmt='k.', zorder=0, label=fr'{sigma} sigma,  $\theta =${theta[i]:.0f}$^\circ$')
            ax.set_ylim([y_data.min()-0.5, 25.5])
            ax.invert_yaxis()
            ax.legend()

        if n_model == 3:
            ue_bulge_g, ue_bulge_r, ue_bar_g, ue_bar_r, u0_g, u0_r, n_bulge, n_bar = self.decomp[n_model-1][0][:rp]
            ue_bulge = {'g': ue_bulge_g, 'r': ue_bulge_r}[band]
            ue_bar = {'g': ue_bar_g, 'r': ue_bar_r}[band]
            u0_disk = {'g': u0_g, 'r': u0_r}[band]
            for i in range(6):
                ax = axis.flatten()[i]
                Re_bulge = self.super_ellipse(np.deg2rad(theta[i]), *self.decomp[n_model-1][0][rp:rp+4])
                Re_bar = self.super_ellipse(np.deg2rad(theta[i]), *self.decomp[n_model-1][0][rp+4:rp+8])
                h_disk = self.super_ellipse(np.deg2rad(theta[i]), *self.decomp[n_model-1][0][rp+8:rp+12])
                ax.plot(x_ax, self.three_component(x_ax, ue_bulge, ue_bar, u0_disk, Re_bulge, Re_bar, h_disk, n_bulge, n_bar), 'r-', label='combined')
                ax.plot(x_ax, self.one_component(x_ax, ue_bulge, Re_bulge, n_bulge), 'g--', label='bulge')
                ax.plot(x_ax, self.one_component(x_ax, ue_bar, Re_bar, n_bar), 'g--', label='bar', color='lime')
                ax.plot(x_ax, self.one_component_disk(x_ax, u0_disk, h_disk), 'b--', label='disk')

        elif n_model == 2:
            ue_g, ue_r, u0_g, u0_r, n_sersic = self.decomp[n_model-1][0][:rp]
            ue_bulge = {'g': ue_g, 'r': ue_r}[band]
            u0_disk = {'g': u0_g, 'r': u0_r}[band]
            for i in range(6):
                ax = axis.flatten()[i]
                Re_bulge = self.super_ellipse(np.deg2rad(theta[i]), *self.decomp[n_model-1][0][rp:rp+4])
                h_disk = self.super_ellipse(np.deg2rad(theta[i]), *self.decomp[n_model-1][0][rp+4:rp+8])
                ax.plot(x_ax, self.two_component(x_ax, ue_bulge, u0_disk, Re_bulge, h_disk, n_sersic), 'r-', label='combined')
                ax.plot(x_ax, self.one_component(x_ax, ue_bulge, Re_bulge, n_sersic), 'g--', label='bulge')
                ax.plot(x_ax, self.one_component_disk(x_ax, u0_disk, h_disk), 'b--', label='disk')

        elif n_model == 1:
            ue_g, ue_r, n_sersic = self.decomp[n_model-1][0][:rp]
            ue_bulge = {'g': ue_g, 'r': ue_r}[band]
            for i in range(6):
                ax = axis.flatten()[i]
                Re_bulge = self.super_ellipse(np.deg2rad(theta[i]), *self.decomp[n_model-1][0][rp:rp+4])
                ax.plot(x_ax, self.one_component(x_ax, ue_bulge, Re_bulge, n_sersic), 'r-', label='bulge')     

        plt.tight_layout()

    def SB_profile(self, r, theta, band, model='all', n_model=2, psf=0):
        rp = [3, 5, 8][n_model-1]
        arcsec2px = np.array([0.262, 0.262, 1, 1])
        models = {'bulge': [self.bulge, self.one_component][psf], 'disk': [self.disk, self.one_component_disk][psf],
                  'two_c': [self.combine, self.two_component][psf], 'three_c': [self.combine_3, self.three_component][psf]}
        if n_model == 3:
            ue_bulge_g, ue_bulge_r, ue_bar_g, ue_bar_r, u0_g, u0_r, n_bulge, n_bar = self.decomp[n_model-1][0][:rp]
            ue_bulge = {'g': ue_bulge_g, 'r': ue_bulge_r}[band]
            ue_bar = {'g': ue_bar_g, 'r': ue_bar_r}[band]
            u0_disk = {'g': u0_g, 'r': u0_r}[band]
            Re_bulge = self.super_ellipse(theta, *self.decomp[n_model-1][0][rp:rp+4]/arcsec2px)
            Re_bar = self.super_ellipse(theta, *self.decomp[n_model-1][0][rp+4:rp+8]/arcsec2px)
            h_disk = self.super_ellipse(theta, *self.decomp[n_model-1][0][rp+8:rp+12]/arcsec2px)
            if model == 'bulge':
                return models['bulge'](r, ue_bulge, Re_bulge, n_bulge)
            elif model == 'bar':
                return models['bulge'](r, ue_bar, Re_bar, n_bar)
            elif model == 'disk':
                return models['disk'](r, u0_disk, h_disk)
            else:
                return models['three_c'](r, ue_bulge, ue_bar, u0_disk, Re_bulge, Re_bar, h_disk, n_bulge, n_bar)
            
        elif n_model == 2:
            ue_g, ue_r, u0_g, u0_r, n_sersic = self.decomp[n_model-1][0][:rp]
            ue_bulge = {'g': ue_g, 'r': ue_r}[band]
            u0_disk = {'g': u0_g, 'r': u0_r}[band]
            Re_bulge = self.super_ellipse(theta, *self.decomp[n_model-1][0][rp:rp+4]/arcsec2px)
            h_disk = self.super_ellipse(theta, *self.decomp[n_model-1][0][rp+4:rp+8]/arcsec2px)
            if model == 'bulge':
                return models['bulge'](r, ue_bulge, Re_bulge, n_sersic)
            elif model == 'disk':
                return models['disk'](r, u0_disk, h_disk)
            else:
                 return models['two_c'](r, ue_bulge, u0_disk, Re_bulge, h_disk, n_sersic)
            
        elif n_model == 1:
            ue_g, ue_r, n_sersic = self.decomp[n_model-1][0][:rp]
            ue_bulge = {'g': ue_g, 'r': ue_r}[band]
            Re_bulge = self.super_ellipse(theta, *self.decomp[n_model-1][0][rp:rp+4]/arcsec2px)
            return models['bulge'](r, ue_bulge, Re_bulge, n_sersic)
    
  
    def SB_isophote(self, isophote, band, n_model, model='both'): 
        def radial(isophote, theta):
            sb_profile = lambda r: self.SB_profile(r, theta, band=band, n_model=n_model, model=model) - isophote
            return fsolve(sb_profile, 1)

        theta = np.linspace(0, 2*np.pi, 50)
        r = np.vectorize(radial)(isophote, theta)

        out_pars = curve_fit(self.super_ellipse, theta, r, p0=[r.max(), r.min(), 0,  2], maxfev=5000)
        check = np.mean(self.SB_profile(r, theta)) - isophote
        return [out_pars[0], np.sqrt(np.diag(out_pars[1])), np.round(check, 3)] 

    def plot_SB_profile(self, band, isophote=False, subtarct=False, n_model=2):
        n_size = np.arange(len(self.image[band]))
        xm, ym = np.meshgrid(n_size, n_size)
        rm = np.linalg.norm(np.stack([xm, ym]).T - self.center, axis=2)

        theta_top = np.arccos(np.dot(np.stack([xm, ym]).T - self.center, np.array([0,1]))/rm)[len(n_size)//2:]
        theta_bot = np.arccos(np.dot(np.stack([xm, ym]).T - self.center, np.array([0,-1]))/rm)[:len(n_size)//2]
        thetam = np.vstack([theta_bot, theta_top])
        thetam[np.isnan(thetam)] = 0

        bulge_arr = self.SB_profile(rm, thetam, model='bulge', n_model=n_model, band=band)
        bar_arr = self.SB_profile(rm, thetam, model='bar', n_model=n_model, band=band)
        disk_arr = self.SB_profile(rm, thetam, model='disk', n_model=n_model, band=band)
        both_arr = self.SB_profile(rm, thetam, model='all', n_model=n_model, band=band)

        std = self.gobj[band].brick['psfsize']/(2*np.sqrt(2*np.log(2)))/0.262
        kernal = Gaussian2DKernel(x_stddev=std, y_stddev=std)
        both_arr = -2.5*np.log10(convolve_fft(10**(-0.4*both_arr), kernal))

        reds, greens, blues = self.alpha_colormap()

        if not subtarct:
            fig, ax = self.gobj[band].plot()
            rp = [3, 5, 8][n_model-1]
            arcsec2px = np.array([0.262, 0.262, 1, 1])
            if n_model == 3:
                self.patch_super_ellipse(self.decomp[n_model-1][0][rp:rp+4]/arcsec2px, self.center, ax, 'red', label='Bulge $R_e$')
                self.patch_super_ellipse(self.decomp[n_model-1][0][rp+4:rp+8]/arcsec2px, self.center, ax, 'green', label='Bar $R_e$')

                disk_a, disk_b, disk_pa, disk_n = self.decomp[n_model-1][0][rp+8:rp+12]
                u0_g, u0_r = self.decomp[n_model-1][0][4:rp-2]
                u0_disk = {'g': u0_g, 'r': u0_r}[band]
                ue_disk, (disk_a, disk_b) = self.transform(u0_disk, [disk_a, disk_b], n=1)
                self.patch_super_ellipse(np.array([disk_a, disk_b, disk_pa, disk_n])/arcsec2px, self.center, ax, 'blue', label='Disk $R_e$')
                ax.imshow(disk_arr, cmap=blues, vmax=26)
                ax.imshow(bar_arr, cmap=greens, vmax=26)

            elif n_model == 2:
                self.patch_super_ellipse(self.decomp[n_model-1][0][rp:rp+4]/arcsec2px, self.center, ax, 'red', label='Bulge $R_e$')
                u0_g, u0_r = self.decomp[n_model-1][0][2:rp-1]
                u0_disk = {'g': u0_g, 'r': u0_r}[band]
                disk_a, disk_b, disk_pa, disk_n = self.decomp[n_model-1][0][rp+4:rp+8]
                ue_disk, (disk_a, disk_b) = self.transform(u0_disk, [disk_a, disk_b], n=1)
                self.patch_super_ellipse(np.array([disk_a, disk_b, disk_pa, disk_n])/arcsec2px, self.center, ax, 'blue', label='Disk $R_e$')
                ax.imshow(disk_arr, cmap=blues, vmax=26)

            elif n_model == 1:
                self.patch_super_ellipse(self.decomp[n_model-1][0][rp:rp+4]/arcsec2px, self.center, ax, 'red', label='Bulge $R_e$')
            
            ax.imshow(bulge_arr, cmap=reds, vmax=26)
            if isophote:
                wi = 0.1
                mag_25 = np.where((self.image.T > isophote-wi) & (self.image[band].T < isophote+wi))
                ax.plot(mag_25[0], mag_25[1], 'g.', c='green', ms=2, zorder=5, label=f'{isophote} mag/arcsec$^2$')

                pars, errs, check = self.SB_isophote(isophote, band, n_model)
                self.patch_super_ellipse(pars, self.center, ax, 'lime') if check == 0 else None

            ax.legend(framealpha=1, fontsize=7, loc=1)

        if subtarct:
            def sub_mag(m1, m2):
                return -2.5*np.log10(np.abs(10**(-0.4*m1) - 10**(-0.4*m2)))
    
            fig, (ax1, ax2, ax3) = plt.subplots(figsize=(18, 6), ncols=3, dpi=100)
            sky_mag = self.gobj[band].cutout['mag_raw'].copy()
            sky_mag[np.isnan(sky_mag)] = self.gobj[band].brick['psfdepth']
            ax1.imshow(sky_mag, origin='lower', cmap='gray', vmax=30, vmin=19)

            ax2.imshow(disk_arr, cmap=blues, origin='lower', vmax=25) if (n_model == 2) or (n_model == 3) else None
            ax2.imshow(bar_arr, cmap=greens, origin='lower', vmax=25) if n_model == 3 else None
            ax2.imshow(bulge_arr, cmap=reds, origin='lower', vmax=25)

            ax3.imshow(sub_mag(sky_mag, both_arr), origin='lower', cmap='gray', vmax=30, vmin=19)
            plt.tight_layout()

