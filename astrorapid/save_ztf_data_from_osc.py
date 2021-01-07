import numpy as np
from collections import OrderedDict
import json
from six.moves.urllib.request import urlopen
from astroquery.irsa_dust import IrsaDust
import astropy.coordinates as coord
import astropy.units as u
import pickle
import urllib.request

from astrorapid.helpers import delete_indexes, convert_lists_to_arrays


def read_json(url):
    response = urlopen(url)
    return json.loads(response.read(), object_pairs_hook=OrderedDict)


def read_lasair_json(object_name='ZTF18acsovsw'):
    """
    Read light curve from lasair website API based on object name.

    Parameters
    ----------
    object_name : str
        The LASAIR object name. E.g. object_name='ZTF18acsovsw'

    """
    print(object_name)
    if isinstance(object_name, tuple):
        object_name, z_in = object_name
    else:
        z_in = None

    url = 'https://lasair.roe.ac.uk/object/{}/json/'.format(object_name)

    data = read_json(url)

    objid = data['objectId']
    ra = data['objectData']['ramean']
    dec = data['objectData']['decmean']
    # lasair_classification = data['objectData']['classification']
    tns_info = data['objectData']['annotation']
    photoZ = None
    for cross_match in data['crossmatches']:
        # print(cross_match)
        photoZ = cross_match['photoZ']
        separation_arcsec = cross_match['separationArcsec']
        catalogue_object_type = cross_match['catalogue_object_type']
    if z_in is not None and not np.isnan(z_in):
        redshift = z_in
    else:
        if photoZ is None:  # TODO: Get correct redshift
            try:
                if "z=" in tns_info:
                    photoZ = tns_info.split('z=')[1]
                    redshift = float(photoZ.replace(')', '').split()[0])
                    # print("PHOTOZZZZZZZZZZZZ", redshift, tns_info)
                elif "Z=" in tns_info:
                    photoZ = tns_info.split('Z=')[1]
                    redshift = float(photoZ.split()[0])
                    # print("PHOTOZZZZZZZZZZZZ", redshift, tns_info)
                else:
                    # return
                    print("TRYING ARBITRARY GUESS REDSHIFT = 0.1")
                    redshift = None
            except Exception as e:
                # return
                redshift = None
                print(e)
        else:
            redshift = photoZ

    print("Redshift is {}".format(redshift))
    if redshift is not None:
        objid += "_z={}".format(round(redshift, 2))

    # Get extinction  TODO: Maybe add this to RAPID code
    coo = coord.SkyCoord(ra * u.deg, dec * u.deg, frame='icrs')
    dust = IrsaDust.get_query_table(coo, section='ebv')
    mwebv = dust['ext SandF mean'][0]
    print("MWEBV")
    print(mwebv)

    mjd = []
    passband = []
    mag = []
    magerr = []
    photflag = []
    zeropoint = []
    dc_mag = []
    dc_magerr = []
    magnr, sigmagnr, isdiffpos = [], [], []
    for cand in data['candidates']:
        mjd.append(cand['mjd'])
        passband.append(cand['fid'])
        mag.append(cand['magpsf'])
        if 'sigmapsf' in cand:
            magerr.append(cand['sigmapsf'])
            photflag.append(4096)

            if cand['magzpsci'] == 0:
                print("NO ZEROPOINT")
                zeropoint.append(26.2)  # LASAIR zeropoints used to be wrong, but no longer using these anyway...
            else:
                zeropoint.append(cand['magzpsci'])  #26.2) #
            # if cand['magzpsci'] == 0:
            #     print(object_name, zeropoint)
            #     raise Exception
            #     return

            zeropoint.append(cand['magzpsci'])
            dc_mag.append(cand['dc_mag'])
            dc_magerr.append(cand['dc_sigmag'])
            magnr.append(cand['magnr'])
            sigmagnr.append(cand['sigmagnr'])
            isdiffpos.append(cand['isdiffpos'])
        else:
            magerr.append(np.nan)#0.01 * cand['magpsf'])  #magerr.append(None)  #magerr.append(0.1 * cand['magpsf'])  #
            photflag.append(0)
            zeropoint.append(np.nan)#26.2)
            dc_mag.append(np.nan)
            dc_magerr.append(np.nan)
            magnr.append(np.nan)
            sigmagnr.append(np.nan)
            isdiffpos.append(None)

    mjd, passband, mag, magerr, photflag, zeropoint, dc_mag, dc_magerr, magnr, sigmagnr, isdiffpos = convert_lists_to_arrays(mjd, passband, mag, magerr, photflag, zeropoint, dc_mag, dc_magerr, magnr, sigmagnr, isdiffpos)

    # deleteindexes = np.where(magerr == None)  # [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]  #
    # mjd, passband, mag, magerr, photflag, zeropoint, dc_mag, dc_magerr, magnr, sigmagnr, isdiffpos = delete_indexes(deleteindexes, mjd, passband, mag, magerr, photflag, zeropoint, dc_mag, dc_magerr, magnr, sigmagnr, isdiffpos)
    deleteindexes = np.where((photflag==0) & (mjd > min(mjd[photflag>0])))  # delete where nondetections after first detection
    mjd, passband, mag, magerr, photflag, zeropoint, dc_mag, dc_magerr, magnr, sigmagnr, isdiffpos = delete_indexes(deleteindexes, mjd, passband, mag, magerr, photflag, zeropoint, dc_mag, dc_magerr, magnr, sigmagnr, isdiffpos)
    deleteindexes = np.where((mag < (np.median(mag[photflag==0]) - 0.5*np.std(mag[photflag==0]))) & (photflag==0)) # Remove non detection outliers
    mjd, passband, mag, magerr, photflag, zeropoint, dc_mag, dc_magerr, magnr, sigmagnr, isdiffpos = delete_indexes(deleteindexes, mjd, passband, mag, magerr, photflag, zeropoint, dc_mag, dc_magerr, magnr, sigmagnr, isdiffpos)

    return mjd, passband, mag, magerr, photflag, zeropoint, ra, dec, objid, redshift, mwebv, dc_mag, dc_magerr, magnr, sigmagnr, isdiffpos


def classify_lasair_light_curves(object_names=('ZTF18acsovsw',), savename='', sntype=''):
    light_curve_list = []
    mjds, passbands, mags, magerrs, zeropoints, photflags = [], [], [], [], [], []
    dc_mags, dc_magerrs, magnrs, sigmagnrs, isdiffposs = [], [], [], [], []
    obj_names = []
    ras, decs, objids, redshifts, mwebvs = [], [], [], [], []
    peakmags_g, peakmags_r = [], []
    for object_name in object_names:
        try:
            mjd, passband, mag, magerr, photflag, zeropoint, ra, dec, objid, redshift, mwebv, dc_mag, dc_magerr, magnr, sigmagnr, isdiffpos = read_lasair_json(object_name)
            sortidx = np.argsort(mjd)
            mjds.append(mjd[sortidx])
            passbands.append(passband[sortidx])
            mags.append(mag[sortidx])
            magerrs.append(magerr[sortidx])
            zeropoints.append(zeropoint[sortidx])
            photflags.append(photflag[sortidx])
            dc_mags.append(dc_mag[sortidx])
            dc_magerrs.append(dc_magerr[sortidx])
            magnrs.append(magnr[sortidx])
            sigmagnrs.append(sigmagnr[sortidx])
            isdiffposs.append(isdiffpos[sortidx])

            obj_names.append(object_name)
            ras.append(ra)
            decs.append(dec)
            objids.append(f"{sntype}_{objid}")
            redshifts.append(redshift)
            mwebvs.append(mwebv)
            peakmags_g.append(min(mag[passband==1]))
            peakmags_r.append(min(mag[passband==2]))

        except Exception as e:
            print(e)
            continue

        A = 10. ** (0.4 * 24.8)
        flux = A * 10. ** (-0.4 * (mag))
        fluxerr = np.abs(flux * magerr * (np.log(10.) / 2.5))

        passband = np.where((passband == 1) | (passband == '1'), 'g', passband)
        passband = np.where((passband == 2) | (passband == '2'), 'r', passband)

        # Set photflag detections when S/N > 5
        photflag2 = np.zeros(len(flux))
        photflag2[flux / fluxerr > 5] = 4096
        photflag2[np.where(mjd == min(mjd[photflag2 == 4096]))] = 6144

        mjd_first_detection = min(mjd[photflag == 4096])
        photflag[np.where(mjd == mjd_first_detection)] = 6144

        deleteindexes = np.where(((passband == 3) | (passband == '3')) | (mjd > mjd_first_detection) & (photflag == 0))
        if deleteindexes[0].size > 0:
            print("Deleting indexes {} at mjd {} and passband {}".format(deleteindexes, mjd[deleteindexes], passband[deleteindexes]))
        mjd, passband, flux, fluxerr, zeropoint, photflag = delete_indexes(deleteindexes, mjd, passband, flux, fluxerr, zeropoint, photflag)

        light_curve_list += [(mjd, flux, fluxerr, passband, photflag, ra, dec, objid, redshift, mwebv)]

    with open(savename, 'wb') as f:
        pickle.dump([mjds, passbands, mags, magerrs, photflags, zeropoints, dc_mags, dc_magerrs, magnrs, sigmagnrs, isdiffposs, ras, decs, objids, redshifts, mwebvs], f)
    np.savez('save_real_ZTF_unprocessed_data_snia_osc_27Oct2020.npz', mjds=mjds, passbands=passbands, mags=mags, magerrs=magerrs, photflags=photflags, zeropoints=zeropoints, ras=ras, decs=decs, objids=objids, redshifts=redshifts, mwebvs=mwebvs)# , peakflux_g=peakfluxes_g, peakflux_r=peakfluxes_r)
    print("finished")


if __name__ == '__main__':
    # Get all sne as json
    json_url = 'https://api.sne.space/catalog?format=json'
    with urllib.request.urlopen(json_url) as url:
        data = json.loads(url.read().decode())

    # Check if ZTF alias
    ZTF_names = []
    ZTF_keys = []
    ZTF_redshifts = []
    ZTF_types = []

    SN = {
        'Ia': {},
        'Ia91T': {},
        'Ia91bg': {},
        'Iapec': {},
        'Iacsm': {},
        'Iax': {},
        'II': {},
        'IIP': {},
        'IIL': {},
        'IIpec': {},
        'IIn': {},
        'IIb': {},
        'Ib': {},
        'Ibn': {},
        'Ic': {},
        'IcBL': {},
        'Ibc': {},
        'SLSN': {},
        'SLSNI': {},
        'SLSNII': {},
        'CC': {}
    }


    for osc_name in data.keys():
        for alias in data[osc_name]['alias']:
            for name in alias.values():
                if name[0:3] == 'ZTF':
                    redshifts = np.array([entry['value'] for entry in data[osc_name]['redshift']]).astype('float')
                    claimed_types = [entry['value'] for entry in data[osc_name]['claimedtype']]
                    redshift = np.median(redshifts)
                    if len(claimed_types) == 0 or (len(claimed_types) == 1 and claimed_types[0] in ['Candidate', 'Other', 'other', 'removed', 'NT']):
                        continue

                    if len(claimed_types) == 1:
                        if claimed_types[0] in ['Ia',]:
                            SN['Ia'][name] = redshift
                        elif claimed_types[0] in ['Ia-91T']:
                            SN['Ia91T'][name] = redshift
                        elif claimed_types[0] in ['Ia-91bg']:
                            SN['Ia91bg'][name] = redshift
                        elif claimed_types[0] in ['Ia-02cx', 'Iax']:
                            SN['Iax'][name] = redshift
                        elif claimed_types[0] in ['Ia CSM']:
                            SN['Iacsm'][name] = redshift
                        elif claimed_types[0] in ['Ia Pec']:
                            SN['Iapec'][name] = redshift
                        elif claimed_types[0] in ['II']:
                            SN['II'][name] = redshift
                        elif claimed_types[0] in ['II P']:
                            SN['IIP'][name] = redshift
                        elif claimed_types[0] in ['II L']:
                            SN['IIL'][name] = redshift
                        elif claimed_types[0] in ['II Pec']:
                            SN['IIpec'][name] = redshift
                        elif claimed_types[0] in ['IIn']:
                            SN['IIn'][name] = redshift
                        elif claimed_types[0] in ['IIb', 'II/IIb']:
                            SN['IIb'][name] = redshift
                        elif claimed_types[0] in ['Ib']:
                            SN['Ib'][name] = redshift
                        elif claimed_types[0] in ['Ibn']:
                            SN['Ibn'][name] = redshift
                        elif claimed_types[0] in ['Ic']:
                            SN['Ic'][name] = redshift
                        elif claimed_types[0] in ['Ic BL']:
                            SN['IcBL'][name] = redshift
                        elif claimed_types[0] in ['Ib/c']:
                            SN['Ibc'][name] = redshift
                        elif claimed_types[0] in ['SLSN']:
                            SN['SLSN'][name] = redshift
                        elif claimed_types[0] in ['SLSN-I']:
                            SN['SLSNI'][name] = redshift
                        elif claimed_types[0] in ['SLSN-II']:
                            SN['SLSNII'][name] = redshift
                        else:
                            pass
                            # ZTF_names.append(name)
                            # ZTF_keys.append(osc_name)
                            # ZTF_redshifts.append(np.median(redshifts))
                            # ZTF_types.append(claimed_types)
                            # print(osc_name, name, redshifts, claimed_types)
                    elif len(claimed_types) == 2:
                        if 'Ia-91T' in claimed_types and 'Ia' in claimed_types:
                            SN['Ia91T'][name] = redshift
                        elif 'Ia-91bg' in claimed_types and 'Ia' in claimed_types:
                            SN['Ia91bg'][name] = redshift
                        elif 'Ia Pec' in claimed_types and 'Ia' in claimed_types:
                            SN['Iapec'][name] = redshift
                        elif 'Ia-99aa' in claimed_types and 'Ia' in claimed_types:
                            SN['Ia91T'][name] = redshift
                        elif 'IIn' in claimed_types and 'II' in claimed_types:
                            SN['IIn'][name] = redshift
                        elif 'II P' in claimed_types and 'II' in claimed_types:
                            SN['IIP'][name] = redshift
                        elif 'IIn' in claimed_types and 'II' in claimed_types:
                            SN['IIn'][name] = redshift
                        elif 'IIb' in claimed_types and 'II' in claimed_types:
                            SN['IIb'][name] = redshift
                        elif 'Ib' in claimed_types and 'CC' in claimed_types:
                            SN['Ib'][name] = redshift
                        elif 'II' in claimed_types and 'CC' in claimed_types:
                            SN['II'][name] = redshift
                        elif claimed_types[0][0:2] in ['II', 'Ib', 'Ic'] and claimed_types[1][0:2] in ['II', 'Ib', 'Ic']:
                            SN['CC'][name] = redshift
                        elif 'SLSN-I' in claimed_types and 'SLSN' in claimed_types:
                            SN['SLSNI'][name] = redshift
                        elif 'SLSN-II' in claimed_types and 'SLSN' in claimed_types:
                            SN['SLSNII'][name] = redshift
                        elif claimed_types[0][0:2] == 'Ia' and claimed_types[1][0:2] == 'Ia':
                            SN['Ia'][name] = redshift
                        else:
                            pass
                            # ZTF_names.append(name)
                            # ZTF_keys.append(osc_name)
                            # ZTF_redshifts.append(np.median(redshifts))
                            # ZTF_types.append(claimed_types)
                            # print(osc_name, name, redshifts, claimed_types)
                    else:
                        pass
                        # ZTF_names.append(name)
                        # ZTF_keys.append(osc_name)
                        # ZTF_redshifts.append(np.median(redshifts))
                        # ZTF_types.append(claimed_types)
                        # print(osc_name, name, redshifts, claimed_types)

                    ZTF_names.append(name)
                    ZTF_keys.append(osc_name)
                    ZTF_redshifts.append(np.median(redshifts))
                    ZTF_types.append(claimed_types)
                    print(osc_name, name, redshifts, claimed_types)

    for sntype in SN.keys():
        savename = f'data/real_ZTF_data_from_osc/ZTF_data_{sntype}_osc-27-Oct-2020_keep_zpt0_objects.pickle'
        names_and_redshifts = list(SN[sntype].items())
        classify_lasair_light_curves(object_names=names_and_redshifts, savename=savename, sntype=sntype)

    # List counts of each type
    print([(sntype, len(snlist)) for sntype, snlist in SN.items()])

    sndata = {}
    for sntype in SN.keys():
        savename = f'data/real_ZTF_data_from_osc/ZTF_data_{sntype}_osc-27-Oct-2020_keep_zpt0_objects.pickle'
        with open(savename, 'rb') as f:
            sndata[sntype] = pickle.load(f)
    print([(sntype, len(data[0])) for sntype, data in sndata.items()])

