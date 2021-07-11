import numpy as np
from collections import OrderedDict
import json
from six.moves.urllib.request import urlopen
from astroquery.irsa_dust import IrsaDust
import astropy.coordinates as coord
import astropy.units as u


from astrorapid.classify import Classify
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
    if z_in is not None:
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
    for cand in data['candidates']:
        mjd.append(cand['mjd'])
        passband.append(cand['fid'])
        mag.append(cand['magpsf'])
        if 'sigmapsf' in cand:
            magerr.append(cand['sigmapsf'])
            photflag.append(4096)
            if cand['magzpsci'] == 0:
                print("NO ZEROPOINT")
                zeropoint.append(26.2)  # TODO: Tell LASAIR their zeropoints are wrong
            else:
                zeropoint.append(cand['magzpsci'])  #26.2) #
            # zeropoint.append(cand['magzpsci'])
        else:
            magerr.append(None)#0.01 * cand['magpsf'])  #magerr.append(None)  #magerr.append(0.1 * cand['magpsf'])  #
            photflag.append(0)
            zeropoint.append(None)

    mjd, passband, mag, magerr, photflag, zeropoint = convert_lists_to_arrays(mjd, passband, mag, magerr, photflag, zeropoint)

    deleteindexes = np.where(magerr == None)  # [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]  #
    mjd, passband, mag, magerr, photflag, zeropoint = delete_indexes(deleteindexes, mjd, passband, mag, magerr, photflag, zeropoint)
    deleteindexes = np.where((photflag==0) & (mjd > min(mjd[photflag>0])))  # delete where nondetections after first detection
    mjd, passband, mag, magerr, photflag, zeropoint = delete_indexes(deleteindexes, mjd, passband, mag, magerr, photflag, zeropoint)
    deleteindexes = np.where((mag < (np.median(mag[photflag==0]) - 0.5*np.std(mag[photflag==0]))) & (photflag==0)) # Remove non detection outliers
    mjd, passband, mag, magerr, photflag, zeropoint = delete_indexes(deleteindexes, mjd, passband, mag, magerr, photflag, zeropoint)

    return mjd, passband, mag, magerr, photflag, zeropoint, ra, dec, objid, redshift, mwebv


def classify_lasair_light_curves(object_names=('ZTF18acsovsw',), plot=True, figdir='.'):
    light_curve_list = []
    peakfluxes_g, peakfluxes_r = [], []
    mjds, passbands, mags, magerrs, zeropoints, photflags = [], [], [], [], [], []
    obj_names = []
    ras, decs, objids, redshifts, mwebvs = [], [], [], [], []
    peakmags_g, peakmags_r = [], []
    for object_name in object_names:
        try:
            mjd, passband, mag, magerr, photflag, zeropoint, ra, dec, objid, redshift, mwebv = read_lasair_json(object_name)
            sortidx = np.argsort(mjd)
            mjds.append(mjd[sortidx])
            passbands.append(passband[sortidx])
            mags.append(mag[sortidx])
            magerrs.append(magerr[sortidx])
            zeropoints.append(zeropoint[sortidx])
            photflags.append(photflag[sortidx])
            obj_names.append(object_name)
            ras.append(ra)
            decs.append(dec)
            objids.append(objid)
            redshifts.append(redshift)
            mwebvs.append(mwebv)
            peakmags_g.append(min(mag[passband==1]))
            peakmags_r.append(min(mag[passband==2]))

        except Exception as e:
            print(e)
            continue

        flux = 10. ** (-0.4 * (mag - zeropoint))
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

        try:
            dummy = max(flux[passband == 'g'])
            dummy = max(flux[passband == 'r'])
        except Exception as e:
            print(e)
            continue

        peakfluxes_g.append(max(flux[passband == 'g']))
        peakfluxes_r.append(max(flux[passband == 'r']))

    # import sys
    # import pickle
    # with open('save_real_ZTF_unprocessed_data_snia_osc_12nov2019.npz', 'wb') as f:
    #     pickle.dump([mjds, passbands, mags, magerrs, photflags, zeropoints, ras, decs, objids, redshifts, mwebvs], f)
    # # np.savez('save_real_ZTF_unprocessed_data_snia_osc_12nov2019.npz', mjds=mjds, passbands=passbands, mags=mags, magerrs=magerrs, photflags=photflags, zeropoints=zeropoints, ras=ras, decs=decs, objids=objids, redshifts=redshifts, mwebvs=mwebvs)# , peakflux_g=peakfluxes_g, peakflux_r=peakfluxes_r)
    # print("finished")
    # # # sys.exit(0)
    # # with open('save_real_ZTF_unprocessed_data_snia_osc_12nov2019.npz', 'rb') as f:
    # #     a = pickle.load(f)

    classification = Classify(known_redshift=True, bcut=False, zcut=None)
    predictions, time_steps = classification.get_predictions(light_curve_list, return_predictions_at_obstime=False)
    print(predictions)

    if plot:
        # try:
        classification.plot_light_curves_and_classifications(step=True, use_interp_flux=False, figdir=figdir, plot_matrix_input=True)
        # except Exception as e:
        #     print(e)
        # classification.plot_light_curves_and_classifications(step=False, use_interp_flux=True)
        # classification.plot_light_curves_and_classifications(step=False, use_interp_flux=False)
        # classification.plot_classification_animation_step()
        # classification.plot_classification_animation()



    return classification.orig_lc, classification.timesX, classification.y_predict


if __name__ == '__main__':
    # classify_lasair_light_curves(object_name='ZTF19aabbnzo')
    # classify_lasair_light_curves(object_name='ZTF18abukavn')
    # classify_lasair_light_curves(object_name='ZTF18aajupnt') # 2018dyk
    # classify_lasair_light_curves(object_name='ZTF18aabtxvd') # 2018zr
    # classify_lasair_light_curves(object_name='ZTF18acpdvos') # 2018hyz

    # classify_lasair_light_curves(object_names=['ZTF19abraqpf', 'ZTF19abqstxq', ('ZTF19abpmetl', 0.068), 'ZTF19abgsssu', 'ZTF19abdooly'], figdir='astrorapid')
    # classify_lasair_light_curves(object_names=[('ZTF18aazblzy', 0.064), 'ZTF19aabyppp'], figdir='astrorapid')
    # classify_lasair_light_curves(object_names=[('ZTF19abzrhgq', 0.0151), ('ZTF19abuhlxk', 0.02), ('ZTF19abylxyt', 0.0533)], figdir='astrorapid') # (2019qiz TDE), (2019pdx SNIa), (2019qid SNIa)

#     classify_lasair_light_curves(object_names=(
# ('ZTF19abfvnns', 0.14),
# ('ZTF19abheamc', 0.037),
# ('ZTF19abpfsks', 0.1),
# ('ZTF19abuzinv', 0.02),
# ('ZTF19abxdnzt', 0.044),
# ('ZTF19abzkiuv', 0.059),
# ('ZTF19abzrhgq', 0.0151),
# ('ZTF19abzsitm', 0.046),
# ('ZTF19abzzhsa', 0.041),
# ('ZTF19acapeun', 0.019917),
# ('ZTF19acbacvu', 0.035),
# ('ZTF19acbiwjk', 0.09),
# ('ZTF19acbnhkd', 0.067),
# ('ZTF19acbswdq', 0.0667),
# ('ZTF19acbunmk', 0.028),
# ('ZTF19acbykmk', 0.061),
# ('ZTF19acchtyp', 0.06),
# ('ZTF19acdihrz', 0.066),
# ('ZTF19acdtpow', 0.075),
# ('ZTF19acdubmt', 0.092),
# ('ZTF19acekreh', 0.03480),
# ('ZTF19acetxvq', 0.063),
# ('ZTF19achagst', 0.068),
# ('ZTF19achejoc', 0.05)
# ))
    # classify_lasair_light_curves(object_names=('ZTF18abcrxoj', ('ZTF20aakyoez', 0.041), 'ZTF20aanvqbi',),)#'ZTF18acmxkzj', 'ZTF18adbntwo', 'ZTF18acmzpbf', 'ZTF18acrdmmw', 'ZTF18acsouxk', 'ZTF18aczrafz', 'ZTF18aceynvm', 'ZTF18acdzzyf', 'ZTF18acybdar', 'ZTF18ablhvfr', 'ZTF18acebefc', 'ZTF18acslpba', 'ZTF18aaxcxih', 'ZTF18aaxcntm', 'ZTF18abspqsn', 'ZTF18abvejqt','ZTF18adbntwo', 'ZTF18adasopt', 'ZTF18aajupnt', 'ZTF18aaxkqgy', 'ZTF18abckutn', 'ZTF18abrlljc', 'ZTF18acyybvg', 'ZTF19aadnmgf', 'ZTF18abxftqm',))

    # orig_lc, timesX, y_predict = classify_lasair_light_curves(object_names=[
    #                                                             ('ZTF19abzcaod', 0.052098),
    #                                                             'ZTF19abgjoth',
    #                                                             # 'ZTF19abhhjcc',
    #                                                             'ZTF19aazcxwk', 'ZTF19abauzma',
    #                                                             'ZTF18abxftqm',  # TDE
    #                                                             'ZTF19aadnmgf',  # SNIa
    #                                                             ('ZTF18acmzpbf', 0.036),  # SNIa
    #                                                             'ZTF19aakzwao',  # SNIa
    #                                                             ], figdir='astrorapid')


    # orig_lc, timesX, y_predict = classify_lasair_light_curves(object_names=[('ZTF19aarioci', 0.074), ('ZTF19aabbnzo', 0.08), ('ZTF19aabyppp', 0.037)], figdir='astrorapid') # ( TDE), ( SNIa), ( SNIa) ('ZTF19aatlmbo', 0.007755), ('ZTF19aabyppp', 0.037), ('ZTF19aabyppp', 0.08), ('ZTF19aadnxat', 0.03)
    # import pickle
    # with open('paper_plot_real_data_new2019.pickle', 'wb') as f:
    #     pickle.dump([orig_lc, timesX, y_predict], f)

    tde_candidates = [('ZTF19aapreis', 0.0512),
                      # 'ZTF18actaqdw',
                      ('ZTF18aahqkbt', 0.051),
                      ('ZTF18abxftqm', 0.09),
                      ('ZTF19aabbnzo', 0.08),
                      'ZTF18acpdvos',
                      'ZTF18aabtxvd',
                      ('ZTF19aarioci', 0.12)]
    classify_lasair_light_curves(object_names=tde_candidates, figdir='test_astrorapid')


    # # nuclear_transients =  ('ZTF19aalzmdv', ('ZTF19aapreis', 0.0512),'ZTF19aandznq', ('ZTF18abxftqm', 0.09), ('ZTF19aarioci', 0.12),'ZTF19aalveag', 'ZTF18aajgjow', 'ZTF19aaluprf', 'ZTF19aamrzte', 'ZTF18achxcsh', 'ZTF19aalzmio', 'ZTF19aamhhae', 'ZTF19aamkowf', 'ZTF18aaizdjz', 'ZTF19aamhhgu', 'ZTF19aamvkpf', 'ZTF19aamfxhg', ('ZTF19aabbnzo', 0.08), 'ZTF19aanekbm', 'ZTF19aaneseo', 'ZTF19aamkgiw', 'ZTF19aanevry', 'ZTF19aanfyey', 'ZTF19aamqlok', 'ZTF19aamquei', 'ZTF19aamszrg', 'ZTF19aanfkuf', 'ZTF19aamhbqx', 'ZTF19aanickg', 'ZTF19aanikcm', 'ZTF19aamqveb', 'ZTF19aaniiaz', 'ZTF19aalzmae', 'ZTF19aamvmer', 'ZTF19aaniibk', 'ZTF19aamkmxv', 'ZTF19aamuxso', 'ZTF19aamvavq', 'ZTF19aanbojt', 'ZTF19aanflwj', 'ZTF19aangfld', 'ZTF19aamtwwv', 'ZTF19aanlews', 'ZTF19aanuuzg', 'ZTF19aamtjzm', 'ZTF19aamkywr', 'ZTF19aangzrq', 'ZTF19aaomlmm', 'ZTF19aanuuen', 'ZTF19aanlvnl', 'ZTF19aanfrmq', 'ZTF19aanetjx', 'ZTF19aaoxijx', 'ZTF19aanetcr', 'ZTF19aaneugz', 'ZTF18aahtjsc', 'ZTF19aaohwqp', 'ZTF19aanhfnk', 'ZTF19aaogtqk', 'ZTF19aaozmrs', 'ZTF19aaoznau', 'ZTF19aaojnrx', 'ZTF19aamqjqc', 'ZTF19aaokcsn', 'ZTF19aalarng', 'ZTF18aailsok', 'ZTF19aaocnda', 'ZTF19aapafew', 'ZTF19aaondau', 'ZTF18aaivonj', 'ZTF19aanxosu', 'ZTF19aantojb', 'ZTF19aamctcx', 'ZTF19aapadsm', 'ZTF19aaoubqg', 'ZTF19aaohaau', 'ZTF19aaoibgs', 'ZTF19aaohasv', 'ZTF19aaohbqk', 'ZTF19aanevcu', 'ZTF19aapfpnz', 'ZTF19aampaby', 'ZTF19aapkjmd', 'ZTF18acnbexb', 'ZTF19aamkrcp', 'ZTF19aaoznax', 'ZTF19aaprgvu', 'ZTF19aanhcyf', 'ZTF19aaoxzej', 'ZTF19aaneuil', 'ZTF19aaoyinh', 'ZTF19aapdbau', 'ZTF19aanxnkd', 'ZTF19aaoigdw', 'ZTF19aaomnkb', 'ZTF19aaopqig', 'ZTF19aapamsd', 'ZTF18abfcmjw', 'ZTF19aanxrkt', 'ZTF19aadopob', 'ZTF19aaoakjo', 'ZTF19aanfzox', 'ZTF19aapsxfl', 'ZTF19aapawxp', 'ZTF19aaptarf', 'ZTF19aapswkx', 'ZTF19aapbfot', 'ZTF19aapavln', 'ZTF19aapbbso', 'ZTF19aaqrlmz', 'ZTF18abldizg', 'ZTF19aapvltt', 'ZTF19aapxlhm',  'ZTF19aarhrru', 'ZTF19aarhrxs', 'ZTF19aapctvv', 'ZTF18aasxzyb', 'ZTF18aawwmga', 'ZTF18aalxmmk', 'ZTF19aaoycfk', 'ZTF19aarifvy', 'ZTF19aaraxhh', 'ZTF19aaqdfwd', 'ZTF19aanjryl', 'ZTF19aarfksi', 'ZTF19aariuqe', 'ZTF19aarjbyj', 'ZTF19aariwpf', 'ZTF19aapusqe', 'ZTF19aarinmw', 'ZTF19aapwoid', 'ZTF18abddtca', 'ZTF19aaricev', 'ZTF19aaqtcmy', 'ZTF19aarkuwd', 'ZTF19aarflsx', 'ZTF19aaqgxpy', 'ZTF19aariudl', 'ZTF19aaqhelq', 'ZTF19aarnrfo', 'ZTF19aarnmqs', 'ZTF18aamuasc', 'ZTF19aaqwwux', 'ZTF19aaqwwvh', 'ZTF19aarhphl', 'ZTF19aaqwtbg', 'ZTF19aaoxlzb', 'ZTF19aaojljx', 'ZTF19aarinqi', 'ZTF19aarmzoi', 'ZTF19aarhyjx', 'ZTF19aarisyg', 'ZTF19aarisna', 'ZTF19aasbphu', 'ZTF19aarhlrp', 'ZTF19aarrqat', 'ZTF19aarrufq', 'ZTF18aceqtdz', 'ZTF18abjjgiz', 'ZTF18aawodxu', 'ZTF19aasensj', 'ZTF18aanxnrz', 'ZTF19aatfkmb', 'ZTF19aarnhuv', 'ZTF19aatgrxv', 'ZTF19aarrcmx', 'ZTF19aarqpeb', 'ZTF19aatjrua', 'ZTF19aarsact', 'ZTF18aaxokeg', 'ZTF19aariuvr', 'ZTF18aaxisdq', 'ZTF19aasejuy', 'ZTF19aatgrxl', 'ZTF19aatfkms', 'ZTF19aasbqaq', 'ZTF19aarfreu', 'ZTF19aascqge', 'ZTF19aascpnd', 'ZTF19aarisvo', 'ZTF18aaqkdwu', 'ZTF19aarbbfu', 'ZTF19aaqesdr', 'ZTF19aatqugy', 'ZTF19aaprehw', 'ZTF19aaafyjr', 'ZTF19aaqtxaf', 'ZTF19aarqihw', 'ZTF19aarfequ', 'ZTF19aatgaiq', 'ZTF19aarspxq', 'ZTF19aaskaxz', 'ZTF19aatgopg', 'ZTF19aathate', 'ZTF19aarhpgn', 'ZTF19aatjfny', 'ZTF19aasemsm', 'ZTF19aatgshx', 'ZTF19aarsyyk', 'ZTF19aaugngt', 'ZTF19aaufzcj', 'ZTF19aathabc', 'ZTF19aarioel', 'ZTF19aaujiwc', 'ZTF19aauishy', 'ZTF18aasympo', 'ZTF19aatnfyv', 'ZTF19aatnfvj', 'ZTF19aaqyovr', 'ZTF19aarycuy', 'ZTF19aaryace', 'ZTF19aauitks', 'ZTF19aauisdr', 'ZTF19aaptbsf', 'ZTF18abhjwvl', 'ZTF19aauplxw', 'ZTF19aaujlou', 'ZTF19aaupklw', 'ZTF19aauhhjy', 'ZTF19aasjatz', 'ZTF19aaujxcd', 'ZTF19aauvjws', 'ZTF19aaumrok', 'ZTF19aaupkrl', 'ZTF19aaulrch', 'ZTF18aaaqgyc', 'ZTF19aavkwcg', 'ZTF19aavkvtm', 'ZTF19aavkwmx', 'ZTF19aavkxgt', 'ZTF19aavledx', 'ZTF19aavldty', 'ZTF19aavlgdy', 'ZTF19aavhyzx', 'ZTF18aaqkcso', 'ZTF19aavjfha', 'ZTF19aavhblr', 'ZTF19aavmlqh', 'ZTF19aatubsj', 'ZTF19aavnotz', 'ZTF19aavlgjq', 'ZTF19aavocxo', 'ZTF19aavlghe', 'ZTF19aavhymd', 'ZTF19aavipfk', 'ZTF19aavoqxg', 'ZTF19aavoudg', 'ZTF18abetcyu', 'ZTF19aavlhmy', 'ZTF19aavnven', 'ZTF19aavgyjd', 'ZTF18aaqcdiv', 'ZTF19aavkuig', 'ZTF19aavnwtu', 'ZTF19aauivtj', 'ZTF19aavrsae', 'ZTF19aarnreg', 'ZTF19aavlnqp', 'ZTF19aavsahh', 'ZTF19aavrthg', 'ZTF19aavhwxs', 'ZTF19aavlnik', 'ZTF18aboczgw', 'ZTF19aavvdfi', 'ZTF19aaqtpet', 'ZTF19aavwexx', 'ZTF19aavqotz', 'ZTF19aavvzwq', 'ZTF19aavwbpc', 'ZTF19aavwcow', 'ZTF19aavweqz', 'ZTF19aavxfib', 'ZTF19aavnzti', 'ZTF18aarlukc', 'ZTF19aauvlhw', 'ZTF19aavxrzc', 'ZTF19aavlesn', 'ZTF19aavxjcr', 'ZTF19aapfoax', 'ZTF19aauqywe', 'ZTF19aawbgdh', 'ZTF19aavanad', 'ZTF19aavpzke', 'ZTF19aavaltf', 'ZTF19aawgckg', 'ZTF19aavnxwj', 'ZTF19aawgfnn', 'ZTF19aavocqa', 'ZTF19aavlfvn', 'ZTF19aavovik', 'ZTF19aavovkq', 'ZTF18aavuziy', 'ZTF19aavrxyy', 'ZTF19aavowpa', 'ZTF19aattnch', 'ZTF18aayooza', 'ZTF18accepvi', 'ZTF19aaudams', 'ZTF19aawlluc', 'ZTF19aavouav', 'ZTF19aawnguu', 'ZTF18aaqpouu', 'ZTF19aatylnl', 'ZTF19aavnwoh', 'ZTF18aahxdfs', 'ZTF19aavszux', 'ZTF19aavsqjw', 'ZTF19aavspux', 'ZTF19aawnqoj', 'ZTF19aavsahi', 'ZTF19aaqapij', 'ZTF19aawscqd', 'ZTF19aavyvkz', 'ZTF19aavnxgx', 'ZTF19aavqoop', 'ZTF19aavwdwy', 'ZTF19aavwarp', 'ZTF19aavzlin', 'ZTF19aavnurw', 'ZTF18aceqkmg', 'ZTF19aavrngz', 'ZTF19aabzzvo', 'ZTF19aapzmzm', 'ZTF19aadhdzx', 'ZTF19aavajrd', 'ZTF19aawfanq', 'ZTF19aaweqnm', 'ZTF19aawfwwv', 'ZTF19aaxbxui', 'ZTF19aawhagd', 'ZTF19aatjgrc', 'ZTF19aavwgol', 'ZTF19aawwlxc', 'ZTF19aavoayk', 'ZTF19aaxffum', 'ZTF18acugblr', 'ZTF19aavnxna', 'ZTF19aavlhjo', 'ZTF19aasbxon', 'ZTF19aawsyhf', 'ZTF19aavnnzk', 'ZTF19aawllso', 'ZTF19aaxibpz', 'ZTF19aaujlpo', 'ZTF19aavnvnx', 'ZTF19aavqiti', 'ZTF19aaxpjpq', 'ZTF19aatvmcr', 'ZTF19aaprwon', 'ZTF19aawmhbu', 'ZTF19aawyepw', 'ZTF18aasvfah', 'ZTF19aawafvn', 'ZTF19aawlgne', 'ZTF19aawupxb', 'ZTF19aavqhfi', 'ZTF19aavrnol', 'ZTF19aardooq', 'ZTF18accvhmi', 'ZTF19aawgqcc', 'ZTF19aayksry', 'ZTF19aavnxfh', 'ZTF19aawyuxv', 'ZTF19aaxdafs', 'ZTF19aauctnx', 'ZTF18acrdlrp', 'ZTF18aaikyip', 'ZTF18acqsqry', 'ZTF18acrukmr', 'ZTF18acrukrr', 'ZTF18acrvcqx', 'ZTF17aaaoqmq', 'ZTF18acrxfwp', 'ZTF18acrxiod', 'ZTF18aaacnpb', 'ZTF18aazwtfj', 'ZTF18acrtykk', 'ZTF18acsjjqr', 'ZTF18acapgdp', 'ZTF18acrfvuf', 'ZTF18acrctos', 'ZTF18acsxsmi', 'ZTF18acrciyr', 'ZTF18acrcizr', 'ZTF18acrvakb', 'ZTF17aabvtgd', 'ZTF18acrultm', 'ZTF18acsyzla', 'ZTF18acsxrge', 'ZTF18acrukpd', 'ZTF18acenyfr', 'ZTF18acrlmvt', 'ZTF18acujaic', 'ZTF18acurbia', 'ZTF18acuring', 'ZTF18acueemg', 'ZTF18acrdtzn', 'ZTF18acuqyyn', 'ZTF18acuqxyf', 'ZTF18acuqskr', 'ZTF18acuschq', 'ZTF18acszxqc', 'ZTF18acsouto', 'ZTF18aajhfdm', 'ZTF18actxhgn', 'ZTF18acrwgfp', 'ZTF18acvddao', 'ZTF18acubbwb', 'ZTF18acrvldh', 'ZTF18acurwua', 'ZTF18acrgunw', 'ZTF18aawxcbp', 'ZTF18aawmxdx', 'ZTF18acurljz', 'ZTF18acrxgbu', 'ZTF18acsowoc', 'ZTF18acueeoo', 'ZTF18acrdtwo', 'ZTF18acslbve', 'ZTF18acvvudh', 'ZTF18acvwchw', 'ZTF18acvwcky', 'ZTF18aaiuynw', 'ZTF18acvwcms', 'ZTF18acvwddd', 'ZTF18acvvved', 'ZTF18acsyanq', 'ZTF18acrexje', 'ZTF18acwhaxj', 'ZTF18acszixl', 'ZTF18acvvuzi', 'ZTF18acvwgqy', 'ZTF18acvwgdu', 'ZTF18acvwbos', 'ZTF17aaanppf', 'ZTF18acubbuv', 'ZTF18acwyvhw', 'ZTF18acuewyy', 'ZTF18aaquudq', 'ZTF18acxgujm', 'ZTF18acetnxh', 'ZTF18acszgpa', 'ZTF17aabwgbf', 'ZTF18aawonvg', 'ZTF18acynmcp', 'ZTF18acsxvpj', 'ZTF18aczawcw', 'ZTF18acxijqw', 'ZTF18actzpxw', 'ZTF18acrunrq', 'ZTF18acxyarg', 'ZTF18acvwhaz', 'ZTF18aczcsmg', 'ZTF18acvgugm', 'ZTF18acvgzab', 'ZTF18aczeraw', 'ZTF18aczerlj', 'ZTF18acuswcl', 'ZTF18acrxtrv', 'ZTF18acynlmf', 'ZTF18acwugwx', 'ZTF18acybdar', 'ZTF18acvgxmp', 'ZTF18acwtykp', 'ZTF18acvgwbt', 'ZTF18aczeqva', 'ZTF18achyngk', 'ZTF18acvdstg', 'ZTF18adaysgv', 'ZTF18acrvdsj', 'ZTF18aaacbjk', 'ZTF18acurmhj', 'ZTF18acvggao', 'ZTF18acywbgz', 'ZTF18adaylyq', 'ZTF18acyxxen', 'ZTF18adalarn', 'ZTF18adaliyg', 'ZTF18acrhegn', 'ZTF18acwzecq', 'ZTF18adalglq', 'ZTF18adaimlf', 'ZTF18adbywzj', 'ZTF18aczenvx', 'ZTF18adazbol', 'ZTF18adbifqw', 'ZTF18aabeyio', 'ZTF18aabgazt', 'ZTF18acwtyfp', 'ZTF18adbmkvr', 'ZTF18aczpdvz', 'ZTF18adcbymm', 'ZTF18adcbytv', 'ZTF18adbdhni', 'ZTF18adbdhtn', 'ZTF18acvrgwv', 'ZTF18acvqsht', 'ZTF18adcedvg', 'ZTF18adaymez', 'ZTF18aczevei', 'ZTF19aaacubf', 'ZTF19aaaegdw', 'ZTF18acuspbi', 'ZTF18adazigr', 'ZTF18adaktri', 'ZTF18acvgqsl', 'ZTF18adbabqs', 'ZTF18aawngqo', 'ZTF19aaafgdr', 'ZTF18adbabqv', 'ZTF18acsowfw', 'ZTF18acyxnyw', 'ZTF18acqssac', 'ZTF19aaafhyc', 'ZTF18aajkgtr', 'ZTF19aaafhdx', 'ZTF19aaafzah', 'ZTF18adaadmh', 'ZTF18adbmiyb', 'ZTF18adbmbgf', 'ZTF18acsxkme', 'ZTF18adbcjkb', 'ZTF18abwtbha', 'ZTF18actadvo', 'ZTF18adasjei', 'ZTF18acytbjt', 'ZTF18acxhvzb', 'ZTF18adbcjin', 'ZTF18adamhjn', 'ZTF18aczdetq', 'ZTF18acvvpov', 'ZTF18acurjjx', 'ZTF18abgkjpd', 'ZTF18acrnhtg', 'ZTF18abedvgv', 'ZTF18adbabzm', 'ZTF18adbacwh', 'ZTF18acueykv', 'ZTF18acnoktl', 'ZTF19aaapnxn', 'ZTF18acyendr', 'ZTF18aavjqvm', 'ZTF18adbadid', 'ZTF18adalfyk', 'ZTF19aaaanri', 'ZTF18abxrhmi', 'ZTF19aaahrff', 'ZTF19aaahryn', 'ZTF19aaaaxeu', 'ZTF18acvdcyf', 'ZTF18acrttfl', 'ZTF18acsybmh', 'ZTF19aaahrno', 'ZTF18actafdw', 'ZTF19aaahsjp', 'ZTF18acukgkk', 'ZTF18adbmmci', 'ZTF18aczwdth', 'ZTF19aaaokub', 'ZTF19aaafldf', 'ZTF19aaafnhx', 'ZTF19aabyyhr', 'ZTF19aaafice', 'ZTF19aaaplqp', 'ZTF19aacivvy', 'ZTF18acwzawa', 'ZTF18acvqkxx', 'ZTF18adbdhwh', 'ZTF18acsxssp', 'ZTF19aacosqh', 'ZTF18actzyzs', 'ZTF19aaabmhd', 'ZTF19aacolgi', 'ZTF18aczawvm', 'ZTF19aaaanyf', 'ZTF18acsyfnk', 'ZTF19aabgefm', 'ZTF19aacqeai', 'ZTF19aabhfdi', 'ZTF18abzbprc', 'ZTF19aaaolce', 'ZTF18acvqugw', 'ZTF18acsmisj', 'ZTF18adazgdh', 'ZTF18aarbklo', 'ZTF18aajgowk', 'ZTF18acurklz', 'ZTF19aabyvwv', 'ZTF19aaczyek', 'ZTF19aabyppp', 'ZTF19aaahjhc', 'ZTF18adbcjdl', 'ZTF18aawohdr', 'ZTF19aadfxra', 'ZTF18acuepcb', 'ZTF19aaafoaf', 'ZTF19aabzgoc', 'ZTF19aaapvoy', 'ZTF18aahegog', 'ZTF19aadoevn', 'ZTF19aacxvyw', 'ZTF19aacivld', 'ZTF19aadyiao', 'ZTF19aacwqjs', 'ZTF19aadomgl', 'ZTF19aaeoqkk', 'ZTF18aacdbzx', 'ZTF19aadgmuq', 'ZTF19aadnxkc', 'ZTF19aaapjps', 'ZTF18acqighv', 'ZTF19aaesfkm', 'ZTF19aaflhbi', 'ZTF18aaisvqg', 'ZTF18aaisjkm', 'ZTF19aaeomsn', 'ZTF19aadgceq', 'ZTF19aaeopbq', 'ZTF19aadfumv', 'ZTF19aaciwam', 'ZTF19aadohat', 'ZTF19aaejtrn', 'ZTF18abcgryu', 'ZTF19aafmidf', 'ZTF19aacsojb', 'ZTF18acvghyv', 'ZTF18aaofdgl', 'ZTF19aailmac', 'ZTF19aailsad', 'ZTF19aailvhf', 'ZTF19aailvdf', 'ZTF18acvdwhd', 'ZTF19aaeraih', 'ZTF19aagnprs', 'ZTF19aajwkbs', 'ZTF19aaaemtp', 'ZTF19aacsofi', 'ZTF19aaikxuj', 'ZTF19aailpww', 'ZTF19aailqpe', 'ZTF19aaaplph', 'ZTF19aailtfn', 'ZTF19aafmyow', 'ZTF19aailtfi', 'ZTF19aaapljd', 'ZTF18adasxhd', 'ZTF19aakljys', 'ZTF19aaeuhgo', 'ZTF19aakswrb', 'ZTF18abvzqwo', 'ZTF19aakykjl', 'ZTF19aalahqa', 'ZTF19aajykpl', 'ZTF18aceotqg', 'ZTF18abixrei', 'ZTF19aaknvlv', 'ZTF19aaknvuh', 'ZTF19aambixl', 'ZTF19aaktpns', 'ZTF19aailrlv', 'ZTF19aalubnr', 'ZTF18aavrmcg', 'ZTF18aayxupv', 'ZTF18aatybbp', 'ZTF18aaziklu', 'ZTF17aaarqox', 'ZTF18aazlsvk', 'ZTF18aajumjz', 'ZTF17aaawcwx', 'ZTF18aazwdak', 'ZTF17aabumwe', 'ZTF18abaphcx', 'ZTF17aaaedvh', 'ZTF18aagvqkp', 'ZTF18aanmmeg', 'ZTF18abbkixx', 'ZTF17aaaeuus', 'ZTF18abbodly', 'ZTF18aarltqh', 'ZTF18abaqvko', 'ZTF18abaambh', 'ZTF18abbmbys', 'ZTF18abcbwnd', 'ZTF18abajspl', 'ZTF18aazhwnh', 'ZTF18abbfwtt', 'ZTF18abcezmh', 'ZTF18abcipbf', 'ZTF18abbowqa', 'ZTF18abbuxcm', 'ZTF18abashqj', 'ZTF18abcreta', 'ZTF18abchhcn', 'ZTF18aazziao', 'ZTF18aazzfhx', 'ZTF18abcptmt', 'ZTF18abcrasg', 'ZTF18abashqr', 'ZTF18abblwsh', 'ZTF18abcdzyc', 'ZTF18abcrzfz', 'ZTF18abcvush', 'ZTF18abawrxq', 'ZTF18abccpuw', 'ZTF18abdbuty', 'ZTF18abcxzqk', 'ZTF18aaseyca', 'ZTF18aansqun', 'ZTF18aajswer', 'ZTF18abcvufc', 'ZTF18abbvbru', 'ZTF18abalzzd', 'ZTF18abditme', 'ZTF18abcyawn', 'ZTF18abguhwj', 'ZTF18abehceu', 'ZTF18abfzkno', 'ZTF18abglpdy', 'ZTF18abgisys', 'ZTF18abcbvxc', 'ZTF18abespgb', 'ZTF18abeakbs', 'ZTF18aawurud', 'ZTF18abhhnnv', 'ZTF18abcfcka', 'ZTF18abdkini', 'ZTF18abgulkc', 'ZTF18abhqdpv', 'ZTF18abetewu', 'ZTF18abaqetc', 'ZTF18abflqxq', 'ZTF18abgitic', 'ZTF17aabvfha', 'ZTF18aapaohn', 'ZTF18abkjfzt', 'ZTF18abkjfzw', 'ZTF18abkhcrj', 'ZTF18aaofsmp', 'ZTF18abhakmn', 'ZTF18abjbnes', 'ZTF18aaqkdwf', 'ZTF18abjstcm', 'ZTF18ablhplf', 'ZTF18aaoxryq', 'ZTF18abjndls', 'ZTF18abfzixe', 'ZTF18abjvgwv', 'ZTF18abdzvgz', 'ZTF18abiirfq', 'ZTF18abklbgb', 'ZTF18abeibdj', 'ZTF18ablowct', 'ZTF18abjwahi', 'ZTF18ablqfmf', 'ZTF18abeajml', 'ZTF18ablqlzp', 'ZTF18ablqqeb', 'ZTF18abkifng', 'ZTF18ablrlbm', 'ZTF18abcpdns', 'ZTF18ablsxjo', 'ZTF18abkhcax', 'ZTF18abkhbrs', 'ZTF18abcfdzu', 'ZTF18ablongw', 'ZTF18abjhbss', 'ZTF18abelhrm', 'ZTF18abeitow', 'ZTF18ablxdqz', 'ZTF18abfxhrt', 'ZTF18abgqvwv', 'ZTF18abgubgi', 'ZTF18abmjgyk', 'ZTF18abmjujg', 'ZTF18abjyjdz', 'ZTF18ablrljh', 'ZTF18abmbiza', 'ZTF18aamvand', 'ZTF18abmnwvc', 'ZTF18ablwbqn', 'ZTF17aaartnt', 'ZTF18ablszje', 'ZTF18aabesgz', 'ZTF18abmxfrc', 'ZTF18abmmtts', 'ZTF17aabvcvr', 'ZTF18ablzvih', 'ZTF18ablsypo', 'ZTF18abbydbi', 'ZTF18abmrhom', 'ZTF18abltbrd', 'ZTF18aanxtko', 'ZTF18abnnnzk', 'ZTF18abncizy', 'ZTF18abmdvcj', 'ZTF18abctyvq', 'ZTF18ablvjom', 'ZTF18abnchro', 'ZTF18aabqtsw', 'ZTF18aboemit', 'ZTF18abnugci', 'ZTF18abntxck', 'ZTF18abmogca', 'ZTF18abmwxvv', 'ZTF18abobkii', 'ZTF18abiitmq', 'ZTF18abahgfm', 'ZTF18aboaeqy', 'ZTF18abfgabf', 'ZTF18abflcof', 'ZTF18abgugua', 'ZTF18abjgzxw', 'ZTF18abhozbd', 'ZTF18abeaymc', 'ZTF18abnuhuy', 'ZTF18abptsco', 'ZTF18abpttky', 'ZTF18abqjvyl', 'ZTF18abmkbqk', 'ZTF18abrzhik', 'ZTF18abnbhpu', 'ZTF18abccnyj', 'ZTF18abcctas', 'ZTF17aaazypn', 'ZTF18abncimo', 'ZTF18aabfmux', 'ZTF18absdeij', 'ZTF18abmwzpt', 'ZTF18aaymqbw', 'ZTF18ablmhjv', 'ZTF18abrlurr', 'ZTF18aarrelk', 'ZTF18ablrlmu', 'ZTF18abimjrs', 'ZTF18aamfybu', 'ZTF18abctdgj', 'ZTF18abokvkt', 'ZTF18abnygkb', 'ZTF18abmnvjc', 'ZTF18abrymgj', 'ZTF18aaakpjg', 'ZTF18abmddcb', 'ZTF18ablwzih', 'ZTF18abnvnqb', 'ZTF18abeachz', 'ZTF18abkvwzj', 'ZTF18aapgbfm', 'ZTF18ablpnuq', 'ZTF18abihxip', 'ZTF18aauherh', 'ZTF18abdlfev', 'ZTF18absyyig', 'ZTF18abdazfd', 'ZTF18absgnio', 'ZTF18abtdxke', 'ZTF18abteera', 'ZTF18abotcjv', 'ZTF18abotbuq', 'ZTF18abhnscs', 'ZTF18abtjlsr', 'ZTF18abtjmns', 'ZTF18abnufuv', 'ZTF18abjmlty', 'ZTF18abbuxbk', 'ZTF18abryqnn', 'ZTF17aabvele', 'ZTF18absljwl', 'ZTF18abtnfnq', 'ZTF18abhowyg', 'ZTF18abscghc', 'ZTF18absqitc', 'ZTF18absqkfg', 'ZTF18abscyhs', 'ZTF18abnvklh', 'ZTF18abmqwgr', 'ZTF18abqblsk', 'ZTF18abshnwm', 'ZTF18abrzbnb', 'ZTF18abrzcdi', 'ZTF18abrqedj', 'ZTF18absrcps', 'ZTF18abslxpd', 'ZTF18abcwjii', 'ZTF18abuatdz', 'ZTF18abjlxwv', 'ZTF18abufdlx', 'ZTF18abskhcp', 'ZTF18abtsobl', 'ZTF18abtmbaz', 'ZTF18abtswjk', 'ZTF18abrzbuj', 'ZTF18aaabzcd', 'ZTF18absmqmw', 'ZTF18abeyoaa', 'ZTF18abdmghg', 'ZTF18abcwzhs', 'ZTF18abgxnrw', 'ZTF18abcgjpn', 'ZTF18abnwvmh', 'ZTF18abijktl', 'ZTF18abhpyor', 'ZTF18aborolm', 'ZTF18abswypg', 'ZTF18abttaxw', 'ZTF18absbyio', 'ZTF18abotfiv', 'ZTF18abbqjbb', 'ZTF18absnayk', 'ZTF18abtnwpa', 'ZTF18abuqugw', 'ZTF18abthuby', 'ZTF18abukavn', 'ZTF18abtvinn', 'ZTF18abuwxii', 'ZTF18abnzzvq', 'ZTF18abuatqi', 'ZTF18abnygwk', 'ZTF18abbuutk', 'ZTF18abufgdj', 'ZTF18abtswll', 'ZTF18abuhyjv', 'ZTF18abvgkcg', 'ZTF18aboztku', 'ZTF18aajsypz', 'ZTF18absnqyo', 'ZTF18abvivzm', 'ZTF18abugmrg', 'ZTF18abtphnh', 'ZTF17aaafrch', 'ZTF18abuhzfc', 'ZTF18abtfwhe', 'ZTF18abmjuya', 'ZTF18abeainm', 'ZTF18abtxynh', 'ZTF18abvfisg', 'ZTF18abvujhp', 'ZTF18abbmenz', 'ZTF18abdebec', 'ZTF18abvwsjn', 'ZTF18abtjrbt', 'ZTF18abskrcm', 'ZTF18aajclem', 'ZTF18abumdjo', 'ZTF18abnzksu', 'ZTF18abvfafa', 'ZTF18abwgaek', 'ZTF18abvzdvj', 'ZTF18abwkans', 'ZTF18abuqhje', 'ZTF18abusyhc', 'ZTF17aabptoh', 'ZTF18absggvu', 'ZTF18abuuqkb', 'ZTF18abuvrnw', 'ZTF18abuxjrd', 'ZTF18abunhyu', 'ZTF18abwkhky', 'ZTF18abuahnw', 'ZTF18abtzzlv', 'ZTF18abvcehx', 'ZTF18abvptts', 'ZTF18abvtbow', 'ZTF18abtnggb', 'ZTF18aazttzj', 'ZTF18abtogdl', 'ZTF18ablprcf', 'ZTF18aburkuo', 'ZTF18abxdtyz', 'ZTF18abehvba', 'ZTF18abslbyb', 'ZTF18abwkrvs', 'ZTF18abmwwni', 'ZTF17aacjaxh', 'ZTF18abvpnnb', 'ZTF18abqwixr', 'ZTF18abvrvdk', 'ZTF18abqaujs', 'ZTF17aabvwlq', 'ZTF18abltbxa', 'ZTF18abxyvxq', 'ZTF18abtlwoj', 'ZTF18abwfktb', 'ZTF18abuiytq', 'ZTF18abqyvzy', 'ZTF18aabgboh', 'ZTF17aabsopd', 'ZTF17aaatdgc', 'ZTF17aaaabds', 'ZTF18abijaxd', 'ZTF18aaeyilr', 'ZTF18abuqoii', 'ZTF18abrwpvo', 'ZTF18abwlupf', 'ZTF18abolvzu', 'ZTF18absnxdb', 'ZTF17aaaeizh', 'ZTF18abvefxs', 'ZTF18abyfcns', 'ZTF18abyfvfr', 'ZTF18abchrps', 'ZTF18abufhtw', 'ZTF18acafztq', 'ZTF18abykeda', 'ZTF18abuifwq', 'ZTF18abtrlla', 'ZTF17aacxpvk', 'ZTF18acaksoq', 'ZTF18abxsmlf', 'ZTF18ablvxdg', 'ZTF18acaubtl', 'ZTF17aaaadiz', 'ZTF18acbvkwl', 'ZTF18acbvifx', 'ZTF18aadrhsi', 'ZTF18acbvzta', 'ZTF18acbwdwg', 'ZTF18abuszlm', 'ZTF18abmfwbn', 'ZTF18acbwhns', 'ZTF18acbwppb', 'ZTF18acbwwpm', 'ZTF18acbwxcc', 'ZTF18acbwxou', 'ZTF18abmefij', 'ZTF18acbxhua', 'ZTF18acbxkop', 'ZTF18abviyvv', 'ZTF18acbxsge', 'ZTF18abtlxdk', 'ZTF18abxmbjy', 'ZTF18abtbxou', 'ZTF18acbvmwr', 'ZTF17aadkbbu', 'ZTF18aaadyuc', 'ZTF18acbzqyi', 'ZTF18aaasgau', 'ZTF18abyfbhl', 'ZTF18acbwfvt', 'ZTF18aaemivw', 'ZTF18abwwdaz', 'ZTF18accdsxf', 'ZTF18accfxnn', 'ZTF18abtpite', 'ZTF18abmenfr', 'ZTF18acchzyo', 'ZTF18abvmkfh', 'ZTF18acckpro', 'ZTF18acbvbox', 'ZTF18abtswhs', 'ZTF18acbuibx', 'ZTF18acbvijf', 'ZTF18acclexy', 'ZTF18accpjff', 'ZTF18acbwopo', 'ZTF18acauwik', 'ZTF18acbyyff', 'ZTF18abwjzxr', 'ZTF18abmolpg', 'ZTF18acbvmzg', 'ZTF18aadkjmo', 'ZTF18abztrfp', 'ZTF18acbzpmu', 'ZTF18aaqkjuw', 'ZTF18aakeywu', 'ZTF18acbznkf', 'ZTF18aalpega', 'ZTF18acbzojv', 'ZTF18acbzpgy', 'ZTF18acdvvhi', 'ZTF18acdways', 'ZTF18abasoah', 'ZTF18abzscns', 'ZTF18absqpbs', 'ZTF18accjecm', 'ZTF18acchhpe', 'ZTF18accicnc', 'ZTF18accjuwd', 'ZTF18acectzf', 'ZTF18accnnyu', 'ZTF18accnbgw', 'ZTF18acbznpp', 'ZTF18acbwfim', 'ZTF18abkwfxq', 'ZTF18abujubn', 'ZTF18aceijsp', 'ZTF18abgjezf', 'ZTF18acbxoie', 'ZTF18absgmhg', 'ZTF18acchlkx', 'ZTF18accvkpt', 'ZTF18accwihs', 'ZTF18acbzoxp', 'ZTF18accfjmp', 'ZTF18abudvzn', 'ZTF18abtjmfm', 'ZTF18abcplgw', 'ZTF17aaakeap', 'ZTF18accvflv', 'ZTF18aafxtrt', 'ZTF18accnlji', 'ZTF18accdtch', 'ZTF18abxjekh', 'ZTF18aceyzwi', 'ZTF18acfejol', 'ZTF18abxzsnv', 'ZTF17aaaruzy', 'ZTF18aaagyuv', 'ZTF18aaapvkn', 'ZTF18accpjkb', 'ZTF18abmrhkl', 'ZTF18acbwtng', 'ZTF18abupgps', 'ZTF18achavgj', 'ZTF18acbvmjq', 'ZTF17aacxkaq', 'ZTF18aceolcv', 'ZTF18acbzvqn', 'ZTF18aamsgjq', 'ZTF18acbzwna', 'ZTF18acbxnpg', 'ZTF18abqavrw', 'ZTF18abnczui', 'ZTF18acfwmqj', 'ZTF18abnzunr', 'ZTF17aacfeuw', 'ZTF17aaayvyb', 'ZTF18acmrkpm', 'ZTF18accsane', 'ZTF18acchhxu', 'ZTF18abtyrmh', 'ZTF18abcpddh', 'ZTF18abnzuqm', 'ZTF18aceaape', 'ZTF18acbudwt', 'ZTF18acdvahc', 'ZTF18acewwqf', 'ZTF18acbwdxy', 'ZTF18abttklf', 'ZTF18aavrleg', 'ZTF18acegbeo', 'ZTF18abxbnvq', 'ZTF18aawmppg', 'ZTF18acnbptb', 'ZTF18aamrkmd', 'ZTF18achdnip', 'ZTF18achdidy', 'ZTF18achdfqm', 'ZTF18acnngyx', 'ZTF18acpeesd', 'ZTF18aceyqpm', 'ZTF18acnbglg', 'ZTF18acdvqhn', 'ZTF18aceykmc', 'ZTF18acbzvzm', 'ZTF18acnbejq', 'ZTF18aathofv', 'ZTF18acppzle', 'ZTF18ackvfyc', 'ZTF18abbuzeu', 'ZTF18abnuwsj', 'ZTF18aaadsrj', 'ZTF18acefgee', 'ZTF18accvmgs', 'ZTF18acnbgpw', 'ZTF18aadezmv', 'ZTF18acnbgvd', 'ZTF18aabklju', 'ZTF18acpuici', 'ZTF18abwskcn', 'ZTF18cnnfup')
    # nuclear_transients_080819 = np.unique(('ZTF19aarioci', 'ZTF19aabbnzo', 'ZTF18abxftqm', 'ZTF19aapreis', 'ZTF19abfdupx', 'ZTF19abfjjwc', 'ZTF19abfjjwc', 'ZTF19abflufo', 'ZTF19abfdupx', 'ZTF19abfqmqz', 'ZTF19abfrydu', 'ZTF19abfvhlx', 'ZTF19abgbdcp', 'ZTF19abgbdcp', 'ZTF19abfdupx', 'ZTF19abflufo', 'ZTF19abfdupx', 'ZTF19abfqmqz', 'ZTF19abfqmqz', 'ZTF19abezcns', 'ZTF19abggmrt', 'ZTF19abgbtla', 'ZTF19abgcbey', 'ZTF19abfvhlx', 'ZTF19abgjlef', 'ZTF19abfvhlx', 'ZTF19abfwfiq', 'ZTF19abfwhja', 'ZTF19abglmpf', 'ZTF19abgbdcp', 'ZTF19abglmpf', 'ZTF19abfjjwc', 'ZTF19abgncfz', 'ZTF19abgffaj', 'ZTF19abipkyb', 'ZTF19abfrydu', 'ZTF19abgcbey', 'ZTF19abfrydu', 'ZTF19abgpnge', 'ZTF19abgncfz', 'ZTF19abgbtla', 'ZTF19abgcbey', 'ZTF19abglmpf', 'ZTF19abfvhlx', 'ZTF19abgjlef', 'ZTF18abtlqoh', 'ZTF19abfvfxp', 'ZTF19abgrhzk', 'ZTF18aarwxum', 'ZTF19abglmpf', 'ZTF19abgrhzk', 'ZTF19abfwfiq', 'ZTF19abgdcyx', 'ZTF19abglmpf', 'ZTF19abgrcxs', 'ZTF19abgmkef', 'ZTF19abfzzxy', 'ZTF19abgbdcp', 'ZTF19abfjjwc', 'ZTF19abgncfz', 'ZTF19abgbtla', 'ZTF19abfdupx', 'ZTF19abglmpf', 'ZTF19aambfxc', 'ZTF19abfdupx', 'ZTF19abeeqoj', 'ZTF19aambfxc', 'ZTF19abfdupx', 'ZTF19abgctni', 'ZTF19abgjfoj', 'ZTF19abhbkdd', 'ZTF19abgpnge', 'ZTF19abgcbey', 'ZTF19abgppki', 'ZTF19abgpnge', 'ZTF19abfrydu', 'ZTF19abgppki', 'ZTF19abgpjed', 'ZTF19abgbtla', 'ZTF19abglmpf', 'ZTF19abflufo', 'ZTF19abghldi', 'ZTF18aarwxum', 'ZTF19abglmpf', 'ZTF19abgqksj', 'ZTF19abgpjed', 'ZTF19abheryh', 'ZTF19abgbtla', 'ZTF19abfdupx', 'ZTF19abghldi', 'ZTF19abglmpf', 'ZTF19abgdcyx', 'ZTF19abghldi', 'ZTF19abglmpf', 'ZTF19abgmkef', 'ZTF19abgrcxs', 'ZTF19abfwhja', 'ZTF19abgbdcp', 'ZTF19abfjjwc', 'ZTF19abgffaj', 'ZTF19abgjfoj', 'ZTF19abfqmqz', 'ZTF19abfvhlx', 'ZTF19abhzdjp', 'ZTF19abiaxpi', 'ZTF19abgvfst', 'ZTF19abglmpf', 'ZTF18aarwxum', 'ZTF19abiiqcu', 'ZTF18aarwxum', 'ZTF19abgjlef', 'ZTF19abicwzc', 'ZTF19abgjlef', 'ZTF19abgqksj', 'ZTF19abhjhes', 'ZTF19abfjjwc', 'ZTF19abhceez', 'ZTF19abheryh', 'ZTF19abgrhzk', 'ZTF19abhenjb', 'ZTF19abgdcyx', 'ZTF19abgrhzk', 'ZTF19abhenjb', 'ZTF19abgbdcp', 'ZTF18aamasph', 'ZTF19abgncfz', 'ZTF19abgdcyx', 'ZTF19abglmpf', 'ZTF19abgpydp', 'ZTF19abglmpf', 'ZTF19abghldi', 'ZTF19abgpydp', 'ZTF19abgctni', 'ZTF19abidfag', 'ZTF19abipnwp', 'ZTF19abfvhlx', 'ZTF19abipktm', 'ZTF19abfqmqz', 'ZTF19abgfnmt', 'ZTF19abipnwp', 'ZTF19abfzwpe', 'ZTF19abiptub', 'ZTF19abfvhlx', 'ZTF19abgfnmt', 'ZTF19abiqpux', 'ZTF19abiszzn', 'ZTF19abgpjed', 'ZTF19abhzdjp', 'ZTF19abgdcyx', 'ZTF19aambfxc', 'ZTF19abgcbey', 'ZTF19abghldi', 'ZTF19abgvfst', 'ZTF19abhzdjp', 'ZTF19abfrydu', 'ZTF19abgncfz', 'ZTF19abiszzn', 'ZTF19abgpjed', 'ZTF19abggmrt', 'ZTF19abgppki', 'ZTF19abiietd', 'ZTF19abgcbey', 'ZTF19abglmpf', 'ZTF19abgqksj', 'ZTF19abgjlef', 'ZTF19abgjoth', 'ZTF19abidbya', 'ZTF19abgmkef', 'ZTF19abhjhes', 'ZTF19abgrcxs', 'ZTF19abgqksj', 'ZTF19abgqksj', 'ZTF19abgmkef', 'ZTF19abhjhes', 'ZTF19abgjlef', 'ZTF19abgrcxs', 'ZTF19abfwhja', 'ZTF19abgncfz', 'ZTF19abfjjwc', 'ZTF19abgvfst', 'ZTF19abglmpf', 'ZTF19abgvfst', 'ZTF19abfdupx', 'ZTF19abjgbgc', 'ZTF19abgmjtu', 'ZTF19abjgdko', 'ZTF19abicvxs', 'ZTF19aambfxc', 'ZTF18aarwxum', 'ZTF18aarwxum', 'ZTF19abglmpf', 'ZTF19abiijov', 'ZTF18aarwxum', 'ZTF19abfdupx', 'ZTF18aarwxum', 'ZTF19abglmpf', 'ZTF19abgpydp', 'ZTF19abgctni', 'ZTF19abipkyb', 'ZTF19abgjfoj', 'ZTF19abgffaj', 'ZTF19abiptub', 'ZTF19abfzwpe', 'ZTF19abjnzsi', 'ZTF19abiovhj', 'ZTF19abhdxcs', 'ZTF19abgjfoj', 'ZTF19abidfag', 'ZTF19abgmmfu', 'ZTF19abipmfl', 'ZTF19abisbgx', 'ZTF19abgrcxs', 'ZTF19aaqfrrl', 'ZTF19abfzzxy', 'ZTF19abhdlxp', 'ZTF19abhdvme', 'ZTF19abhusrq', 'ZTF19abisbgx', 'ZTF19abhdlxp', 'ZTF19abjravi', 'ZTF19abiietd', 'ZTF19abhenjb', 'ZTF19abgbdcp', 'ZTF19abglmpf', 'ZTF19abfdupx', 'ZTF18aarwxum', 'ZTF19abgppki', 'ZTF19abgpnge', 'ZTF19abiietd', 'ZTF19abgpjed', 'ZTF19abgbdcp', 'ZTF19abgncfz', 'ZTF19abhzdjp', 'ZTF19abjibet', 'ZTF19abiszzn', 'ZTF19abgppki', 'ZTF19abhoyxd', 'ZTF19abiaxpi', 'ZTF19abfvhlx', 'ZTF19abidbya', 'ZTF19abiptrq', 'ZTF19abgjlef', 'ZTF19abgqksj', 'ZTF19abgmkef', 'ZTF19abhjhes', 'ZTF19abidbya', 'ZTF19abfvhlx', 'ZTF19abiovhj', 'ZTF19abipktm', 'ZTF19abkcbri', 'ZTF19abgjlef', 'ZTF19abkfmjp', 'ZTF19abkfxfb', 'ZTF19abhdxcs', 'ZTF19abgrcxs', 'ZTF19abgafkt', 'ZTF19abgmkef', 'ZTF19abhjhes', 'ZTF19abkfxfb', 'ZTF19abjgdko', 'ZTF19abhdxcs', 'ZTF18aarwxum', 'ZTF18aarwxum', 'ZTF19abkcbri', 'ZTF19ablesob', 'ZTF19abidbya', 'ZTF19abgmmfu', 'ZTF19abipmfl', 'ZTF19abjnzsi', 'ZTF19abiqqve', 'ZTF19abgmjtu', 'ZTF19abicvxs', 'ZTF19abjgbwx', 'ZTF19abjgazt', 'ZTF19abjgdxx', 'ZTF19abjgdko', 'ZTF19ablesob', 'ZTF19abgdcyx', 'ZTF19abjibet', 'ZTF19abgvfst', 'ZTF19abghldi', 'ZTF18aarwxum', 'ZTF19abjibet', 'ZTF19abfdupx', 'ZTF19abghldi', 'ZTF18aarwxum', 'ZTF19abglmpf', 'ZTF19abgctni', 'ZTF19abgpydp', 'ZTF19abgmmfu', 'ZTF19abipmfl', 'ZTF19abidfag', 'ZTF19abjnzsi', 'ZTF19abiovhj', 'ZTF19abiovhj', 'ZTF19abiopky', 'ZTF19abgmmfu', 'ZTF19abgpydp', 'ZTF19abiptub', 'ZTF19abhusrq', 'ZTF19abgrcxs', 'ZTF19abisbgx', 'ZTF19abhdvme', 'ZTF19abhdvme', 'ZTF19abjravi', 'ZTF19abhdlxp', 'ZTF19abhdvme', 'ZTF19abhusrq', 'ZTF19abirbnk', 'ZTF19abiqpux', 'ZTF19ablovot', 'ZTF19abjpick', 'ZTF19aaqfrrl', 'ZTF19abgppki', 'ZTF19abhbtlo', 'ZTF18aawfquu', 'ZTF19abhzdjp', 'ZTF19abjioie', 'ZTF19abgcbey', 'ZTF19abhzdjp', 'ZTF19abiszzn', 'ZTF19abgppki', 'ZTF19abiietd', 'ZTF19abfdupx', 'ZTF18aarwxum', 'ZTF19abglmpf', 'ZTF19abjibet', 'ZTF18aarwxum', 'ZTF19abfdupx', 'ZTF18aarwxum', 'ZTF19abglmpf', 'ZTF19abfvhlx', 'ZTF19abipktm', 'ZTF19abfvhlx', 'ZTF19abgqksj', 'ZTF19abiyyun', 'ZTF19abidbya', 'ZTF19abgjlef', 'ZTF19abgmkef', 'ZTF19abhjhes', 'ZTF19abiztbh', 'ZTF19abgrcxs', 'ZTF19ablesob', 'ZTF19abgjlef', 'ZTF19abgmkef', 'ZTF19abhjhes', 'ZTF19abkdeae', 'ZTF19ablesob', 'ZTF19abgrcxs', 'ZTF19abhjhes', 'ZTF19abjpimw', 'ZTF19ablesob', 'ZTF19abjgaye', 'ZTF19abjgbgc', 'ZTF19abjgcad', 'ZTF19abgmjtu', 'ZTF19abjgdko', 'ZTF19abkfxfb', 'ZTF19abkfmjp', 'ZTF19abgmjtu', 'ZTF19abkfxfb', 'ZTF19abjgdko', 'ZTF19abgncfz', 'ZTF19abjibet', 'ZTF18aarwxum', 'ZTF19abglmpf', 'ZTF19abgdcyx', 'ZTF19abjibet', 'ZTF18aarwxum', 'ZTF18aarwxum', 'ZTF19abglmpf', 'ZTF19abjmnlw', 'ZTF19abirbnk', 'ZTF19abhusrq', 'ZTF19abhdxcs', 'ZTF19abhusrq', 'ZTF19abjpord', 'ZTF19abiqpux', 'ZTF19ablovot', 'ZTF19abjqytt', 'ZTF19abisbgx', 'ZTF19ablovot', 'ZTF19abgrcxs', 'ZTF19aaqfrrl', 'ZTF19abhoyxd', 'ZTF19abgppki', 'ZTF19abiszzn', 'ZTF19abfrydu', 'ZTF19abhzdjp', 'ZTF19abjioie', 'ZTF19abgcbey', 'ZTF19abjioie', 'ZTF19abhzdjp', 'ZTF19abgncfz', 'ZTF19abgdcyx', 'ZTF19abjibet', 'ZTF18aarwxum', 'ZTF18aarwxum', 'ZTF19abglmpf', 'ZTF19abjibet', 'ZTF18aarwxum', 'ZTF19abljkea', 'ZTF18aarwxum', 'ZTF19abglmpf', 'ZTF19abljinr', 'ZTF19abfvhlx', 'ZTF19abipktm', 'ZTF19ablesob', 'ZTF19abgjlef', 'ZTF19abgmkef', 'ZTF19abhjhes', 'ZTF19abkdlkl', 'ZTF19abgjlef', 'ZTF19abgmkef', 'ZTF19abhjhes', 'ZTF19abkdeae', 'ZTF19ablesob', 'ZTF19abhjhes', 'ZTF19abjibet', 'ZTF18aarwxum', 'ZTF19abglmpf', 'ZTF19abgdcyx', 'ZTF19abjibet', 'ZTF19abglmpf', 'ZTF19abixauz', 'ZTF19abipnwp', 'ZTF19abkevcb', 'ZTF19abipkyb', 'ZTF19abixauz', 'ZTF19abkfmjp', 'ZTF19abiqifg', 'ZTF19abkdeae', 'ZTF19ablesob', 'ZTF19ablybjv', 'ZTF19abgdcyx', 'ZTF19abjibet', 'ZTF18aarwxum', 'ZTF19abljkea', 'ZTF19abglmpf', 'ZTF19abgdcyx', 'ZTF19abjibet', 'ZTF18aarwxum', 'ZTF19abglmpf', 'ZTF19abljinr', 'ZTF19abgctni', 'ZTF19abgcbey', 'ZTF19abjibet', 'ZTF19abgvfst', 'ZTF18aarwxum', 'ZTF19abghldi', 'ZTF19abglmpf'))
    # nuclear_transients_210819 = np.unique(('ZTF19aapreis', 'ZTF18abxftqm', "ZTF18abccpuw", "ZTF19abglmpf", "ZTF19ablovot", "ZTF18acvqugw", "ZTF18acrttfl", "ZTF18acvqugw", "ZTF18acrttfl", "ZTF18acvqugw", "ZTF18acrttfl", "ZTF18acvqugw", "ZTF18acrttfl", "ZTF18acvqugw", "ZTF18acrttfl", "ZTF18acvqugw", "ZTF18acrttfl", "ZTF18acvqugw", "ZTF18acrttfl", "ZTF18acvqugw", "ZTF18acrttfl", "ZTF18acvqugw", "ZTF18acrttfl", "ZTF18acvqugw", "ZTF18acrttfl", "ZTF18acvqugw", "ZTF18acrttfl", "ZTF18acvqugw", "ZTF18acrttfl", "ZTF18acvqugw", "ZTF18acrttfl", "ZTF18acvqugw", "ZTF18acrttfl", "ZTF18acvqugw", "ZTF18acrttfl", "ZTF18acvqugw", "ZTF18acrttfl", "ZTF18acvqugw", "ZTF18acrttfl", "ZTF18acvqugw", "ZTF18acrttfl", "ZTF18acvqugw", "ZTF18acrttfl", "ZTF18acvqugw", "ZTF18acrttfl", "ZTF19abjmnlw", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aanxnrz", "ZTF18aanxnrz", "ZTF18aaojsvk", "ZTF18abvbcln", "ZTF19ablovot", "ZTF19ablovot", "ZTF19ablovot", "ZTF19ablovot", "ZTF19ablovot", "ZTF18abmogca",))
    # nuclear_transients_210819 = np.unique(np.concatenate((nuclear_transients_080819, nuclear_transients_210819)))
    # classify_lasair_light_curves(object_names=nuclear_transients_210819, figdir='astrorapid/nuclear_transients_210819')

    # nuc_tran_plot = np.load('plot_nuclear_transients_210819.npz')
    # real_tde_probs = nuc_tran_plot['real_tde_probs']
    # probabilities = nuc_tran_plot['probabilities']
    # names = nuc_tran_plot['names']
    # radecs = nuc_tran_plot['radec']
    # peakmags = nuc_tran_plot['peakmags']
    # objids = nuc_tran_plot['objids']
    #
    # # classify_lasair_light_curves(object_names=objids[probabilities > 0.4486], figdir='astrorapid/nuclear_transients_210819')
    # np.savetxt('/Users/danmuth/PycharmProjects/astrorapid/paper_plots/TDE_candidates_210819.csv', np.hstack([radecs, peakmags, np.array([probabilities, objids]).T])[probabilities > 0.4486], fmt='%s')
    #
    # import matplotlib.pyplot as plt
    # plt.rcParams['xtick.labelsize'] = 15
    # plt.rcParams['ytick.labelsize'] = 15
    #
    # plt.figure()
    # for i in range(len(real_tde_probs)):
    #     plt.text(real_tde_probs[i]-0.025, 64, names[i], rotation=90, color='red')
    #     plt.axvline(x=real_tde_probs[i], color='red', linewidth=0.7)
    #
    # import seaborn as sns
    # sns.set_style("white")
    # print(probabilities)
    # sns.distplot(probabilities, color='dodgerblue', label='Nuclear Transients in past 30 days (from 21-Aug-2019)', kde=False, norm_hist=False)
    # plt.xlim(0, 1.)
    #
    # plt.ylabel("Number of transients", fontsize=15)
    # plt.xlabel("TDE Classification probability", fontsize=15)
    # plt.ylim(0, 250)
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig('astrorapid/prob2_unnormalised_210819.pdf')
    #
    # plt.figure()
    # for i in range(len(real_tde_probs)):
    #     plt.text(real_tde_probs[i]-0.02, 180, names[i], rotation=90, color='red')
    #     plt.axvline(x=real_tde_probs[i], color='red', linewidth=0.7)
    #
    # plt.hist(probabilities, 100, cumulative=-1, color='dodgerblue', histtype='step', linewidth=1.5)
    # plt.xlim(0.3, 1.0)
    # plt.ylim(0, 200)
    # plt.xlabel("TDE Classification probability, $p$", fontsize=15)
    # plt.ylabel("Number of objects with at least $p$ probability", fontsize=15)
    # plt.tight_layout()
    # plt.savefig('astrorapid/cumulativeTDE_210819.pdf')

    # # Type Ia SNe in ZTF from OSC
    # snia_names = []
    # import json
    # with open('ZTF_SNIa_osc_12092019.json') as json_file:
    #     data = json.load(json_file)
    # for i, sn in enumerate(data):
    #     try:
    #         snia_name = [name for name in data[i]['Name'].split() if 'ZTF' in name][0]
    #         snia_z = float(data[i]['z'].split()[0])
    #         snia_names.append((snia_name, snia_z))
    #         print(i, snia_name, snia_z)
    #     except IndexError as e:
    #         print(f"failed on {i} {data[i]['Name']}")
    #
    # classify_lasair_light_curves(object_names=snia_names, figdir='astrorapid/real_snia_from_osc/newmodel_noAGN_wredshift_batch500')
    #



#LASAIR QUERY for nuclear transients

#SELECT DISTINCT o.objectId FROM objects o, candidates c WHERE o.sherlock_classification IN ('NT') AND o.jdmin > JDNOW() - 30 AND o.ncand > 3 AND c.objectId = o.objectId AND (c.jd > JDNOW() - 30) AND c.magpsf < 20 AND c.rb >= 0.75 AND c.nbad = 0 AND c.isdiffpos = 't' AND c.fwhm <= 5 AND ABS(c.magdiff) <= 0.1 AND c.elong <= 1.2

#SELECT DISTINCT o.objectId FROM objects o, candidates c WHERE o.sherlock_classification IN ('NT') AND o.jdmin > JDNOW() - 200 AND o.jdmin <= JDNOW() - 100 AND o.ncand > 3 AND c.objectId = o.objectId AND (c.jd > JDNOW() - 200 AND c.jd <= JDNOW() - 100) AND c.magpsf < 20 AND c.rb >= 0.75 AND c.nbad = 0 AND c.isdiffpos = 't' AND c.fwhm <= 5 AND ABS(c.magdiff) <= 0.1 AND c.elong <= 1.2

#SELECT
#objects.objectId
#FROM
#objects, candidates, sherlock_classifications
#WHERE ORDER
# sherlock_classifications.classification IN ("NT")
#         AND objects.ncand > 3
#         AND candidates.objectId = objects.objectId
#         AND candidates.magpsf < 20
#         AND candidates.rb >= 0.75
#         AND candidates.nbad = 0
#         AND candidates.isdiffpos = "t"
#         AND candidates.fwhm <= 5
#         AND ABS(candidates.magdiff) <= 0.1
#         AND candidates.elong <= 1.2
#         AND objects.jdmin > JDNOW() - 30
#         AND candidates.jd > JDNOW() - 30

#WHERE o.sherlock_classification IN ('NT')
# AND o.jdmin > JDNOW() - 200 AND o.jdmin <= JDNOW() - 100
# AND o.ncand > 3
# AND c.objectId = o.objectId
# AND (c.jd > JDNOW() - 200 AND c.jd <= JDNOW() - 100)
# AND c.magpsf < 20
# AND c.rb >= 0.75
# AND c.nbad = 0
# AND c.isdiffpos = 't'
# AND c.fwhm <= 5
# AND ABS(c.magdiff) <= 0.1
# AND c.elong <= 1.2

#WHERE o.sherlock_classification IN ('NT')
# AND o.jdmin > JDNOW() - 30
# AND o.ncand > 3
# AND c.objectId = o.objectId
# AND (c.jd > JDNOW() - 30)
# AND c.magpsf < 20
# AND c.rb >= 0.75 AND c.nbad = 0 AND c.isdiffpos = 't' AND c.fwhm <= 5 AND ABS(c.magdiff) <= 0.1 AND c.elong <= 1.2





