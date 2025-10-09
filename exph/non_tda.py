import os
import numpy as np
from yambopy.dbs.excitondb import YamboExcitonDB
from yambopy.dbs.latticedb import YamboLatticeDB
from yambopy.dbs.wfdb import YamboWFDB
from yambopy.bse.rotate_excitonwf import rotate_exc_wf
from yambopy.tools.degeneracy_finder import find_degeneracy_evs
from point_group_ops import get_pg_info, decompose_rep2irrep

def compute_exc_rep(path='.', bse_dir='SAVE', iqpt=1, nstates=-1, degen_tol = 1e-2, degen_rtol=1e-3):
    # Load the lattice database
    lattice = YamboLatticeDB.from_db_file(os.path.join(path, 'SAVE', 'ns.db1'))
    filename = 'ndb.BS_diago_Q%d' % (iqpt)
    excdb = YamboExcitonDB.from_db_file(lattice, filename=filename,
                                             folder=os.path.join(path, bse_dir),
                                             Load_WF=True, neigs=-1)
    # Load the wavefunction database
    wfdb = YamboWFDB(path=path, latdb=lattice,
                      bands_range=[np.min(excdb.table[:, 1]) - 1,
                    np.max(excdb.table[:, 2])])
    Akcv = excdb.get_Akcv()
    Akcv_left = Akcv
    if Akcv.shape[1] == 2:
        if os.path.exists('BS_left_ev_Cache.npy'):
            Akcv_left = np.load('BS_left_ev_Cache.npy')
        else:
            Akcv_left = np.linalg.inv(Akcv.reshape(len(Akcv),-1)).conj().T.reshape(Akcv.shape)
            np.save('BS_left_ev_Cache',Akcv_left)
    #
    eigs = excdb.eigenvalues.real
    sort_idx = np.argsort(eigs)
    eigs = eigs[sort_idx].copy()
    pos_idx = np.where(eigs > 0)
    assert len(pos_idx) >0, "No postive eigenvalues found"
    pos_idx = pos_idx[0][0]
    #
    Ak_r = Akcv[sort_idx][pos_idx-nstates:pos_idx+nstates]
    Ak_l = Akcv_left[sort_idx][pos_idx-nstates:pos_idx+nstates].conj()
    exe_ene = eigs[pos_idx-nstates:pos_idx+nstates]
    #
    degen_idx = find_degeneracy_evs(exe_ene,atol=degen_tol,rtol=degen_rtol)
    uni_eigs = []
    degen_eigs = []
    for i in degen_idx:
        uni_eigs.append(np.mean(exe_ene[i]))
        degen_eigs.append(len(i))
    uni_eigs = np.array(uni_eigs)
    degen_eigs = np.array(degen_eigs,dtype=int)

    excQpt = excdb.car_qpoint
    # Convert the q-point to crystal coordinates
    excQpt = lattice.lat @ excQpt

    lat_vec = lattice.lat
    lat_vec_inv = np.linalg.inv(lat_vec)
    if os.path.exists('Dmat_elec_Cache.npy'):
        dmats = np.load('Dmat_elec_Cache.npy')
    else:
        dmats = wfdb.Dmat()
        np.save('Dmat_elec_Cache',dmats)
    #
    trev_fac = 1 + int(np.rint(lattice.time_rev))
    ## print some data about the degeneracies
    print('=' * 40)
    print('Group theory analysis for Q point : ', excQpt)
    print('*' * 40)

    trace_all_real = []
    trace_all_imag = []
    little_group = []
    #
    for isym in range(len(lattice.sym_car)//trev_fac):
        symm_mat = lattice.sym_car[isym]
        symm_mat_red = lat_vec@symm_mat@lat_vec_inv
        #isym = 2
        Sq_minus_q = np.einsum('ij,j->i', symm_mat_red,
                           excQpt) - excQpt
        #print(Sq_minus_q)
        #diff = Sq_minus_q.copy()
        Sq_minus_q = Sq_minus_q - np.rint(Sq_minus_q)
        ## check if Sq = q
        if np.linalg.norm(Sq_minus_q) > 10**-5: continue
        little_group.append(isym + 1)
        tau_dot_k = 1.0#np.exp(1j * 2 * np.pi *
        #               np.dot(excQpt, frac_trans[isym]))
        #assert(np.linalg.norm(Sq_minus_q)<10**-5)
        rot_Akcv = rotate_exc_wf(Ak_r, symm_mat_red, wfdb.kBZ, excQpt, dmats[isym], False, ktree=wfdb.ktree)
        rep = np.einsum('m...,n...->mn',Ak_l,rot_Akcv,optimize=True)

        #print('Symmetry number : ',isym + 1)
        ## print characters
        irrep_sum = 0
        real_trace = []
        imag_trace = []
        for iirepp in range(len(uni_eigs)):
            idegen = degen_eigs[iirepp]
            idegen2 = irrep_sum + idegen
            trace_tmp = np.trace(rep[irrep_sum:idegen2, irrep_sum:idegen2])
            real_trace.append(trace_tmp.real.round(4))
            imag_trace.append(trace_tmp.imag.round(4))
            irrep_sum = idegen2
        # print('Real : ',real_trace)
        # print('Imag : ',imag_trace)
        trace_all_real.append(real_trace)
        trace_all_imag.append(imag_trace)

    little_group = np.array(little_group, dtype=int)

    pg_label, classes, class_dict, char_tab, irreps = get_pg_info(
        lattice.sym_car[little_group - 1])

    print('Little group : ', pg_label)
    print('Little group symmetries : ', little_group)
    # print class info
    print('Classes (symmetry indices in each class): ')
    req_sym_characters = np.zeros(len(classes), dtype=int)
    class_orders = np.zeros(len(classes), dtype=int)
    for ilab, iclass in class_dict.items():
        print("%16s    : " % (classes[ilab]), little_group[np.array(iclass)])
        req_sym_characters[ilab] = min(iclass)
        class_orders[ilab] = len(iclass)
    print()
    trace_all_real = np.array(trace_all_real)
    trace_all_imag = np.array(trace_all_imag)
    trace = trace_all_real + 1j * trace_all_imag
    trace_req = trace[req_sym_characters, :].T
    print("====== Exciton representations ======")
    print("Energy (eV),  degenercy  : representation")
    print('-' * 40)
    for i in range(len(trace_req)):
        rep_str_tmp = decompose_rep2irrep(trace_req[i], char_tab, len(little_group),
                                          class_orders, irreps)
        print('%.4f        %9d  : ' % (uni_eigs[i], degen_eigs[i]), rep_str_tmp)
    print('*' * 40)



if __name__ == "__main__":
    compute_exc_rep(path='..', bse_dir='GW_BSE', iqpt=1, nstates=3, degen_tol = 1e-2)

