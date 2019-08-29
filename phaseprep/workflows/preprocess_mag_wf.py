import nipype.pipeline.engine as pe
import nipype.interfaces.utility as ul
import nipype.interfaces.fsl as fsl
import nipype.interfaces.afni as afni
import phaseprep.interfaces as pp

def pickfirst(files):
    if isinstance(files, list):
        return files[0]
    else:
        return files

def calcrelmotion(in_file):
    import numpy as np
    disp = np.loadtxt(in_file, skiprows=2)
    rel = np.diff(disp)
    rel = np.append(rel, max(rel))
    np.savetxt(str(in_file[:-6])+'_rd.1D', rel, header='# '+str(in_file), fmt='%.4f')
    return str(in_file[:-6])+'_rd.1D'

def wraptuple(filelist):
    fout = []
    print(filelist)
    for f in filelist:
        if isinstance(f, list):
            fsubout = []
            for fi in f:
                fsubout.append((fi, ''))
        fout.append(fsubout)
    print(fout)
    return fout

def create_preprocess_mag_wf():
    preprocmag = pe.Workflow(name="preprocmag")
    preprocmag.config['execution']['remove_unnecessary_outputs'] = False

    # define inputs
    inputspec = pe.Node(ul.IdentityInterface(fields=['input_mag', # raw phase data
                                                     'frac', # BET franction (-f parameter)
                                                     'rest', # volumes of rest in block design
                                                     'task', # volumes of task in block design
                                                     ]),
                        name='inputspec')

    # convert image to float
    img2float = pe.MapNode(interface=fsl.ImageMaths(out_data_type='float', op_string='', suffix='_dtype'),
                           iterfield=['in_file'],
                           name='img2float')

    # motion correct each run
    volreg = pe.MapNode(interface=afni.Volreg(), name='volreg', iterfield='in_file')
    volreg.inputs.outputtype = 'NIFTI_GZ'

    # calculate relative motions
    calcrel = pe.MapNode(ul.Function(['in_file'], ['out_file'], calcrelmotion),
        name='calcrel', iterfield=['in_file'])

    #generate motion plots
    plotmc = pe.MapNode(interface=fsl.PlotMotionParams(), name='plotmc', iterfield='in_file')
    plotmc.inputs.in_source = 'fsl'
    plotmc.iterables = ("plot_type", ['rotations', 'translations', 'displacement'])

    # register each run to first volume of first run
    # A) extract the first volume of the first run
    extract_ref = pe.MapNode(interface=fsl.ExtractROI(t_size=1, t_min=0), name='extract_ref', iterfield=['in_file'])

    # B) registration
    align2first = pe.MapNode(interface=afni.Allineate(), name='align2first', iterfield=['in_file'])
    align2first.inputs.num_threads = 2
    align2first.inputs.out_matrix = 'align2first'

    # merge xfm from moco and first run alignment
    merge_xfm = pe.MapNode(interface=ul.Merge(2), name='merge_xfm', iterfield=['in1', 'in2'])

    # concatenate moco and alignment to run 1
    cat_xfm = pe.MapNode(interface=afni.CatMatvec(oneline=True), name='cat_xfm', iterfield=['in_file'])
    cat_xfm.inputs.out_file = 'concated_xfm.aff12.1D'

    # apply first volume registration and motion correction in a single step
    applyalign = pe.MapNode(interface=afni.Allineate(), name='applyalign', iterfield=['in_file', 'in_matrix'])
    applyalign.inputs.num_threads = 2
    applyalign.inputs.final_interpolation = 'nearestneighbour'
    applyalign.inputs.outputtype = 'NIFTI_GZ'

    # afni messes with the header (unobliques the data) this puts it back
    cpgeommoco = pe.MapNode(interface=fsl.CopyGeom(), name='cpgeommoco', iterfield=['dest_file', 'in_file'])

    # linear detrending prior to SNR calculation
    detrend = pe.MapNode(interface=pp.DetrendMag(), name='detrend', iterfield=['mag'])

    # get the mean functional of run 1 for brain extraction
    meanfunc = pe.MapNode(interface=fsl.ImageMaths(op_string='-Tmean',
                                                suffix='_mean'),
                       name='meanfunc', iterfield=['in_file'])

    # calculate the phase noise (takes in volume of activation, if none provided them assumes resting state)
    calcSNR = pe.MapNode(interface=pp.RestAverage(), name='calcSNR', iterfield=['func', 'rest', 'task'])

    # extract brain with fsl and save the mask
    extractor = pe.Node(interface=fsl.BET(), name="extractor")
    extractor.inputs.mask = True

    # apply the mask to all runs
    maskfunc = pe.MapNode(interface=fsl.ImageMaths(suffix='_bet',
                                                   op_string='-mas'),
                          iterfield=['in_file'],
                          name='maskfunc')

    # outputspec
    outputspec = pe.Node(ul.IdentityInterface(fields=['proc_mag','motion_par',
                                                      'motion_data', 'maxdisp_data' ,
                                                      'motion_plot', 'run_txfm',
                                                      'mask_file','mean_file','snr']),
                        name='outputspec')

    preprocmag = pe.Workflow(name='preprocmag')
    preprocmag.connect([(inputspec, img2float, [('input_mag', 'in_file')]),
                        (img2float, volreg, [('out_file', 'in_file')]),
                        (volreg, extract_ref, [('out_file', 'in_file')]),
                        (extract_ref, align2first, [('roi_file', 'in_file')]),
                        (extract_ref, align2first, [(('roi_file', pickfirst), 'reference')]),
                        (extract_ref, applyalign, [(('roi_file', pickfirst), 'reference')]),
                        (volreg, merge_xfm, [('oned_matrix_save', 'in2')]),
                        (align2first, merge_xfm, [('out_matrix', 'in1')]),
                        (merge_xfm, cat_xfm, [(('out', wraptuple), 'in_file')]),
                        (volreg,applyalign, [('out_file', 'in_file')]),
                        (volreg, calcrel, [('md1d_file', 'in_file')]),
                        (volreg, plotmc, [('oned_file', 'in_file')]),
                        (cat_xfm, applyalign, [('out_file', 'in_matrix')]),
                        (img2float, cpgeommoco, [('out_file', 'in_file')]),
                        (applyalign, cpgeommoco, [('out_file', 'dest_file')]),
                        (cpgeommoco, detrend, [('out_file', 'mag')]),
                        (detrend, meanfunc, [('detrended_mag', 'in_file')]),
                        (inputspec, calcSNR, [('rest', 'rest'),
                                              ('task', 'task')]),
                        (detrend, calcSNR, [('detrended_mag', 'func')]),
                        (inputspec, extractor, [('frac', 'frac')]),
                        (meanfunc, extractor, [(('out_file', pickfirst), 'in_file')]),
                        (cpgeommoco, maskfunc, [('out_file', 'in_file')]),
                        (extractor, maskfunc, [('mask_file', 'in_file2')]),
                        (maskfunc, outputspec, [('out_file', 'proc_mag')]),
                        (volreg, outputspec, [('oned_matrix_save', 'motion_par')]),
                        (volreg, outputspec, [('oned_file', 'motion_data')]),
                        (volreg, outputspec, [('md1d_file', 'maxdisp_data')]),
                        (plotmc, outputspec, [('out_file', 'motion_plot')]),
                        (cat_xfm, outputspec, [('out_file', 'run_txfm')]),
                        (extractor, outputspec, [('mask_file', 'mask_file')]),
                        (extractor, outputspec, [('out_file', 'mean_file')]),
                        (calcSNR, outputspec, [('tsnr', 'snr')]),
                        ])

    return preprocmag

if __name__=="__main__":
    workflow=create_preprocess_mag_wf()
