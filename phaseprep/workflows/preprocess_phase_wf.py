import nipype.pipeline.engine as pe
import nipype.interfaces.fsl as fsl
import nipype.interfaces.afni as afni
import phaseprep.interfaces as pp
import nipype.interfaces.utility as ul


def findscalingarg(in_file, bit_depth=12):
    import nibabel as nb
    import numpy as np
    img = nb.load(in_file)
    if img.dataobj.slope != 1.0:
        print('Removing rescale before conversion')
    mul = np.pi/(2**(bit_depth-1)*img.dataobj.slope)
    sub = np.pi*((img.dataobj.slope+1)/(2**(bit_depth-1)*img.dataobj.slope))
    return '-mul %s -sub %s' % (mul, sub)


def create_preprocess_phase_wf():
    """Create's phase preprocessing workflow with the following steps:

    1) Convert data to float
    2) Determine scaling required for radians
    3) Apply radian scaling
    4) Convert to real and imaginary
    5) Apply magnitude motion correction parameters
    6) Convert back to phase
    7) Unwrap and detrend data
    8) Correct geometry changes (AFNI issue)
    9) Mask data using magnitude mask
    10) Calculate noise from data

    """
    preprocphase = pe.Workflow(name="preprocphase")
    preprocphase.config['execution']['remove_unnecessary_outputs'] = False

    # define inputs
    inputspec = pe.Node(ul.IdentityInterface(fields=['input_phase', # raw phase data
                                                     'input_mag', # raw mag data
                                                     'motion_par', # afni transform concatenated from magnitude data
                                                     'mask_file', # bet mask from magnitude data
                                                     'siemensbool', # true if data is in siemens units (0,4095)
                                                     'rest', # volumes of rest in block design
                                                     'task', # volumes of task in block design
                                                     ]),
                        name='inputspec')

    # 1) Convert data to float
    img2float = pe.MapNode(interface=fsl.ImageMaths(out_data_type='float', op_string='', suffix='_dtype'),
                           iterfield=['in_file'],
                           name='img2float')

    # 2) Determine radian scaling required
    findscaling = pe.MapNode(interface=ul.Function(input_names=['in_file'],
                                                   output_names=['scaling_arg'],
                                                   function=findscalingarg),
                             name='findscaling', iterfield=['in_file'])

    # 3) Apply radian scaling
    convert2rad = pe.MapNode(interface=fsl.maths.MathsCommand(),
                             name='convert2rad', iterfield=['in_file', 'args'])

    # 4) Convert to real and imaginary (2 step process)
    makecomplex = pe.MapNode(interface=fsl.Complex(complex_polar=True), name='makecomplex',
                             iterfield=['magnitude_in_file', 'phase_in_file'])

    splitcomplex = pe.MapNode(interface=fsl.Complex(real_cartesian=True), name='splitcomplex',
                              iterfield=['complex_in_file'])

    # 5) Apply magnitude motion correction parameters
    mocoreal = pe.MapNode(interface=afni.Allineate(), name='mocoreal',
                          iterfield=['in_file', 'in_matrix'])
    mocoreal.inputs.outputtype = 'NIFTI_GZ'
    mocoreal.inputs.out_file = 'mocophase.nii.gz'
    mocoreal.inputs.num_threads = 2
    mocoimag = mocoreal.clone('mocoimag')

    # 6) Convert back to phase (2 step process)
    makecomplexmocomplex = pe.MapNode(interface=fsl.Complex(complex_cartesian=True), name='makecomplexmoco',
                                      iterfield=['real_in_file', 'imaginary_in_file'])

    splitcomplexmoco = pe.MapNode(interface=fsl.Complex(real_polar=True), name='splitcomplexmoco',
                                  iterfield=['complex_in_file'])

    # 7) Remove first volume, unwrap and detrend phase data
    prepphase = pe.MapNode(interface=pp.PreprocessPhase(), name='prepphase', iterfield=['phase'])

    # 8) Correct geometry changes (AFNI issue)
    cpgeommoco = pe.MapNode(interface=fsl.CopyGeom(), name='cpgeommoco', iterfield=['dest_file', 'in_file'])

    # 9) Mask data using magnitude mask
    maskfunc = pe.MapNode(interface=fsl.ImageMaths(suffix='_bet',
                                                   op_string='-mas'),
                          iterfield=['in_file'],
                          name='maskfunc')
    # 10) Calculate noise from data
    calcSNR = pe.MapNode(interface=pp.RestAverage(), name='calcSNR', iterfield=['func', 'rest', 'task'])

    # outputspec
    outputspec = pe.Node(ul.IdentityInterface(fields=['proc_phase', 'uw_phase', 'delta_phase','std_phase']),
                         name='outputspec')

    preprocphase = pe.Workflow(name='preprocphase')
    preprocphase.connect([(inputspec, img2float, [('input_phase', 'in_file')]),
                          (inputspec, findscaling, [('input_phase', 'in_file')]),
                          (findscaling, convert2rad, [('scaling_arg', 'args')]),
                          (img2float, convert2rad, [('out_file', 'in_file')]),
                          (convert2rad, makecomplex, [('out_file', 'phase_in_file')]),
                          (inputspec, makecomplex, [('input_mag', 'magnitude_in_file')]),
                          (makecomplex, splitcomplex, [('complex_out_file', 'complex_in_file')])
                          # (inputspec, moco, [('motion_par', 'in_matrix')]),
                          # (prepphase, moco, [('detrended_phase', 'in_file')]),
                          # (img2float, cpgeommoco, [('out_file', 'in_file')]),
                          # (moco, cpgeommoco, [('out_file', 'dest_file')]),
                          # (cpgeommoco, maskfunc, [('out_file', 'in_file')]),
                          # (inputspec, maskfunc, [('mask_file', 'in_file2')]),
                          # (maskfunc, outputspec,[('out_file', 'proc_phase')]),
                          # (prepphase, outputspec, [('uw_phase', 'uw_phase')]),
                          # (prepphase, outputspec, [('delta_phase', 'delta_phase')]),
                          # (inputspec, calcSNR, [('rest', 'rest'),
                          #                       ('task', 'task')]),
                          # (prepphase, calcSNR, [('detrended_phase', 'func')]),
                          # (calcSNR, outputspec, [('noise', 'std_phase')])
                          ])

    return preprocphase


if __name__ == "__main__":
    workflow = create_preprocess_phase_wf()
