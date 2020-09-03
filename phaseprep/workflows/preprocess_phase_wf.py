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
    6) Correct geometry changes (AFNI issue)
    7) Convert back to phase
    8) Unwrap and detrend data
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
    # modified from fslcomplex to fslmaths in Sep 2020, bonus also preserves geometry info
    convert2real = pe.MapNode(interface=fsl.maths.MultiImageMaths(op_string=' -cos -mul %s'), name='convert2real', iterfield=['in_file','operand_files'])
    convert2imag = pe.MapNode(interface=fsl.maths.MultiImageMaths(op_string=' -sin -mul %s'), name='convert2imag', iterfield=['in_file','operand_files'])

    # 5) Apply magnitude motion correction parameters
    mocoreal = pe.MapNode(interface=afni.Allineate(), name='mocoreal',
                          iterfield=['in_file', 'in_matrix'])
    mocoreal.inputs.outputtype = 'NIFTI_GZ'
    mocoreal.inputs.out_file = 'mocophase.nii.gz'
    mocoreal.inputs.num_threads = 2
    mocoimag = mocoreal.clone('mocoimag')

    # 6) Correct geometry changes (AFNI issue)
    cpgeommocoreal = pe.MapNode(interface=fsl.CopyGeom(), name='cpgeommoco', iterfield=['dest_file', 'in_file'])
    cpgeommocoimag = cpgeommocoreal.clone('cpgeommocoimag')

    # 7) Convert back to phase custom interface to use atan2 and avoid sign ambiguity
    convert2phase = pe.MapNode(interface=pp.Convert2Phase(), name='convert2phase', iterfield=['real_image','imaginary_image'])

    # 8) Remove first volume, unwrap and detrend phase data
    prepphase = pe.MapNode(interface=pp.PreprocessPhase(), name='prepphase', iterfield=['phase'])

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
    preprocphase.connect([(inputspec, img2float, [('input_phase', 'in_file')]), # 1
                          (inputspec, findscaling, [('input_phase', 'in_file')]), # 2
                          (findscaling, convert2rad, [('scaling_arg', 'args')]),
                          (img2float, convert2rad, [('out_file', 'in_file')]),
                          (convert2rad, convert2real, [('out_file', 'in_file')]),
                          (convert2rad, convert2imag, [('out_file', 'in_file')]),
                          (inputspec, convert2real, [('input_mag', 'operand_files')]),
                          (inputspec, convert2imag, [('input_mag', 'operand_files')]),
                          (inputspec, mocoreal, [('motion_par', 'in_matrix')]), # 5 real
                          (convert2real, mocoreal, [('out_file', 'in_file')]),
                          (mocoreal, cpgeommocoreal, [('out_file','dest_file')]), #6 real
                          (img2float, cpgeommocoreal, [('out_file', 'in_file')]),
                          (inputspec, mocoimag, [('motion_par', 'in_matrix')]), # 5 imag
                          (convert2imag, mocoimag, [('out_file', 'in_file')]),
                          (mocoimag, cpgeommocoimag, [('out_file','dest_file')]), # 6 imag
                          (img2float, cpgeommocoimag, [('out_file', 'in_file')]),
                          (cpgeommocoimag, convert2phase, [('out_file', 'imaginary_image')]), # 7
                          (cpgeommocoreal, convert2phase, [('out_file', 'real_image')]),
                          (convert2phase, prepphase, [('phase_image', 'phase')]), # 8
                          (prepphase, maskfunc, [('detrended_phase', 'in_file')]), # 9
                          (inputspec, maskfunc, [('mask_file', 'in_file2')]),
                          (maskfunc, outputspec, [('out_file', 'proc_phase')]),
                          (prepphase, outputspec, [('uw_phase', 'uw_phase')]),
                          (prepphase, outputspec, [('delta_phase', 'delta_phase')]),
                          (inputspec, calcSNR, [('rest', 'rest'), # 10
                                                ('task', 'task')]),
                          (prepphase, calcSNR, [('detrended_phase', 'func')]),
                          (calcSNR, outputspec, [('noise', 'std_phase')])
                          ])

    return preprocphase


if __name__ == "__main__":
    workflow = create_preprocess_phase_wf()
