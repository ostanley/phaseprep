import nipype.pipeline.engine as pe
import nipype.interfaces.fsl as fsl
import nipype.interfaces.afni as afni
import phaseprep.interfaces as pp
import nipype.interfaces.utility as ul

def create_preprocess_phase_wf():
    preprocphase = pe.Workflow(name="preprocphase")
    preprocphase.config['execution']['remove_unnecessary_outputs'] = False

    # define inputs
    inputspec = pe.Node(ul.IdentityInterface(fields=['input_phase', # raw phase data
                                                     'motion_par', # afni transform concatenated from magnitude data
                                                     'mask_file', # bet mask from magnitude data
                                                     'siemensbool', # true if data is in siemens units (0,4095)
                                                     'rest', # volumes of rest in block design
                                                     'task', # volumes of task in block design
                                                     ]),
                        name='inputspec')

    # convert image to float
    img2float = pe.MapNode(interface=fsl.ImageMaths(out_data_type='float', op_string='', suffix='_dtype'),
                           iterfield=['in_file'],
                           name='img2float')

    # prepare phase (first volume substraction, temporal unwraping, linear detrending)
    prepphase = pe.MapNode(interface=pp.PreprocessPhase(), name='prepphase', iterfield=['phase'])
    prepphase.inputs.bit_depth=12

    # motion correct each run
    moco = pe.MapNode(interface=afni.Allineate(), name='moco', iterfield=['in_file', 'in_matrix'])
    moco.inputs.outputtype = 'NIFTI_GZ'
    moco.inputs.out_file = 'mocophase.nii.gz'
    moco.inputs.num_threads = 2

    # afni messes with the header (unobliques the data) this puts it back
    cpgeommoco = pe.MapNode(interface=fsl.CopyGeom(), name='cpgeommoco', iterfield=['dest_file', 'in_file'])

    # apply the mask to all runs
    maskfunc = pe.MapNode(interface=fsl.ImageMaths(suffix='_bet',
                                                   op_string='-mas'),
                          iterfield=['in_file'],
                          name='maskfunc')

    # calculate the phase noise (takes in volume of activation, if none provided them assumes resting state)
    calcSNR = pe.MapNode(interface=pp.RestAverage(), name='calcSNR', iterfield=['func', 'rest', 'task'])
    # outputspec
    outputspec = pe.Node(ul.IdentityInterface(fields=['proc_phase', 'uw_phase', 'delta_phase','std_phase']),
                         name='outputspec')

    preprocphase = pe.Workflow(name='preprocphase')
    preprocphase.connect([(inputspec, img2float, [('input_phase', 'in_file')]),
                          (inputspec, prepphase, [('siemensbool','siemens')]),
                          (img2float, prepphase, [('out_file', 'phase')]),
                          (inputspec, moco, [('motion_par', 'in_matrix')]),
                          (prepphase, moco, [('detrended_phase', 'in_file')]),
                          (img2float, cpgeommoco, [('out_file', 'in_file')]),
                          (moco, cpgeommoco, [('out_file', 'dest_file')]),
                          (cpgeommoco, maskfunc, [('out_file', 'in_file')]),
                          (inputspec, maskfunc, [('mask_file', 'in_file2')]),
                          (maskfunc, outputspec,[('out_file', 'proc_phase')]),
                          (prepphase, outputspec, [('uw_phase', 'uw_phase')]),
                          (prepphase, outputspec, [('delta_phase', 'delta_phase')]),
                          (inputspec, calcSNR, [('rest', 'rest'),
                                                ('task', 'task')]),
                          (prepphase, calcSNR, [('detrended_phase', 'func')]),
                          (calcSNR, outputspec, [('noise', 'std_phase')])
                          ])

    return preprocphase


if __name__ == "__main__":
    workflow = create_preprocess_phase_wf()
