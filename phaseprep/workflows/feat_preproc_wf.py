import nipype.pipeline.engine as pe
import nipype.interfaces.fsl as fsl
import nipype.interfaces.utility as ul

def getthreshop(thresh):
    if isinstance(thresh[0], list):
        ops=[]
        for i in range(len(thresh)):
            ops.append('-thr %.10f -Tmin -bin' % (0.1 * thresh[i][1]))
        return ops
    else:
        return '-thr %.10f -Tmin -bin' % (0.1 * thresh[0][1])

def pickfirst(files):
    if isinstance(files, list):
        return files[0]
    else:
        return files

def getinormscale(medianvals):
    return ['-mul %.10f' % (10000. / val) for val in medianvals]

def highpasssetup(highpass, TR):
    return '-bptf %d -1 -add ' % (highpass / TR)

def create_feat_preproc_wf(name='feat_preproc'):
    feat = pe.Workflow(name=name)
    feat.config['execution']['remove_unnecessary_outputs'] = False

    # define inputs
    inputspec = pe.Node(ul.IdentityInterface(fields=['in_file', 'highpass', 'TR']),
                        name='inputspec')

    # Get 2 and 98th percentiles
    getthresh = pe.MapNode(interface=fsl.ImageStats(op_string='-p 2 -p 98'),
                           iterfield=['in_file'],
                           name='getthreshold')

    # Threshold the first run of the functional data at 10% of the 98th percentile
    threshold = pe.MapNode(interface=fsl.ImageMaths(out_data_type='char',
                                                 suffix='_thresh'),
                        name='threshold', iterfield=['in_file', 'op_string'])

    # get median value using the mask
    medianval = pe.MapNode(interface=fsl.ImageStats(op_string='-k %s -p 50'),
                           iterfield=['in_file', 'mask_file'],
                           name='medianval')

    # dilate the mask
    dilatemask = pe.MapNode(interface=fsl.ImageMaths(suffix='_dil',
                                                  op_string='-dilF'),
                         name='dilatemask', iterfield=['in_file'])

    # mask the data with dilated mask
    maskfunc = pe.MapNode(interface=fsl.ImageMaths(suffix='_mask',
                                                    op_string='-mas'),
                           iterfield=['in_file', 'in_file2'],
                           name='maskfunc')

    # scale the run to have a median of 10000
    intnorm = pe.MapNode(interface=fsl.ImageMaths(suffix='_intnorm'),
                         iterfield=['in_file', 'op_string'],
                         name='intnorm')

    # get the mean from each run
    meanfunc = pe.MapNode(interface=fsl.ImageMaths(op_string='-Tmean',
                                                    suffix='_mean'),
                           iterfield=['in_file'],
                           name='meanfunc')

    # calculate the cutoff
    calculatehpcutoff = pe.Node(interface=ul.Function(function=highpasssetup, input_names=["highpass", "TR"],
                                  output_names=["op_string"]), name="calculatehpcutoff")

    # highpass the data
    highpass = pe.MapNode(interface=fsl.ImageMaths(suffix='_tempfilt'),
                          iterfield=['in_file','in_file2'],
                          name='highpass')
    highpass.inputs.suffix = '_hpf'

    # outputspec
    outputspec = pe.Node(ul.IdentityInterface(fields=['filtered_functional_data',
                                                      'mean_file','scalingfactor']),
                        name='outputspec')

    feat.connect([(inputspec, getthresh, [('in_file', 'in_file')]),
                  (inputspec, threshold, [('in_file', 'in_file')]),
                  (inputspec, medianval, [('in_file', 'in_file')]),
                  (inputspec, calculatehpcutoff, [('highpass', 'highpass'),
                                         ('TR', 'TR')]),
                  (getthresh, threshold, [(('out_stat', getthreshop), 'op_string')]),
                  (threshold, medianval, [('out_file', 'mask_file')]),
                  (threshold, dilatemask, [('out_file', 'in_file')]),
                  (inputspec, maskfunc, [('in_file', 'in_file')]),
                  (dilatemask, maskfunc, [('out_file', 'in_file2')]),
                  (maskfunc, intnorm, [('out_file', 'in_file')]),
                  (medianval, intnorm, [(('out_stat', getinormscale), 'op_string')]),
                  (intnorm, meanfunc, [('out_file', 'in_file')]),
                  (calculatehpcutoff, highpass, [('op_string', 'op_string')]),
                  (intnorm, highpass, [('out_file', 'in_file')]),
                  (meanfunc, highpass, [('out_file', 'in_file2')]),
                  (highpass, outputspec, [('out_file','filtered_functional_data')]),
                  (meanfunc, outputspec, [(('out_file', pickfirst), 'mean_file')]),
                  (medianval, outputspec, [('out_stat', 'scalingfactor')])
                  ])

    return feat
